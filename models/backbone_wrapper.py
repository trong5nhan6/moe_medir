import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG, BACKBONE_REGISTRY


def _set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


class BackboneWrapper(nn.Module):
    """
    Wraps a pretrained backbone for end-to-end fine-tuning.

    forward(imgs) → [B, feature_dim]
    Use freeze_all() then unfreeze_partial() for 2-stage training.
    """

    def __init__(self, backbone_name: str, device):
        super().__init__()
        self.backbone_name = backbone_name
        self.info          = BACKBONE_REGISTRY[backbone_name]
        self.loader_type   = self.info["loader"]
        self.model         = self._load(device)

    def _load(self, device):
        info   = self.info
        loader = self.loader_type

        if loader == "open_clip":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                info["model_name"], pretrained=info["pretrained"])
            return model.to(device)

        elif loader == "open_clip_hub":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(info["model_name"])
            return model.to(device)

        elif loader == "hf_dinov2":
            from transformers import AutoModel
            return AutoModel.from_pretrained(info["model_name"]).to(device)

        elif loader == "torchvision_cnn":
            import torchvision.models as tvm
            model_fn = getattr(tvm, info["model_name"])
            return model_fn(weights=info["pretrained"]).to(device)

        raise ValueError(f"Unknown loader: {loader}")

    def freeze_all(self):
        _set_requires_grad(self.model, False)

    def unfreeze_partial(self, n_blocks: int = 2):
        """
        Unfreeze last n_blocks transformer blocks (ViT) or CNN stages.
        CNN: n_blocks=0 unfreezes all; otherwise last n_blocks stages of features.
        ViT: last n_blocks + final norm only.
        """
        loader = self.loader_type

        if loader == "torchvision_cnn":
            if n_blocks == 0:
                _set_requires_grad(self.model, True)
            else:
                for stage in list(self.model.features.children())[-n_blocks:]:
                    _set_requires_grad(stage, True)

        elif loader in ("open_clip", "open_clip_hub"):
            vis = self.model.visual
            if hasattr(vis, "trunk"):
                # BiomedCLIP — TimmModel (timm ViT)
                for blk in vis.trunk.blocks[-n_blocks:]:
                    _set_requires_grad(blk, True)
                _set_requires_grad(vis.trunk.norm, True)
            else:
                # Standard CLIP ViT
                for blk in vis.transformer.resblocks[-n_blocks:]:
                    _set_requires_grad(blk, True)
                _set_requires_grad(vis.ln_post, True)

        elif loader == "hf_dinov2":
            for layer in self.model.encoder.layer[-n_blocks:]:
                _set_requires_grad(layer, True)
            _set_requires_grad(self.model.layernorm, True)

    def backbone_parameters(self):
        """Returns unfrozen backbone parameters (for differential LR optimizer)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        loader = self.loader_type

        if loader in ("open_clip", "open_clip_hub"):
            vis = self.model.visual
            if hasattr(vis, "trunk"):
                # BiomedCLIP — TimmModel
                trunk = vis.trunk
                x = trunk.patch_embed(imgs)
                x = trunk._pos_embed(x)
                if hasattr(trunk, "norm_pre"):
                    x = trunk.norm_pre(x)
                x = trunk.blocks(x)
                x = trunk.norm(x)
            else:
                # Standard CLIP ViT
                x = vis.conv1(imgs)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls_tok = vis.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([cls_tok, x], dim=1)
                x = x + vis.positional_embedding
                x = vis.ln_pre(x)
                x = vis.transformer(x)
                x = vis.ln_post(x)
            cls        = x[:, 0, :]
            patch_mean = x[:, 1:, :].mean(dim=1)

        elif loader == "hf_dinov2":
            outputs    = self.model(pixel_values=imgs)
            hidden     = outputs.last_hidden_state
            cls        = hidden[:, 0, :]
            patch_mean = hidden[:, 1:, :].mean(dim=1)

        elif loader == "torchvision_cnn":
            feat_map   = self.model.features(imgs)
            cls        = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
            patch_mean = F.adaptive_max_pool2d(feat_map, 1).flatten(1)

        else:
            raise ValueError(f"Unknown loader: {loader}")

        if CFG.feature_mode == "cls":
            return cls
        return torch.cat([cls, patch_mean], dim=-1)
