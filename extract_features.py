"""
Step 1 — Run ONCE per backbone before training.

Loads the backbone (frozen), iterates over all 4 MedMNIST datasets × 3 splits,
extracts features and saves them as .npy files.

Supported backbones (set in config.py → backbone):
  clip_vitb32    CLIP ViT-B/32 (OpenAI)
                   CLS[768] + PatchMean[768] = 1536  |  CLS only = 768
  biomedclip     BiomedCLIP ViT-B/16 (Microsoft, PubMed-pretrained)
                   CLS[768] + PatchMean[768] = 1536  |  CLS only = 768
  dinov2_vitb14  DINOv2 ViT-B/14 (Meta, self-supervised)
                   CLS[768] + PatchMean[768] = 1536  |  CLS only = 768

Features are always saved as [N, 1536] (concat) regardless of feature_mode —
feature_mode slicing happens at load time in data/dataset.py.

Saved files in data/features/{backbone}/:
  {dataset}_{split}_feat.npy   float32 [N, backbone_dim*2]
  {dataset}_{split}_label.npy  int64   [N]  local class id (0-based per dataset)

Usage:
  python extract_features.py              # uses backbone from config.py
  python extract_features.py --backbone biomedclip
  python extract_features.py --backbone dinov2_vitb14

Estimated time: ~5 min GPU, ~20 min CPU per backbone
"""
import os, argparse, zipfile, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

from config import CFG, BACKBONE_REGISTRY

# ── Image normalisation stats per backbone ────────────────────────────────────
NORM_STATS = {
    "clip_vitb32":    ((0.48145466, 0.4578275,  0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
    "biomedclip":     ((0.48145466, 0.4578275,  0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
    "dinov2_vitb14":  ((0.485, 0.456, 0.406),
                       (0.229, 0.224, 0.225)),
}


# ── Backbone loaders ──────────────────────────────────────────────────────────

def load_backbone(backbone: str, device):
    """Load model. Returns (model, loader_type)."""
    info = BACKBONE_REGISTRY[backbone]

    if info["loader"] == "open_clip":
        import open_clip
        print(f"  Loading {info['model_name']} (pretrained={info['pretrained']}) via open_clip...")
        model, _, _ = open_clip.create_model_and_transforms(
            info["model_name"], pretrained=info["pretrained"]
        )
        return model.to(device).eval(), "open_clip"

    elif info["loader"] == "open_clip_hub":
        import open_clip
        print(f"  Loading {info['model_name']} via open_clip hub...")
        model, _, _ = open_clip.create_model_and_transforms(info["model_name"])
        return model.to(device).eval(), "open_clip"

    elif info["loader"] == "hf_dinov2":
        from transformers import AutoModel
        print(f"  Loading {info['model_name']} via HuggingFace transformers...")
        model = AutoModel.from_pretrained(info["model_name"])
        return model.to(device).eval(), "hf_dinov2"

    else:
        raise ValueError(f"Unknown loader type: {info['loader']}")


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_batch(model, imgs: torch.Tensor, loader_type: str) -> torch.Tensor:
    """
    Extract [B, dim*2] features (CLS + PatchMean) for a batch of images.
    Always returns full concat features; slicing to CLS-only happens in dataset.py.
    """
    if loader_type == "open_clip":
        vis = model.visual
        # Patch embedding
        x = vis.conv1(imgs)                                          # [B, D, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # [B, L, D]
        # Prepend CLS token
        cls_tok = vis.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tok, x], dim=1)                          # [B, L+1, D]
        x = x + vis.positional_embedding
        x = vis.ln_pre(x)
        x = vis.transformer(x)
        x = vis.ln_post(x)
        cls        = x[:, 0, :]                                      # [B, 768]
        patch_mean = x[:, 1:, :].mean(dim=1)                        # [B, 768]

    elif loader_type == "hf_dinov2":
        outputs    = model(pixel_values=imgs)
        hidden     = outputs.last_hidden_state                       # [B, L+1, 768]
        cls        = hidden[:, 0, :]                                 # [B, 768]
        patch_mean = hidden[:, 1:, :].mean(dim=1)                   # [B, 768]

    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")

    return torch.cat([cls, patch_mean], dim=-1)                      # [B, 1536]


# ── Per-dataset extraction ────────────────────────────────────────────────────

def get_transform(n_channels: int, backbone: str):
    mean, std = NORM_STATS[backbone]
    base = [
        transforms.Resize((CFG.image_size, CFG.image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if n_channels == 1:
        base.append(transforms.Grayscale(num_output_channels=3))
    base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(base)


@torch.no_grad()
def extract_and_save(model, loader_type: str, dataset_name: str,
                     split: str, backbone: str, device):
    info       = INFO[dataset_name]
    n_channels = info["n_channels"]
    transform  = get_transform(n_channels, backbone)

    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, transform=transform, download=True, size=28)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_feats, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"  {dataset_name}/{split}", leave=False):
        feat = extract_batch(model, imgs.to(device), loader_type)
        all_feats.append(feat.cpu().numpy())
        all_labels.append(labels.squeeze(-1).numpy())

    feats  = np.concatenate(all_feats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)

    out_dir = CFG.feature_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{dataset_name}_{split}_feat.npy"),  feats)
    np.save(os.path.join(out_dir, f"{dataset_name}_{split}_label.npy"), labels)
    print(f"    {dataset_name}/{split}: {feats.shape}  labels {labels.shape}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None,
                        choices=list(BACKBONE_REGISTRY.keys()),
                        help="Override backbone (default: from config.py)")
    args = parser.parse_args()

    backbone = args.backbone or CFG.backbone
    info     = BACKBONE_REGISTRY[backbone]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Backbone  : {backbone}")
    print(f"Note      : {info['note']}")
    print(f"Feature dir: data/features/{backbone}/")
    print(f"Device    : {device}")
    print()

    model, loader_type = load_backbone(backbone, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded. ({n_params:.0f}M params)\n")

    for ds_name in CFG.datasets:
        print(f"[{ds_name}]")
        for split in ["train", "val", "test"]:
            extract_and_save(model, loader_type, ds_name, split, backbone, device)
        print()

    print(f"Done. Features saved to: data/features/{backbone}/")

    # Auto-zip for easy upload to Kaggle / Drive
    zip_path = f"data/features_{backbone}.zip"
    feat_dir = f"data/features/{backbone}"
    print(f"\nZipping features → {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(feat_dir)):
            if fname.endswith(".npy"):
                zf.write(os.path.join(feat_dir, fname),
                         arcname=os.path.join(backbone, fname))
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"Zip saved: {zip_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
