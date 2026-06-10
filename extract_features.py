"""
Step 1 — Run ONCE before training.

Loads CLIP ViT-B/32 (frozen, ~350MB), iterates over all 4 MedMNIST datasets × 3 splits,
extracts 1536-dim features and saves them as .npy files.

Feature = CLS token [768] concat mean(patch tokens) [768] = [1536]  (CLIP ViT-B/32)
  CLS captures global semantics
  PatchMean captures spatial average (richer than CLS alone)

Saved files in data/features/:
  {dataset}_{split}_feat.npy   float32 [N, 1536]
  {dataset}_{split}_label.npy  int64   [N]  local class id (0-based per dataset)

Usage: python extract_features.py
Estimated time: ~5 min GPU, ~20 min CPU
Disk usage: ~1 GB total
"""
import os, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import open_clip

from config import CFG

# CLIP ViT-B/32 standard normalisation (same as OpenAI CLIP)
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def get_transform(n_channels: int):
    """Handle grayscale (1ch) and RGB (3ch) datasets."""
    base = [
        transforms.Resize((CFG.image_size, CFG.image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if n_channels == 1:
        base.append(transforms.Grayscale(num_output_channels=3))
    base += [
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ]
    return transforms.Compose(base)


@torch.no_grad()
def extract_and_save(model, dataset_name: str, split: str, device):
    info       = INFO[dataset_name]
    n_channels = info["n_channels"]
    transform  = get_transform(n_channels)

    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, transform=transform,
                   download=True, size=28)   # download 28px, transform resizes to 224
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_feats, all_labels = [], []

    for imgs, labels in tqdm(loader,
                             desc=f"  {dataset_name}/{split}", leave=False):
        imgs = imgs.to(device)                                   # [B, 3, 224, 224]

        # CLIP ViT-B/32: visual transformer forward
        # model.visual returns final CLS embedding by default
        # To get patch tokens, use the internal transformer
        vis   = model.visual
        x     = vis.conv1(imgs)                                  # patch embed
        x     = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, L, D]
        x     = torch.cat([vis.class_embedding.unsqueeze(0).unsqueeze(0)
                            .expand(x.shape[0], -1, -1), x], dim=1)      # prepend CLS
        x     = x + vis.positional_embedding
        x     = vis.ln_pre(x)
        x     = vis.transformer(x)                               # [B, L+1, D]
        x     = vis.ln_post(x)

        cls        = x[:, 0, :]                                  # [B, 768]
        patch_mean = x[:, 1:, :].mean(dim=1)                    # [B, 768]
        feat       = torch.cat([cls, patch_mean], dim=-1)        # [B, 1536]

        all_feats.append(feat.cpu().numpy())
        all_labels.append(labels.squeeze(-1).numpy())   # medmnist returns [N,1]

    feats  = np.concatenate(all_feats,  axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    os.makedirs(CFG.feature_dir, exist_ok=True)
    np.save(os.path.join(CFG.feature_dir, f"{dataset_name}_{split}_feat.npy"),  feats)
    np.save(os.path.join(CFG.feature_dir, f"{dataset_name}_{split}_label.npy"), labels)
    print(f"    {dataset_name}/{split}: features {feats.shape}  labels {labels.shape}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLIP {CFG.backbone} (pretrained={CFG.backbone_pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(
        CFG.backbone, pretrained=CFG.backbone_pretrained
    )
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.visual.parameters()) / 1e6
    print(f"CLIP visual encoder loaded. ({n_params:.0f}M params)\n")

    for ds_name in CFG.datasets:
        print(f"[{ds_name}]")
        for split in ["train", "val", "test"]:
            extract_and_save(model, ds_name, split, device)
        print()

    print("Done. Features saved to:", CFG.feature_dir)


if __name__ == "__main__":
    main()
