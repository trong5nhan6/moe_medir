"""
HAM10000 dataset loader for optional real-world validation.
Downloads from Kaggle (requires kaggle CLI) or accepts local path.

HAM10000: 10,015 dermatoscopy images, 7 classes
  akiec, bcc, bkl, df, mel, nv, vasc

Usage:
    from data.ham10000 import get_ham10000_loader
    loader = get_ham10000_loader(root="data/ham10000", split="test")
"""
import os, torch, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}

HAM_MEAN = [0.7629, 0.5456, 0.5703]
HAM_STD  = [0.1409, 0.1520, 0.1695]


class HAM10000Dataset(Dataset):
    """
    Expects the HAM10000 directory structure:
      root/
        HAM10000_metadata.csv
        images/
          ISIC_0024306.jpg
          ...
    """
    def __init__(self, root: str, split: str = "test",
                 transform=None, val_frac: float = 0.1):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(HAM_MEAN, HAM_STD),
        ])

        meta_path = os.path.join(root, "HAM10000_metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata not found at {meta_path}.\n"
                "Download HAM10000 from https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection"
            )

        df = pd.read_csv(meta_path)
        df["label"] = df["dx"].map(CLASS_TO_IDX)

        # Reproducible train/val/test split
        np.random.seed(42)
        idx = np.random.permutation(len(df))
        n_val  = int(len(df) * val_frac)
        n_test = int(len(df) * val_frac)

        if split == "train":
            df = df.iloc[idx[n_val + n_test:]]
        elif split == "val":
            df = df.iloc[idx[:n_val]]
        elif split == "test":
            df = df.iloc[idx[n_val:n_val + n_test]]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = df[["image_id", "label"]].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root, "images", f"{row['image_id']}.jpg")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, int(row["label"])


def get_ham10000_loader(root: str, split: str = "test",
                        batch_size: int = 64, num_workers: int = 4):
    ds = HAM10000Dataset(root, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=True)


# ── Feature extraction helper (uses BiomedCLIP, same as main pipeline) ────
def extract_ham10000_features(root: str, out_dir: str = "data/features"):
    """Extract and save HAM10000 features using BiomedCLIP."""
    import open_clip, torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.visual.to(device).eval()

    for split in ["train", "val", "test"]:
        loader = get_ham10000_loader(root, split, batch_size=128)
        all_feats, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                out  = model.forward_features(imgs)     # [B, 197, 768] ViT-B
                cls  = out[:, 0, :]                     # [B, 768]
                patch_mean = out[:, 1:, :].mean(dim=1)  # [B, 768]
                feat = torch.cat([cls, patch_mean], dim=-1)  # [B, 1536] — note: larger than MedMNIST
                all_feats.append(feat.cpu().numpy())
                all_labels.append(labels.numpy())

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"ham10000_{split}_feat.npy"),
                np.concatenate(all_feats))
        np.save(os.path.join(out_dir, f"ham10000_{split}_label.npy"),
                np.concatenate(all_labels))
        print(f"Saved ham10000 {split} features.")
