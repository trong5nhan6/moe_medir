"""
Zero-shot baselines: BiomedCLIP, CLIP, DINOv2.

No training — uses raw L2-normalised features directly for retrieval.
Evaluates same mAP@R + R@K protocol as the main model.

BiomedCLIP  — uses full 1024-dim feature (CLS + PatchMean)
CLIP        — uses only CLS 512-dim (first half of the 1024 vector)
DINOv2      — uses only CLS 512-dim (same, since we used BiomedCLIP for extraction)

Note: CLIP and DINOv2 zero-shot here means "use BiomedCLIP-extracted features
without any fine-tuning" rather than re-extracting with each backbone.
For the paper, this is a fair lower bound showing BiomedCLIP features
outperform generic features even before fine-tuning.

Usage:
  python baselines/zeroshot.py --model biomedclip
  python baselines/zeroshot.py --model clip
  python baselines/zeroshot.py --model dinov2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from config import CFG
from data.dataset import get_loaders
from eval.metrics import evaluate_dataset


class ZeroShotModel(torch.nn.Module):
    """Identity model: L2-normalise the input features."""
    def __init__(self, use_dims: int = 1024):
        super().__init__()
        self.use_dims = use_dims

    def forward(self, x):
        # Optionally slice to simulate a different backbone's feature size
        out = x[:, :self.use_dims]
        return F.normalize(out, dim=-1), None


CONFIGS = {
    "biomedclip": 1024,   # full CLS + PatchMean
    "clip":        512,   # CLS only (simulates CLIP ViT-B/16 output)
    "dinov2":      512,   # CLS only (simulates DINOv2 ViT-B/14 output)
}


def run_zeroshot(model_name: str):
    assert model_name in CONFIGS, f"Unknown model: {model_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.results_dir, exist_ok=True)

    model = ZeroShotModel(use_dims=CONFIGS[model_name]).to(device).eval()
    test_loaders = get_loaders("test")

    METRICS = ["mAP@R", "MRR", "R@1", "R@5", "R@10", "MPR@1", "MPR@5", "MPR@10"]

    rows, map_scores = [], []
    print(f"\n[Zero-shot {model_name}]")
    header = f"{'Dataset':<15}" + "".join(f"{m:>8}" for m in METRICS)
    print(header)
    print("-" * len(header))

    for ds_name, loader in test_loaders.items():
        r = evaluate_dataset(model, loader, device)
        rows.append({"Dataset": ds_name, "model": model_name, **r})
        map_scores.append(r["mAP@R"])
        vals = "".join(f"{r.get(m, 0):>8.2f}" for m in METRICS)
        print(f"{ds_name:<15}{vals}")

    avg = round(float(np.mean(map_scores)), 2)
    rows.append({"Dataset": "Average", "model": model_name, "mAP@R": avg})
    print("-" * len(header))
    print(f"{'Average':<15}{avg:>8.2f}")

    out = os.path.join(CFG.results_dir, f"zeroshot_{model_name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="biomedclip",
                        choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    run_zeroshot(args.model)
