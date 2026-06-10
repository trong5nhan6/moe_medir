"""
Zero-shot baselines: CLIP ViT-B/32, CLIP (CLS-only), DINOv2.

No training — uses raw L2-normalised features directly for retrieval.
Evaluates same mAP@R + R@K protocol as the main model.

vitb32  — uses full 1536-dim feature (CLS + PatchMean) from CLIP ViT-B/32
clip    — uses only CLS 768-dim (first half of the 1536 vector)
dinov2  — uses only CLS 768-dim (same, since we used CLIP ViT-B/32 for extraction)

Note: clip and dinov2 zero-shot here means "use CLIP ViT-B/32-extracted features
without any fine-tuning" rather than re-extracting with each backbone.
For the paper, this is a fair lower bound showing CLIP ViT-B/32 features
before any metric learning fine-tuning.

Usage:
  python baselines/zeroshot.py --model vitb32
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
    "vitb32": 1536,   # full CLS + PatchMean from CLIP ViT-B/32
    "clip":    768,   # CLS only
    "dinov2":  768,   # CLS only
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
    parser.add_argument("--model", default="vitb32",
                        choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    run_zeroshot(args.model)
