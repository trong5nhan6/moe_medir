"""
Evaluation script for fine-tuned backbone + MoE head (train_finetune.py output).

Checkpoint format (different from train.py):
  {"backbone": state_dict, "head": state_dict, "epoch": int, "avg_mAP@R": float}

Uses raw images via data/image_dataset.py (same as training).

Usage:
  python eval/evaluate_finetune.py
  python eval/evaluate_finetune.py --name finetune_clip_vitb32_moe
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, numpy as np, pandas as pd
from config import CFG
from data.image_dataset import get_image_loaders
from models.full_model import MoEMedIR
from models.backbone_wrapper import BackboneWrapper
from eval.metrics import evaluate_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=None,
                    help="run name (default: finetune_{backbone}_moe)")
args = parser.parse_args()

run_name = args.name or f"finetune_{CFG.backbone}_moe"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────
ckpt_path = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        f"Run: python train_finetune.py")

ckpt = torch.load(ckpt_path, map_location=device)

backbone = BackboneWrapper(CFG.backbone, device)
backbone.load_state_dict(ckpt["backbone"])
backbone.eval()

head = MoEMedIR().to(device)
head.load_state_dict(ckpt["head"])
head.eval()

print(f"Loaded  : {ckpt_path}")
print(f"Backbone: {CFG.backbone}  |  epoch={ckpt.get('epoch','?')}  "
      f"|  best val mAP@R={ckpt.get('avg_mAP@R','?')}\n")


# ── Combined model for evaluate_dataset ──────────────────────────────────
class _FullModel(nn.Module):
    def __init__(self, bb, h):
        super().__init__()
        self.bb = bb
        self.h  = h

    def forward(self, imgs):
        return self.h(self.bb(imgs))


model = _FullModel(backbone, head)

# ── Evaluate on test set ──────────────────────────────────────────────────
METRICS = ["mAP@R", "MRR", "R@1", "R@5", "R@10", "MPR@1", "MPR@5", "MPR@10"]
test_loaders = get_image_loaders("test")
rows, map_scores = [], []

header = f"{'Dataset':<15}" + "".join(f"{m:>8}" for m in METRICS)
print(header)
print("-" * len(header))

for ds_name, loader in test_loaders.items():
    r = evaluate_dataset(model, loader, device)
    map_scores.append(r["mAP@R"])
    rows.append({"Dataset": ds_name, **r})
    vals = "".join(f"{r.get(m, 0):>8.2f}" for m in METRICS)
    print(f"{ds_name:<15}{vals}")

avg_map = round(float(np.mean(map_scores)), 2)
rows.append({"Dataset": "Average", "mAP@R": avg_map})
print("-" * len(header))
print(f"{'Average':<15}{'':>8}{avg_map:>8.2f}")

os.makedirs(CFG.results_dir, exist_ok=True)
out = os.path.join(CFG.results_dir, f"test_{run_name}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved: {out}")
