"""
Evaluation script for SOTA baselines (GeM, DOLG, CSQ).

Mirrors the pattern of evaluate.py / evaluate_finetune.py.
Loads best checkpoint and reports per-dataset metrics on val or test split.

Usage:
  python eval/evaluate_sota.py --model gem
  python eval/evaluate_sota.py --model dolg
  python eval/evaluate_sota.py --model csq
  python eval/evaluate_sota.py --model gem --split val
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, pandas as pd
from config import CFG
from eval.metrics import evaluate_all

IMAGE_BASED = {"gem", "dolg"}   # need raw images; csq uses pre-extracted features

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["gem", "dolg", "csq"],
                    help="Which baseline to evaluate")
parser.add_argument("--split", default="test", choices=["val", "test"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────
ckpt_path = os.path.join(CFG.checkpoint_dir, f"best_{args.model}.pt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        f"Run: python baselines/{args.model}_baseline.py")

if args.model == "gem":
    from baselines.gem_baseline import GeMModel
    model = GeMModel()
elif args.model == "dolg":
    from baselines.dolg_baseline import DOLGModel
    model = DOLGModel()
elif args.model == "csq":
    from baselines.csq_baseline import CSQHead, N_BITS
    model = CSQHead(N_BITS)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model = model.to(device).eval()
print(f"Loaded  : {ckpt_path}")

# ── Data loaders ──────────────────────────────────────────────────────────
if args.model in IMAGE_BASED:
    from data.image_dataset import get_image_loaders
    loaders = get_image_loaders(args.split)
else:
    from data.dataset import get_loaders
    loaders = get_loaders(args.split)

# ── Evaluate ──────────────────────────────────────────────────────────────
METRICS = ["mAP@R", "MRR", "R@1", "R@5", "R@10"]
results  = evaluate_all(model, loaders, device)
avg_map  = results["avg_mAP@R"]

header = f"{'Dataset':<15}" + "".join(f"{m:>8}" for m in METRICS)
print(f"\n{args.model.upper()}  [{args.split}]")
print(header)
print("-" * len(header))

rows = []
for ds in CFG.datasets:
    r    = results[ds]
    vals = "".join(f"{r.get(m, 0):>8.2f}" for m in METRICS)
    print(f"{ds:<15}{vals}")
    rows.append({"Dataset": ds, **r})

print("-" * len(header))
print(f"{'Average':<15}{'':>8}{avg_map:>8.2f}")
rows.append({"Dataset": "Average", "mAP@R": avg_map})

os.makedirs(CFG.results_dir, exist_ok=True)
out = os.path.join(CFG.results_dir, f"{args.split}_{args.model}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved: {out}")
