import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, torch, numpy as np, pandas as pd
from config import CFG
from data.dataset import get_loaders
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline
from eval.metrics import evaluate_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="moe", choices=["moe","linear","mlp"])
parser.add_argument("--name",  default=None)
args = parser.parse_args()

run_name = args.name or args.model
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAP = {"moe": MoEMedIR, "linear": LinearBaseline, "mlp": MLPBaseline}
model = MODEL_MAP[args.model]().to(device)
ckpt  = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}\nRun: python train.py --model {args.model}")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print(f"Loaded: {ckpt}\n")

METRICS = ["mAP@R", "MRR", "R@1", "R@5", "R@10", "MPR@1", "MPR@5", "MPR@10"]

test_loaders = get_loaders("test")
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
