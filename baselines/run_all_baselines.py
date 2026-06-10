"""
Run ALL baseline comparisons and aggregate into one CSV table.

Baselines implemented:
  Tier 1 (Zero-shot — no training):
    - CLIP ViT-B/32 (zero-shot)
    - CLIP (zero-shot)
    - DINOv2 (zero-shot)
  Tier 2 (Fine-tuned embedding models):
    - CLIP ViT-B/32 + Linear (train.py --model linear)
    - CLIP ViT-B/32 + MLP (train.py --model mlp)
  Tier 3 (Our method):
    - CLIP ViT-B/32 + MoE (train.py --model moe)

Hash-based methods (HashNet, CSQ, GreedyHash) require separate reimplementation.
VTHSC-MIR is the closest prior work — implement in hashnet.py / csq.py separately.

Usage: python baselines/run_all_baselines.py
"""
import subprocess, os, sys, pandas as pd, numpy as np
from config import CFG

os.makedirs(CFG.results_dir, exist_ok=True)

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  {' '.join(cmd)}")
    ret = subprocess.run(cmd, check=True)
    return ret.returncode == 0


# ── Tier 1: Zero-shot ──────────────────────────────────────────────────────
for model in ["vitb32", "clip", "dinov2"]:
    run(["python", "baselines/zeroshot.py", "--model", model],
        f"Zero-shot {model}")

# ── Tier 2: Fine-tuned heads ───────────────────────────────────────────────
run(["python", "train.py", "--model", "linear", "--name", "linear"],
    "CLIP ViT-B/32 + Linear")
run(["python", "train.py", "--model", "mlp", "--name", "mlp"],
    "CLIP ViT-B/32 + MLP")

# ── Tier 3: Ours ──────────────────────────────────────────────────────────
run(["python", "train.py", "--model", "moe", "--name", "moe"],
    "CLIP ViT-B/32 + MoE (Ours)")

# ── Aggregate Results ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Aggregating results...")

dfs = []

# Zero-shot CSVs
for model in ["vitb32", "clip", "dinov2"]:
    path = os.path.join(CFG.results_dir, f"zeroshot_{model}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)

# Fine-tuned CSVs
for name in ["linear", "mlp", "moe"]:
    path = os.path.join(CFG.results_dir, f"history_{name}.csv")
    if os.path.exists(path):
        hist = pd.read_csv(path)
        best = hist.loc[hist["avg_mAP@R"].idxmax()]
        # Extract per-dataset metrics from best epoch row
        label = {"linear": "CLIP ViT-B/32 + Linear", "mlp": "CLIP ViT-B/32 + MLP",
                 "moe": "CLIP ViT-B/32 + MoE (Ours)"}[name]
        for ds in CFG.datasets:
            row = {"dataset": ds, "model": label}
            for m in ["mAP@R", "R@1", "R@5", "R@10"]:
                key = f"{ds}.{m}" if f"{ds}.{m}" in best.index else m
                row[m] = best.get(key, np.nan)
            dfs.append(pd.DataFrame([row]))
        # Avg row
        dfs.append(pd.DataFrame([{
            "dataset": "Average", "model": label,
            "mAP@R": best["avg_mAP@R"]
        }]))

if dfs:
    all_df = pd.concat(dfs, ignore_index=True)
    pivot = all_df[all_df["dataset"] == "Average"].set_index("model")["mAP@R"]
    print("\nSummary (avg mAP@R):")
    print(pivot.to_string())

    out = os.path.join(CFG.results_dir, "all_baselines.csv")
    all_df.to_csv(out, index=False)
    print(f"\nFull results saved: {out}")
else:
    print("No results found — check individual scripts for errors.")
