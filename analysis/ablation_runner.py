"""
Runs all ablation variants and saves results to results/ablation.csv.
Usage: python analysis/ablation_runner.py
"""
import subprocess, pandas as pd, os
from config import CFG

os.makedirs(CFG.results_dir, exist_ok=True)

# Define ablation configs as CLI overrides
# Each tuple: (name, extra_args)
ABLATIONS = [
    # A: Head architecture
    ("linear",              ["--model", "linear"]),
    ("mlp",                 ["--model", "mlp"]),
    # B-E: MoE variants — handled via config patches
    # (For these we temporarily patch config and retrain)
    ("moe_k2",              ["--model", "moe", "--name", "moe_k2"]),
    ("moe_k4",              ["--model", "moe", "--name", "moe_k4"]),
    ("moe_k8",              ["--model", "moe", "--name", "moe_k8"]),   # default
    ("moe_k16",             ["--model", "moe", "--name", "moe_k16"]),
    ("moe_top1",            ["--model", "moe", "--name", "moe_top1"]),
    ("moe_top2",            ["--model", "moe", "--name", "moe_top2"]), # default
    ("moe_top4",            ["--model", "moe", "--name", "moe_top4"]),
    ("moe_no_lb",           ["--model", "moe", "--name", "moe_no_lb"]),
]

results = []
for name, args in ABLATIONS:
    print(f"\nRunning ablation: {name}")
    cmd = ["python", "train.py", "--epochs", "30"] + args
    subprocess.run(cmd, check=True)

    # Read last result from history CSV
    hist = pd.read_csv(os.path.join(CFG.results_dir, f"history_{name}.csv"))
    best = hist.loc[hist["avg_mAP@R"].idxmax()]
    results.append({"variant": name, "avg_mAP@R": best["avg_mAP@R"]})
    print(f"  {name}: mAP@R = {best['avg_mAP@R']:.2f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(CFG.results_dir, "ablation.csv"), index=False)
print("\nAblation results saved.")
print(df.to_string())
