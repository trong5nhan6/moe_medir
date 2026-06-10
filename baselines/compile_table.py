"""
Compile all experiment results into the main comparison table.

Reads:
  results/zeroshot_biomedclip.csv
  results/zeroshot_clip.csv
  results/zeroshot_dinov2.csv
  results/test_hashnet64.csv
  results/test_linear.csv
  results/test_mlp.csv
  results/test_moe.csv

Outputs:
  1. Console table  (mAP@R per dataset + avg)
  2. results/table_main.csv   (full 4-metric table)
  3. results/table_main.tex   (LaTeX booktabs table for paper)

Usage: python baselines/compile_table.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np
from config import CFG

RESULTS  = CFG.results_dir
DATASETS = CFG.datasets
METRICS  = ["mAP@R", "MRR", "R@1", "R@5", "R@10", "MPR@1", "MPR@5", "MPR@10"]

# Dataset short names for table columns
DS_SHORT = {
    "pathmnist":  "Path",
    "dermamnist": "Derma",
    "octmnist":   "OCT",
    "bloodmnist": "Blood",
}

# Ordered method list: (display_name, group, csv_file, model_col_filter)
METHODS = [
    # Group 1: Zero-shot
    ("BiomedCLIP (zero-shot)", "Zero-shot", "zeroshot_biomedclip.csv", "biomedclip"),
    ("CLIP (zero-shot)",       "Zero-shot", "zeroshot_clip.csv",       "clip"),
    ("DINOv2 (zero-shot)",     "Zero-shot", "zeroshot_dinov2.csv",     "dinov2"),
    # Group 2: Hash-based
    ("HashNet-64",             "Hash",      "test_hashnet64.csv",       None),
    # Group 3: Ablation baselines
    ("BiomedCLIP + Linear",   "Ablation",  "test_linear.csv",          None),
    ("BiomedCLIP + MLP",      "Ablation",  "test_mlp.csv",             None),
    # Group 4: Ours
    ("MoE-MedIR (ours)",      "Ours",      "test_moe.csv",             None),
]


def load_results(csv_file: str, model_filter=None) -> dict:
    """
    Parse a results CSV into {dataset_name: {metric: value}}.
    dataset_name is lowercased and stripped.
    """
    path = os.path.join(RESULTS, csv_file)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if model_filter and "model" in df.columns:
        df = df[df["model"] == model_filter]

    out = {}
    for _, row in df.iterrows():
        ds = str(row.get("Dataset", row.get("dataset", ""))).lower().strip()
        out[ds] = {m: float(row[m]) for m in METRICS if m in row and not pd.isna(row[m])}
    return out


def get_avg(data: dict) -> float:
    """Average mAP@R across DATASETS from a results dict."""
    scores = [data.get(ds, {}).get("mAP@R", np.nan) for ds in DATASETS]
    return round(float(np.nanmean(scores)), 2)


# ── Build table ───────────────────────────────────────────────────────────
def build_table():
    all_rows = []
    missing  = []

    for display, group, csv_file, mfilter in METHODS:
        data = load_results(csv_file, mfilter)
        if data is None:
            missing.append(csv_file)
            continue

        row = {"Method": display, "Group": group}
        for ds in DATASETS:
            ds_data = data.get(ds, {})
            for m in METRICS:
                row[f"{DS_SHORT[ds]}_{m}"] = ds_data.get(m, np.nan)

        # Average column: from "average" row in CSV, or computed
        avg_data = data.get("average", {})
        row["Avg_mAP@R"] = avg_data.get("mAP@R", get_avg(data))
        all_rows.append(row)

    if missing:
        print("\nMissing result files (skipped):")
        for f in missing:
            print(f"  {f}")

    return pd.DataFrame(all_rows)


def print_console(df: pd.DataFrame):
    """Print mAP@R summary table to console."""
    cols = [DS_SHORT[ds] for ds in DATASETS] + ["Avg"]
    header = f"{'Method':<28}" + "".join(f"{c:>8}" for c in cols)
    print("\n" + "=" * len(header))
    print("MAIN RESULTS — mAP@R (%)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    last_group = None
    for _, row in df.iterrows():
        if row["Group"] != last_group:
            if last_group is not None:
                print()
            last_group = row["Group"]

        vals = [row.get(f"{DS_SHORT[ds]}_mAP@R", np.nan) for ds in DATASETS]
        vals.append(row.get("Avg_mAP@R", np.nan))
        line = f"{row['Method']:<28}"
        line += "".join(f"{v:>8.2f}" if not np.isnan(v) else f"{'--':>8}"
                        for v in vals)
        print(line)

    print("=" * len(header))


def to_latex(df: pd.DataFrame) -> str:
    """Generate LaTeX booktabs table (mAP@R only, one column per dataset)."""
    ds_names  = [DS_SHORT[ds] for ds in DATASETS]
    col_count = 1 + len(ds_names) + 1          # Method + datasets + Avg

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval mAP@R (\%) on MedMNIST V2. "
        r"\textbf{Bold}: best per column.}",
        r"\label{tab:main}",
        r"\begin{tabular}{l" + "c" * (col_count - 1) + "}",
        r"\toprule",
        "Method & " + " & ".join(ds_names) + r" & Avg \\ \midrule",
    ]

    # Find best per column
    best = {}
    for ds in DATASETS:
        col = [row.get(f"{DS_SHORT[ds]}_mAP@R", np.nan) for _, row in df.iterrows()]
        best[ds] = np.nanmax(col)
    best_avg = df["Avg_mAP@R"].dropna().max()

    last_group = None
    for _, row in df.iterrows():
        if row["Group"] != last_group and last_group is not None:
            lines.append(r"\midrule")
        last_group = row["Group"]

        cells = [row["Method"].replace("&", r"\&")]
        for ds in DATASETS:
            v = row.get(f"{DS_SHORT[ds]}_mAP@R", np.nan)
            if np.isnan(v):
                cells.append("--")
            elif abs(v - best[ds]) < 0.01:
                cells.append(f"\\textbf{{{v:.2f}}}")
            else:
                cells.append(f"{v:.2f}")

        avg_v = row.get("Avg_mAP@R", np.nan)
        if np.isnan(avg_v):
            cells.append("--")
        elif abs(avg_v - best_avg) < 0.01:
            cells.append(f"\\textbf{{{avg_v:.2f}}}")
        else:
            cells.append(f"{avg_v:.2f}")

        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Compiling results...")
    df = build_table()

    if df.empty:
        print("\nNo results found. Run:  bash run_table.sh")
    else:
        print_console(df)

        csv_out = os.path.join(RESULTS, "table_main.csv")
        df.to_csv(csv_out, index=False)
        print(f"\nCSV: {csv_out}")

        tex_out = os.path.join(RESULTS, "table_main.tex")
        latex   = to_latex(df)
        with open(tex_out, "w") as f:
            f.write(latex)
        print(f"LaTeX: {tex_out}")
        print("\n--- LaTeX snippet ---")
        print(latex)
