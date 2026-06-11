"""
SuperGlobal re-ranking via alpha-QE (alpha Query Expansion) — ICCV 2023.

Applies post-hoc on embeddings from any trained model. No retraining required.

Core idea (simplified α-QE variant of SuperGlobal):
  For each query embedding q:
    1. Find top-K most similar gallery items {g_1,...,g_K}
    2. Weighted aggregate: q' = normalize(q + Σ sim(q,g_i)^α * g_i)
  Re-rank with new query q' → typically +1~3% mAP@R

The intuition: if top-K neighbors are all correct, aggregating them
sharpens the query representation and improves boundary cases.

Works on top of any model (MoE, GeM, DOLG, CSQ, Linear, MLP).

Usage:
  python baselines/superglobal.py                    # re-rank MoE (default)
  python baselines/superglobal.py --model gem
  python baselines/superglobal.py --model dolg --alpha 3.0 --top_k 10
  python baselines/superglobal.py --model csq  --alpha 2.0 --top_k 5
"""
import os, sys, argparse
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from utils import set_seed
from data.dataset import get_loaders
from eval.metrics import _map_at_r

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="moe",
                    choices=["moe", "gem", "dolg", "csq", "linear", "mlp"])
parser.add_argument("--alpha",  type=float, default=3.0,
                    help="QE weight exponent: higher → more selective top-K")
parser.add_argument("--top_k",  type=int,   default=10,
                    help="Number of neighbors to aggregate")
parser.add_argument("--split",  default="val", choices=["val", "test"])
args = parser.parse_args()

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model loader ──────────────────────────────────────────────────────────

def load_model(name: str):
    ckpt = os.path.join(CFG.checkpoint_dir, f"best_{name}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nRun the corresponding training script first.")

    if name == "moe":
        from models.full_model import MoEMedIR
        m = MoEMedIR()
    elif name == "linear":
        from models.full_model import LinearBaseline
        m = LinearBaseline()
    elif name == "mlp":
        from models.full_model import MLPBaseline
        m = MLPBaseline()
    elif name == "gem":
        from baselines.gem_baseline import GeMHead
        m = GeMHead()
    elif name == "dolg":
        from baselines.dolg_baseline import DOLGHead
        m = DOLGHead()
    elif name == "csq":
        from baselines.csq_baseline import CSQHead, N_BITS
        m = CSQHead(N_BITS)
    else:
        raise ValueError(f"Unknown model: {name}")

    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


# ── Re-ranking functions ──────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader):
    all_embs, all_labels = [], []
    for feats, labels, _ in loader:
        embs, _ = model(feats.to(device))
        all_embs.append(embs.cpu())
        all_labels.append(labels if isinstance(labels, torch.Tensor)
                          else torch.tensor(labels))
    return torch.cat(all_embs), torch.cat(all_labels)


@torch.no_grad()
def alpha_qe(embs: torch.Tensor, alpha: float, top_k: int) -> torch.Tensor:
    """
    Alpha Query Expansion:
      q' = normalize(q + Σ_i sim(q, g_i)^alpha * g_i)  for i in top-K

    alpha controls selectivity:
      alpha=1  → equal weight to all neighbors
      alpha=3  → strong bias toward highest-similarity neighbors (default)
    """
    sim = embs @ embs.T                                      # [N, N]
    sim.fill_diagonal_(-1e9)
    topk_vals, topk_idx = sim.topk(top_k, dim=1)            # [N, K]
    weights = topk_vals.clamp(min=0).pow(alpha)              # [N, K]
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
    topk_feats  = embs[topk_idx]                             # [N, K, D]
    aggregated  = embs + (weights.unsqueeze(-1) * topk_feats).sum(dim=1)
    return F.normalize(aggregated, dim=-1)


def compute_map_at_r(embs: torch.Tensor, labels: torch.Tensor) -> float:
    sim = embs @ embs.T
    sim.fill_diagonal_(-1e9)
    sorted_idx    = sim.argsort(dim=1, descending=True)
    sorted_labels = labels[sorted_idx]
    is_correct    = (sorted_labels == labels.unsqueeze(1))
    return _map_at_r(is_correct)


def compute_recall(embs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    sim = embs @ embs.T
    sim.fill_diagonal_(-1e9)
    sorted_idx    = sim.argsort(dim=1, descending=True)
    sorted_labels = labels[sorted_idx]
    is_correct    = (sorted_labels == labels.unsqueeze(1))
    hit = is_correct[:, :k].any(dim=1).float()
    return round(hit.mean().item() * 100, 2)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    model   = load_model(args.model)
    loaders = get_loaders(args.split)

    print(f"\nSuperGlobal (α-QE) re-ranking")
    print(f"  model={args.model}  alpha={args.alpha}  top_k={args.top_k}  split={args.split}")
    print("=" * 70)

    all_before, all_after = [], []

    for ds_name, loader in loaders.items():
        embs, labels = extract_embeddings(model, loader)

        map_before = compute_map_at_r(embs, labels)
        r1_before  = compute_recall(embs, labels, k=1)

        embs_rr   = alpha_qe(embs, alpha=args.alpha, top_k=args.top_k)
        map_after = compute_map_at_r(embs_rr, labels)
        r1_after  = compute_recall(embs_rr, labels, k=1)

        diff = map_after - map_before
        sign = "+" if diff >= 0 else ""
        print(f"  {ds_name:<12}  "
              f"mAP@R: {map_before:.2f} → {map_after:.2f} ({sign}{diff:.2f})  |  "
              f"R@1: {r1_before:.1f} → {r1_after:.1f}")
        all_before.append(map_before)
        all_after.append(map_after)

    avg_before = round(float(np.mean(all_before)), 2)
    avg_after  = round(float(np.mean(all_after)),  2)
    diff_avg   = avg_after - avg_before
    print("─" * 70)
    print(f"  {'avg':<12}  "
          f"mAP@R: {avg_before:.2f} → {avg_after:.2f} "
          f"({'+'if diff_avg>=0 else ''}{diff_avg:.2f})")
    print("=" * 70)
    print(f"\nSuperGlobal re-ranking improved avg mAP@R by {diff_avg:+.2f}")


if __name__ == "__main__":
    main()
