"""
Evaluation metrics for image retrieval.

evaluate_dataset(model, loader, device)
    -> dict with mAP@R, MRR, R@1/5/10, P@1/5/10

evaluate_all(model, loaders, device)
    -> dict with per-dataset metrics + avg_mAP@R

Metric definitions:
  mAP@R  Mean Average Precision at R
         R = number of relevant items (same class) in gallery
         AP@R = precision averaged at each correct hit in top-R
         mAP@R = mean AP@R over all queries

  MRR    Mean Reciprocal Rank
         For each query: 1 / rank_of_first_correct_result
         MRR = mean over all queries
         Range: (0, 1]. MRR=1 means every query finds correct at rank 1.

  R@K    Recall at K
         = fraction of queries where >= 1 correct appears in top-K
         Binary per query (0 or 1), then averaged.
         Question: "Does at least 1 correct result exist in top-K?"

  P@K    Precision at K
         = fraction of top-K results that are correct class
         Continuous per query [0,1], then averaged.
         Question: "How many of top-K results are actually correct?"

  R@K vs P@K example (K=5, query has 3 correct in gallery):
    Retrieved: [correct, wrong, correct, wrong, wrong]
    R@5 = 1.0  (at least 1 correct found)
    P@5 = 2/5 = 0.40  (2 out of 5 are correct)

All metrics use cosine similarity. Self-retrieval excluded.
All values reported as % (0-100) except MRR (0-100 scale too).
"""
import torch
import numpy as np
from config import CFG


def _map_at_r(is_correct: torch.Tensor) -> float:
    """
    Pure-PyTorch mAP@R (no faiss needed).

    is_correct: [N, N] bool — is_correct[i, j] = True if rank-j item
                for query i is relevant (same class), self excluded.
    Rows are already sorted by descending similarity.

    For each query:
      R   = total number of relevant items in gallery
      AP@R = (1/R) * sum_{k=1}^{R} P@k * rel@k
    mAP@R = mean over all queries
    """
    N = is_correct.shape[0]
    ap_scores = []

    # Number of relevant items per query (same class, excl. self)
    R_per_query = is_correct.sum(dim=1)  # [N]

    for i in range(N):
        R = int(R_per_query[i].item())
        if R == 0:
            continue
        top_r   = is_correct[i, :R].float()          # top-R hits
        cum_hits = top_r.cumsum(dim=0)               # cumulative correct
        ranks    = torch.arange(1, R + 1, dtype=torch.float32)
        precision_at_k = cum_hits / ranks             # P@k for k=1..R
        ap = (precision_at_k * top_r).sum() / R
        ap_scores.append(ap.item())

    return round(float(np.mean(ap_scores)) * 100, 2) if ap_scores else 0.0


@torch.no_grad()
def evaluate_dataset(model, loader, device) -> dict:
    """
    Evaluate retrieval on one dataset.

    Returns dict (all values in %, 0-100):
      mAP@R, MRR, R@1, R@5, R@10, P@1, P@5, P@10
    """
    model.eval()
    all_embs, all_labels = [], []

    for feats, labels, _ in loader:
        embs, _ = model(feats.to(device))
        all_embs.append(embs.cpu())
        all_labels.append(labels if isinstance(labels, torch.Tensor)
                          else torch.tensor(labels))

    embs   = torch.cat(all_embs,   dim=0)   # [N, D]
    labels = torch.cat(all_labels, dim=0)   # [N]
    N      = len(labels)

    # ── Cosine similarity matrix (L2-normalised embeddings) ───────────────
    sim = embs @ embs.T                     # [N, N]
    sim.fill_diagonal_(-1e9)                # exclude self-retrieval

    # Sort descending once — reuse for all K-based metrics
    sorted_idx = sim.argsort(dim=1, descending=True)   # [N, N]
    sorted_labels = labels[sorted_idx]                  # [N, N]
    is_correct = (sorted_labels == labels.unsqueeze(1)) # [N, N] bool

    # ── MRR ───────────────────────────────────────────────────────────────
    # rank of first correct result (1-indexed)
    # argmax on bool tensor returns first True position (0-indexed)
    first_correct_rank = is_correct.float().argmax(dim=1) + 1  # [N], 1-indexed
    # If no correct result exists (all False), argmax returns 0 → rank=1 (wrong)
    # Guard: set to N+1 if query has no positives in gallery
    has_positive = is_correct.any(dim=1)
    rr = torch.where(has_positive,
                     1.0 / first_correct_rank.float(),
                     torch.zeros(N))
    mrr = round(rr.mean().item() * 100, 2)

    # ── Recall@K ──────────────────────────────────────────────────────────
    recall = {}
    for k in CFG.recall_k:
        hit = is_correct[:, :k].any(dim=1).float()
        recall[f"R@{k}"] = round(hit.mean().item() * 100, 2)

    # ── MPR@K (Mean Precision at K) ───────────────────────────────────────
    # MPR@K = fraction of top-K results that are correct class, averaged over queries
    mpr = {}
    for k in CFG.recall_k:
        p_at_k = is_correct[:, :k].float().mean(dim=1)   # [N]
        mpr[f"MPR@{k}"] = round(p_at_k.mean().item() * 100, 2)

    # ── mAP@R (pure PyTorch, no faiss) ───────────────────────────────────
    map_r = _map_at_r(is_correct)

    return {
        "mAP@R":  map_r,
        "MRR":    mrr,
        **recall,   # R@1, R@5, R@10
        **mpr,      # MPR@1, MPR@5, MPR@10
    }


def evaluate_all(model, val_loaders: dict, device) -> dict:
    """
    Evaluate on all datasets.

    Returns:
      {
        "pathmnist":  {"mAP@R": .., "MRR": .., "R@1": .., "P@1": .., ...},
        "dermamnist": {...},
        ...
        "avg_mAP@R":  float
      }
    """
    results    = {}
    map_scores = []

    for ds_name, loader in val_loaders.items():
        r = evaluate_dataset(model, loader, device)
        results[ds_name] = r
        map_scores.append(r["mAP@R"])

    results["avg_mAP@R"] = round(float(np.mean(map_scores)), 2)
    return results
