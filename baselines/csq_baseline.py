"""
CSQ (Central Similarity Quantization) retrieval baseline — CVPR 2020.

Trains binary hash codes by pulling features toward class-specific
hash centers (pre-generated fixed {-1,+1} codes per class).

Loss:
  L_csq   = BCE(proj(x), (center[label]+1)/2)  ← pull toward class center
  L_quant = mean(1 - tanh(proj(x))^2)          ← push codes toward ±1

Retrieval: cosine similarity on tanh(proj(x)) codes (real-valued).
Hash retrieval (inference): sign(proj(x)) → Hamming distance.

Works with any feature_mode (cls or concat).

Usage:
  python baselines/csq_baseline.py
  python baselines/csq_baseline.py --n_bits 64
"""
import os, sys, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from utils import set_seed
from data.dataset import get_loaders
from eval.metrics import evaluate_all

parser = argparse.ArgumentParser()
parser.add_argument("--n_bits", type=int, default=64, help="Hash code length")
args = parser.parse_args()

N_BITS = args.n_bits

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSQHead(nn.Module):
    """
    Hash code head: feature → n_bits real-valued codes via tanh.
    Retrieval uses cosine similarity on tanh codes.
    Final binary codes = sign(proj(x)).
    """
    def __init__(self, n_bits: int = N_BITS):
        super().__init__()
        self.n_bits = n_bits
        self.proj   = nn.Linear(CFG.feature_dim, n_bits)

    def forward(self, x):
        codes = torch.tanh(self.proj(x))         # [B, n_bits] ∈ (-1, 1)
        emb   = F.normalize(codes, dim=-1)       # L2-norm for cosine retrieval
        return emb, None

    @torch.no_grad()
    def get_binary_hash(self, x):
        return self.proj(x).sign()               # {-1, +1} binary codes


def generate_hash_centers(n_classes: int, n_bits: int, seed: int = 42) -> torch.Tensor:
    """
    Generate balanced {-1,+1} hash centers for each class.
    Uses random Gaussian → sign to get approximately balanced codes.
    """
    rng     = np.random.RandomState(seed)
    centers = np.sign(rng.randn(n_classes, n_bits)).astype(np.float32)
    return torch.from_numpy(centers)


def csq_loss(logits: torch.Tensor, labels: torch.Tensor,
             hash_centers: torch.Tensor) -> torch.Tensor:
    """
    logits:       [B, n_bits]  (proj(x), before tanh)
    labels:       [B]          global class IDs 0..total_classes-1
    hash_centers: [C, n_bits]  {-1,+1} target codes per class
    """
    targets = hash_centers[labels].to(logits.device)     # [B, n_bits]
    # BCE loss: pull logits toward class hash center
    t_binary = (targets + 1) / 2                         # {-1,+1} → {0,1}
    l_bce    = F.binary_cross_entropy_with_logits(logits, t_binary)
    # Quantization regularizer: push tanh activations toward ±1
    l_quant  = (1 - torch.tanh(logits).pow(2)).mean()
    return l_bce + 0.1 * l_quant


def main():
    hash_centers = generate_hash_centers(CFG.total_classes, N_BITS)
    model  = CSQHead(N_BITS).to(device)
    optim  = torch.optim.AdamW(model.parameters(),
                               lr=CFG.lr, weight_decay=CFG.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, total_iters=CFG.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, CFG.epochs - CFG.warmup_epochs))
    sched  = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[CFG.warmup_epochs])

    train_loader = get_loaders("train")
    val_loaders  = get_loaders("val")
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.results_dir,    exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CSQ  |  backbone={CFG.backbone}  |  n_bits={N_BITS}  |  "
          f"params={n_params:,}  |  device={device}")

    best_map, history = 0.0, []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels, _ in tqdm(train_loader,
                                     desc=f"CSQ Ep {epoch:3d}", leave=False):
            feats  = feats.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels, dtype=torch.long, device=device)
            feats  = feats + CFG.feat_noise * torch.randn_like(feats)

            logits = model.proj(feats)
            loss   = csq_loss(logits, labels, hash_centers)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
        sched.step()

        if epoch % 5 == 0 or epoch == CFG.epochs:
            results = evaluate_all(model, val_loaders, device)
            avg_map = results["avg_mAP@R"]
            row     = {"epoch": epoch, "avg_mAP@R": avg_map,
                       "loss": round(total_loss / len(train_loader), 4)}
            for ds in CFG.datasets:
                row[f"{ds}.mAP@R"] = results[ds]["mAP@R"]
            history.append(row)
            per_ds = "  ".join(
                f"{ds[:4]}={results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
            print(f"CSQ Ep {epoch:3d} | loss={row['loss']:.4f} | "
                  f"avg mAP@R={avg_map:.2f} | {per_ds}")
            if avg_map > best_map:
                best_map = avg_map
                torch.save(model.state_dict(),
                           os.path.join(CFG.checkpoint_dir, "best_csq.pt"))
                print(f"         -> New best: {best_map:.2f}")

    pd.DataFrame(history).to_csv(
        os.path.join(CFG.results_dir, "history_csq.csv"), index=False)
    print(f"\nCSQ done. Best val mAP@R: {best_map:.2f}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n--- Test set evaluation (best checkpoint) ---")
    model.load_state_dict(torch.load(
        os.path.join(CFG.checkpoint_dir, "best_csq.pt"), map_location=device))
    model.eval()
    test_loaders  = get_loaders("test")
    test_results  = evaluate_all(model, test_loaders, device)
    avg_test      = test_results["avg_mAP@R"]
    per_ds        = "  ".join(
        f"{ds[:4]}={test_results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
    print(f"CSQ  TEST  avg mAP@R={avg_test:.2f}  |  {per_ds}")
    rows_test = [{"Dataset": ds, **test_results[ds]} for ds in CFG.datasets]
    rows_test.append({"Dataset": "Average", "mAP@R": avg_test})
    pd.DataFrame(rows_test).to_csv(
        os.path.join(CFG.results_dir, "test_csq.csv"), index=False)
    print(f"Saved: {CFG.results_dir}/test_csq.csv")


if __name__ == "__main__":
    main()
