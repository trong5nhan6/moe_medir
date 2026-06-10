"""
HashNet baseline on pre-extracted CLIP ViT-B/32 features.
Reference: Cao et al., "HashNet: Deep Learning to Hash by Continuation" ICCV 2017.

Adaptation:
  - Input: pre-extracted features [1024] (no image backbone needed)
  - Model: MLP → tanh(scale * x)  where scale anneals 1→20
  - Loss: weighted pairwise cross-entropy on cosine similarity
  - Eval: same cosine mAP@R protocol as MoEMedIR (fair comparison)

Usage:
    python baselines/hashnet.py --bits 64
    python baselines/hashnet.py --bits 32
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from tqdm import tqdm
from config import CFG
from data.dataset import get_loaders
from eval.metrics import evaluate_dataset

# ── Model ──────────────────────────────────────────────────────────────────
class HashNetModel(nn.Module):
    """
    MLP encoder + tanh with annealing scale.
    Outputs continuous hash codes ∈ (-1, 1)^hash_bits.
    """
    def __init__(self, input_dim: int = 1024, hash_bits: int = 64):
        super().__init__()
        self.hash_bits = hash_bits
        self.scale = 1.0        # annealed during training

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, hash_bits),
        )

    def forward(self, x):
        h = self.encoder(x)
        # Continuation: tanh with increasing scale → converges to sign()
        codes = torch.tanh(self.scale * h)
        # Return (embedding, None) to match the common eval interface
        return F.normalize(codes, dim=-1), None

    def set_scale(self, scale: float):
        self.scale = scale


# ── Loss ───────────────────────────────────────────────────────────────────
def hashnet_loss(codes: torch.Tensor, labels: torch.Tensor,
                 alpha: float = 1.0) -> torch.Tensor:
    """
    Pairwise weighted cross-entropy loss.
    s_ij = 0.5 * hash_bits * (codes_i · codes_j)  (inner product similarity)
    y_ij = 1 if same class else 0
    L = mean( log(1 + exp(s_ij)) - y_ij * s_ij ) * w_ij
    w_ij = 1 + alpha * |y_ij|  (upweight positive pairs)
    """
    # Similarity matrix in hash space (not normalised — raw inner product)
    inner = codes @ codes.T                           # [B, B]
    # Label matrix
    y = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
    # Weighted BCE
    w = 1.0 + alpha * y
    loss = w * (torch.log(1 + torch.exp(inner)) - y * inner)
    # Exclude diagonal
    mask = 1 - torch.eye(len(labels), device=labels.device)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


# ── Training ───────────────────────────────────────────────────────────────
def train_hashnet(hash_bits: int = 64, epochs: int = 50,
                  scale_start: float = 1.0, scale_end: float = 20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.results_dir, exist_ok=True)

    model = HashNetModel(input_dim=CFG.feature_dim, hash_bits=hash_bits).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    train_loader = get_loaders("train")
    val_loaders  = get_loaders("val")

    best_map, history = 0.0, []

    for epoch in range(1, epochs + 1):
        # Anneal scale: linearly from scale_start to scale_end
        scale = scale_start + (scale_end - scale_start) * (epoch / epochs)
        model.set_scale(scale)
        model.train()
        total_loss = 0.0

        for feats, labels, _ in tqdm(train_loader, desc=f"HashNet ep{epoch}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            codes, _ = model(feats)
            # Use raw (unnormalized) codes for the pairwise loss
            raw_codes = torch.tanh(scale * model.encoder(feats))
            loss = hashnet_loss(raw_codes, labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()

        sched.step()

        if epoch % 5 == 0 or epoch == epochs:
            results = {}
            map_scores = []
            for ds_name, loader in val_loaders.items():
                r = evaluate_dataset(model, loader, device)
                results[ds_name] = r
                map_scores.append(r["mAP@R"])
            avg_map = round(float(np.mean(map_scores)), 2)
            print(f"  Ep {epoch:3d} | loss={total_loss/len(train_loader):.4f} "
                  f"| scale={scale:.1f} | val mAP@R={avg_map:.2f}")
            history.append({"epoch": epoch, "avg_mAP@R": avg_map,
                            "loss": total_loss/len(train_loader)})

            if avg_map > best_map:
                best_map = avg_map
                ckpt = os.path.join(CFG.checkpoint_dir, f"best_hashnet{hash_bits}.pt")
                torch.save(model.state_dict(), ckpt)

    # Save history
    run_name = f"hashnet{hash_bits}"
    pd.DataFrame(history).to_csv(
        os.path.join(CFG.results_dir, f"history_{run_name}.csv"), index=False)
    print(f"\nHashNet-{hash_bits}: Best val mAP@R = {best_map:.2f}")
    return best_map


# ── Test evaluation ────────────────────────────────────────────────────────
def eval_hashnet(hash_bits: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HashNetModel(input_dim=CFG.feature_dim, hash_bits=hash_bits).to(device)
    ckpt  = os.path.join(CFG.checkpoint_dir, f"best_hashnet{hash_bits}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    model.set_scale(20.0)     # use final scale for test

    METRICS = ["mAP@R", "MRR", "R@1", "R@5", "R@10", "MPR@1", "MPR@5", "MPR@10"]

    test_loaders = get_loaders("test")
    rows, map_scores = [], []

    print(f"\nHashNet-{hash_bits} Test Results:")
    header = f"{'Dataset':<15}" + "".join(f"{m:>8}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for ds_name, loader in test_loaders.items():
        r = evaluate_dataset(model, loader, device)
        rows.append({"Dataset": ds_name, "model": f"HashNet-{hash_bits}", **r})
        map_scores.append(r["mAP@R"])
        vals = "".join(f"{r.get(m, 0):>8.2f}" for m in METRICS)
        print(f"{ds_name:<15}{vals}")

    avg = round(float(np.mean(map_scores)), 2)
    rows.append({"Dataset": "Average", "model": f"HashNet-{hash_bits}", "mAP@R": avg})
    print("-" * len(header))
    print(f"{'Average':<15}{avg:>8.2f}")

    df = pd.DataFrame(rows)
    out = os.path.join(CFG.results_dir, f"test_hashnet{hash_bits}.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits",   type=int,   default=64)
    parser.add_argument("--epochs", type=int,   default=50)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        train_hashnet(hash_bits=args.bits, epochs=args.epochs)
    eval_hashnet(hash_bits=args.bits)
