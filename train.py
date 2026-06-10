"""
Main training script.

Usage:
  python train.py                          # train MoE (default)
  python train.py --model linear           # train Linear baseline
  python train.py --model mlp              # train MLP baseline
  python train.py --model moe --epochs 30  # fewer epochs (fast debug)

Outputs:
  results/checkpoints/best_{name}.pt       best checkpoint by avg val mAP@R
  results/history_{name}.csv               per-epoch train loss + val metrics

Improvements over v1:
  B1 — MultiSimilarityMiner: hard negative mining for SupConLoss
  C  — Feature noise + input dropout for robustness
  D  — Linear warmup (5 ep) + CosineAnnealingLR for training stability
  A  — Expert Specialization Loss: ép router học route đúng modality
"""
import os, argparse, torch, torch.nn.functional as F, pandas as pd, numpy as np
from tqdm import tqdm
from pytorch_metric_learning import losses as pml_losses, miners as pml_miners

from config import CFG
from utils import set_seed
from data.dataset import get_loaders
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline
from losses.load_balance import load_balance_loss
from eval.metrics import evaluate_all

os.makedirs(CFG.checkpoint_dir, exist_ok=True)
os.makedirs(CFG.results_dir,    exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="moe", choices=["moe", "linear", "mlp"])
parser.add_argument("--epochs", type=int, default=CFG.epochs)
parser.add_argument("--name",   default=None,
                    help="run name (used for checkpoint/csv filenames)")
args = parser.parse_args()

run_name = args.name or args.model
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(CFG.seed)
print(f"Model: {run_name}  |  Device: {device}  |  Seed: {CFG.seed}")

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_MAP = {"moe": MoEMedIR, "linear": LinearBaseline, "mlp": MLPBaseline}
model = MODEL_MAP[args.model]().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# ── Loss & miner (B1) ─────────────────────────────────────────────────────
supcon = pml_losses.SupConLoss(temperature=CFG.temperature)
miner  = pml_miners.MultiSimilarityMiner(epsilon=0.1)

# ── Optimiser & scheduler (D: warmup + cosine) ────────────────────────────
optim  = torch.optim.AdamW(model.parameters(),
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
warmup = torch.optim.lr_scheduler.LinearLR(
    optim, start_factor=0.1, total_iters=CFG.warmup_epochs)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=max(1, args.epochs - CFG.warmup_epochs))
sched  = torch.optim.lr_scheduler.SequentialLR(
    optim, schedulers=[warmup, cosine], milestones=[CFG.warmup_epochs])

# ── Data ──────────────────────────────────────────────────────────────────
train_loader = get_loaders("train")
val_loaders  = get_loaders("val")
print(f"Train batches/epoch: {len(train_loader)}")

is_moe = args.model == "moe"

# ── Training loop ─────────────────────────────────────────────────────────
best_map, history = 0.0, []

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, total_sc, total_lb, total_spec = 0.0, 0.0, 0.0, 0.0

    for feats, labels, ds_ids in tqdm(train_loader,
                                      desc=f"Epoch {epoch:3d}", leave=False):
        feats  = feats.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device) \
                 if not isinstance(labels, torch.Tensor) else labels.to(device)
        ds_ids = torch.tensor(ds_ids, dtype=torch.long).to(device) \
                 if not isinstance(ds_ids, torch.Tensor) else ds_ids.to(device)

        # C: Gaussian noise on features
        feats = feats + CFG.feat_noise * torch.randn_like(feats)

        embs, router_logits = model(feats)

        # B1: Hard negative mining
        hard_pairs = miner(embs, labels)
        sc_loss    = supcon(embs, labels, hard_pairs)

        lb_loss = load_balance_loss(router_logits) \
                  if router_logits is not None else torch.tensor(0.0, device=device)

        # A: Expert specialization loss (MoE only)
        if is_moe and router_logits is not None:
            spec_logits = model.spec_head(router_logits)
            spec_loss   = F.cross_entropy(spec_logits, ds_ids)
        else:
            spec_loss = torch.tensor(0.0, device=device)

        loss = sc_loss + CFG.lambda_lb * lb_loss + CFG.lambda_spec * spec_loss

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item()
        total_sc   += sc_loss.item()
        total_lb   += lb_loss.item()
        total_spec += spec_loss.item()

    sched.step()
    avg_loss = total_loss / len(train_loader)

    # ── Validation every 5 epochs ─────────────────────────────────────────
    if epoch % 5 == 0 or epoch == args.epochs:
        results  = evaluate_all(model, val_loaders, device)
        avg_map  = results["avg_mAP@R"]

        row = {"epoch": epoch, "loss": round(avg_loss, 4), "avg_mAP@R": avg_map,
               "sc_loss":   round(total_sc   / len(train_loader), 4),
               "lb_loss":   round(total_lb   / len(train_loader), 4),
               "spec_loss": round(total_spec / len(train_loader), 4)}
        for ds_name in CFG.datasets:
            for metric, val in results[ds_name].items():
                row[f"{ds_name}.{metric}"] = val
        history.append(row)

        per_ds = "  ".join(
            f"{ds[:4]}={results[ds]['mAP@R']:.1f}"
            for ds in CFG.datasets
        )
        print(f"Ep {epoch:3d} | loss={avg_loss:.4f} "
              f"(sc={total_sc/len(train_loader):.3f} "
              f"lb={total_lb/len(train_loader):.3f} "
              f"sp={total_spec/len(train_loader):.3f}) | "
              f"avg mAP@R={avg_map:.2f} | {per_ds}")

        if avg_map > best_map:
            best_map = avg_map
            ckpt = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"         -> New best: {best_map:.2f}  saved to {ckpt}")

# ── Save history ──────────────────────────────────────────────────────────
hist_path = os.path.join(CFG.results_dir, f"history_{run_name}.csv")
pd.DataFrame(history).to_csv(hist_path, index=False)
print(f"\nTraining done. Best val mAP@R: {best_map:.2f}")
print(f"History saved: {hist_path}")
