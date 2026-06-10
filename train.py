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

Loss components (MoE only):
  sc_loss      SupConLoss with MultiSimilarityMiner hard negative mining
  lb_loss      Load-balance loss (token_choice mode only — expert_choice
               is balanced by construction, lb_loss is skipped)
  orth_loss    Expert weight orthogonality (NeurIPS 2025)
  affinity_loss Modality routing diversity — maximises between-modality
               routing variance (MI proxy, no extra learnable params)
"""
import os, argparse, torch, torch.nn.functional as F, pandas as pd, numpy as np
from tqdm import tqdm
from pytorch_metric_learning import losses as pml_losses, miners as pml_miners

from config import CFG
from utils import set_seed
from data.dataset import get_loaders
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline
from losses.load_balance import load_balance_loss
from losses.orthogonality import expert_orthogonality_loss
from losses.modality_affinity import modality_affinity_loss
from eval.metrics import evaluate_all

os.makedirs(CFG.checkpoint_dir, exist_ok=True)
os.makedirs(CFG.results_dir,    exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="moe", choices=["moe", "linear", "mlp"])
parser.add_argument("--epochs", type=int, default=CFG.epochs)
parser.add_argument("--name",   default=None)
args = parser.parse_args()

run_name = args.name or args.model
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(CFG.seed)
print(f"Model: {run_name}  |  Device: {device}  |  Seed: {CFG.seed}")
print(f"Routing: {CFG.routing_mode}  |  capacity_factor: {CFG.capacity_factor}")

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_MAP = {"moe": MoEMedIR, "linear": LinearBaseline, "mlp": MLPBaseline}
model = MODEL_MAP[args.model]().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# ── Loss & miner ──────────────────────────────────────────────────────────
supcon = pml_losses.SupConLoss(temperature=CFG.temperature)
miner  = pml_miners.MultiSimilarityMiner(epsilon=0.1)

# ── Optimiser & scheduler (linear warmup → cosine) ────────────────────────
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

is_moe          = args.model == "moe"
use_lb          = is_moe and CFG.routing_mode == "token_choice"
use_expert_aux  = is_moe   # orth + affinity apply to both routing modes

# ── Training loop ─────────────────────────────────────────────────────────
best_map, history = 0.0, []

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = total_sc = total_lb = total_orth = total_aff = 0.0

    for feats, labels, ds_ids in tqdm(train_loader,
                                      desc=f"Epoch {epoch:3d}", leave=False):
        feats  = feats.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device) \
                 if not isinstance(labels, torch.Tensor) else labels.to(device)
        ds_ids = torch.tensor(ds_ids, dtype=torch.long).to(device) \
                 if not isinstance(ds_ids, torch.Tensor) else ds_ids.to(device)

        # Feature noise for robustness
        feats = feats + CFG.feat_noise * torch.randn_like(feats)

        embs, router_logits = model(feats)

        # Hard negative mining → SupConLoss
        hard_pairs = miner(embs, labels)
        sc_loss    = supcon(embs, labels, hard_pairs)

        # Load-balance loss (token_choice only)
        lb_loss = load_balance_loss(router_logits) if use_lb \
                  else torch.tensor(0.0, device=device)

        # Expert orthogonality + modality affinity (MoE only)
        if use_expert_aux and router_logits is not None:
            orth_loss = expert_orthogonality_loss(model.moe)
            aff_loss  = modality_affinity_loss(
                router_logits, ds_ids, len(CFG.datasets))
        else:
            orth_loss = torch.tensor(0.0, device=device)
            aff_loss  = torch.tensor(0.0, device=device)

        loss = (sc_loss
                + CFG.lambda_lb       * lb_loss
                + CFG.lambda_orth     * orth_loss
                + CFG.lambda_affinity * aff_loss)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item()
        total_sc   += sc_loss.item()
        total_lb   += lb_loss.item()
        total_orth += orth_loss.item()
        total_aff  += aff_loss.item()

    sched.step()
    n = len(train_loader)
    avg_loss = total_loss / n

    # ── Validation every 5 epochs ─────────────────────────────────────────
    if epoch % 5 == 0 or epoch == args.epochs:
        results = evaluate_all(model, val_loaders, device)
        avg_map = results["avg_mAP@R"]

        row = {"epoch": epoch, "loss": round(avg_loss, 4), "avg_mAP@R": avg_map,
               "sc_loss":   round(total_sc   / n, 4),
               "lb_loss":   round(total_lb   / n, 4),
               "orth_loss": round(total_orth / n, 4),
               "aff_loss":  round(total_aff  / n, 4)}
        for ds_name in CFG.datasets:
            for metric, val in results[ds_name].items():
                row[f"{ds_name}.{metric}"] = val
        history.append(row)

        per_ds = "  ".join(
            f"{ds[:4]}={results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
        print(f"Ep {epoch:3d} | loss={avg_loss:.4f} "
              f"(sc={total_sc/n:.3f} lb={total_lb/n:.3f} "
              f"orth={total_orth/n:.3f} aff={total_aff/n:.3f}) | "
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
