"""
End-to-end fine-tuning: backbone + MoE head trained on raw images.

2-Stage strategy
  Stage 1 (--frozen_epochs): backbone fully frozen, only MoE head warms up.
  Stage 2 (remaining):       last --finetune_blocks blocks unfrozen (ViT) or
                             full model (CNN), with backbone_lr << head_lr.

CNN (convnext_base) — full model unfrozen in Stage 2.
ViT (clip, biomedclip, dino) — last 2 blocks + final norm unfrozen.

Usage:
  python train_finetune.py
  python train_finetune.py --epochs 50 --frozen_epochs 5 --finetune_blocks 2
  python train_finetune.py --backbone_lr 1e-5 --head_lr 1e-4

Outputs:
  results/checkpoints/best_finetune_{backbone}_{name}.pt  (backbone + head state)
  results/history_finetune_{backbone}_{name}.csv
  results/config_finetune_{backbone}_{name}.json
"""
import os, argparse, json, dataclasses
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pytorch_metric_learning import losses as pml_losses

from config import CFG
from utils import set_seed
from data.image_dataset import get_image_loaders
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline
from models.backbone_wrapper import BackboneWrapper
from losses.load_balance import load_balance_loss
from losses.orthogonality import expert_orthogonality_loss
from losses.modality_affinity import modality_affinity_loss
from eval.metrics import evaluate_all
from analysis.routing_analysis import run_routing_analysis

os.makedirs(CFG.checkpoint_dir, exist_ok=True)
os.makedirs(CFG.results_dir,    exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",           default="moe", choices=["moe", "linear", "mlp"])
parser.add_argument("--epochs",          type=int,   default=CFG.epochs)
parser.add_argument("--frozen_epochs",   type=int,   default=5,
                    help="Stage 1: epochs with backbone fully frozen (warm-up)")
parser.add_argument("--finetune_blocks", type=int,   default=2,
                    help="Last N transformer blocks to unfreeze in Stage 2 (ViT only)")
parser.add_argument("--backbone_lr",     type=float, default=1e-5,
                    help="LR for unfrozen backbone params in Stage 2")
parser.add_argument("--head_lr",         type=float, default=CFG.lr,
                    help="LR for MoE head (both stages)")
parser.add_argument("--name",            default=None)
args = parser.parse_args()

run_name = args.name or f"finetune_{CFG.backbone}_{args.model}"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(CFG.seed)
print(f"Run             : {run_name}")
print(f"Device          : {device}  |  Seed: {CFG.seed}")
print(f"Backbone        : {CFG.backbone}  |  feature_mode={CFG.feature_mode}  |  feature_dim={CFG.feature_dim}")
print(f"Stage 1 frozen  : {args.frozen_epochs} epochs  →  head_lr={args.head_lr}")
print(f"Stage 2 unfreeze: last {args.finetune_blocks} blocks  →  backbone_lr={args.backbone_lr}")

# ── Backbone (frozen at start) ────────────────────────────────────────────
backbone = BackboneWrapper(CFG.backbone, device)
backbone.freeze_all()

# ── Head ──────────────────────────────────────────────────────────────────
MODEL_MAP = {"moe": MoEMedIR, "linear": LinearBaseline, "mlp": MLPBaseline}
head = MODEL_MAP[args.model]().to(device)

n_head     = sum(p.numel() for p in head.parameters() if p.requires_grad)
n_backbone = sum(p.numel() for p in backbone.model.parameters())
print(f"Head params     : {n_head:,} trainable")
print(f"Backbone params : {n_backbone:,} (frozen until Stage 2)")

# ── Loss ──────────────────────────────────────────────────────────────────
supcon = pml_losses.SupConLoss(temperature=CFG.temperature)

# ── Optimiser factory ─────────────────────────────────────────────────────
def make_optimizer(stage: int):
    if stage == 1:
        return torch.optim.AdamW(head.parameters(),
                                 lr=args.head_lr, weight_decay=CFG.weight_decay)
    return torch.optim.AdamW([
        {"params": head.parameters(),             "lr": args.head_lr},
        {"params": backbone.backbone_parameters(), "lr": args.backbone_lr},
    ], weight_decay=CFG.weight_decay)

optim = make_optimizer(1)

# ── Data ──────────────────────────────────────────────────────────────────
train_loader = get_image_loaders("train")
val_loaders  = get_image_loaders("val")
print(f"Train batches/epoch: {len(train_loader)}")

is_moe         = args.model == "moe"
use_lb         = is_moe and CFG.routing_mode == "token_choice"
use_expert_aux = is_moe


# ── Wrapper model for evaluate_all ────────────────────────────────────────
class _EndToEndModel(nn.Module):
    """Wraps backbone + head so evaluate_all can call model(imgs)."""
    def __init__(self, bb, h):
        super().__init__()
        self.bb = bb
        self.h  = h

    def forward(self, imgs):
        return self.h(self.bb(imgs))


# ── Training loop ─────────────────────────────────────────────────────────
best_map, history = 0.0, []
stage = 1

for epoch in range(1, args.epochs + 1):

    # ── Stage 1 → Stage 2 transition ─────────────────────────────────────
    if stage == 1 and epoch > args.frozen_epochs:
        stage = 2
        backbone.unfreeze_partial(n_blocks=args.finetune_blocks)
        optim = make_optimizer(2)
        n_unfrozen = sum(p.numel() for p in backbone.backbone_parameters())
        print(f"\n--- Stage 2 start (epoch {epoch}) ---")
        print(f"    Unfrozen backbone params: {n_unfrozen:,}\n")

    if stage == 1:
        backbone.model.eval()     # keep BN/dropout in eval mode while frozen
    else:
        backbone.model.train()
    head.train()

    total_loss = total_sc = total_lb = total_orth = total_aff = 0.0

    for imgs, labels, ds_ids in tqdm(train_loader,
                                     desc=f"Ep {epoch:3d} [S{stage}]", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                 else torch.tensor(labels, dtype=torch.long, device=device)
        ds_ids = ds_ids.to(device) if isinstance(ds_ids, torch.Tensor) \
                 else torch.tensor(ds_ids, dtype=torch.long, device=device)

        # Stage 1: no grad through backbone (save memory + speed)
        if stage == 1:
            with torch.no_grad():
                feats = backbone(imgs)
        else:
            feats = backbone(imgs)

        feats = feats + CFG.feat_noise * torch.randn_like(feats)

        embs, router_logits = head(feats)
        sc_loss = supcon(embs, labels)

        lb_loss = load_balance_loss(router_logits) if use_lb \
                  else torch.tensor(0.0, device=device)

        if use_expert_aux and router_logits is not None:
            orth_loss = expert_orthogonality_loss(head.moe)
            aff_loss  = modality_affinity_loss(router_logits, ds_ids, len(CFG.datasets))
        else:
            orth_loss = torch.tensor(0.0, device=device)
            aff_loss  = torch.tensor(0.0, device=device)

        loss = (sc_loss
                + CFG.lambda_lb       * lb_loss
                + CFG.lambda_orth     * orth_loss
                + CFG.lambda_affinity * aff_loss)

        optim.zero_grad()
        loss.backward()
        all_params = list(head.parameters()) + backbone.backbone_parameters()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optim.step()

        total_loss += loss.item()
        total_sc   += sc_loss.item()
        total_lb   += lb_loss.item()
        total_orth += orth_loss.item()
        total_aff  += aff_loss.item()

    n        = len(train_loader)
    avg_loss = total_loss / n

    # ── Validation every 5 epochs ─────────────────────────────────────────
    if epoch % 5 == 0 or epoch == args.epochs:
        eval_model = _EndToEndModel(backbone, head)
        results    = evaluate_all(eval_model, val_loaders, device)
        avg_map    = results["avg_mAP@R"]

        row = {"epoch": epoch, "stage": stage, "loss": round(avg_loss, 4),
               "avg_mAP@R": avg_map,
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
        print(f"Ep {epoch:3d} [S{stage}] | loss={avg_loss:.4f} "
              f"(sc={total_sc/n:.3f} lb={total_lb/n:.3f} "
              f"orth={total_orth/n:.3f} aff={total_aff/n:.3f}) | "
              f"avg mAP@R={avg_map:.2f} | {per_ds}")

        if avg_map > best_map:
            best_map = avg_map
            ckpt = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
            torch.save({
                "backbone":  backbone.state_dict(),
                "head":      head.state_dict(),
                "epoch":     epoch,
                "avg_mAP@R": best_map,
            }, ckpt)
            print(f"         -> New best: {best_map:.2f}  saved to {ckpt}")

# ── Save history & config ──────────────────────────────────────────────────
hist_path = os.path.join(CFG.results_dir, f"history_{run_name}.csv")
pd.DataFrame(history).to_csv(hist_path, index=False)

cfg_dict = dataclasses.asdict(CFG)
cfg_dict.update({
    "run_name":        run_name,
    "frozen_epochs":   args.frozen_epochs,
    "finetune_blocks": args.finetune_blocks,
    "backbone_lr":     args.backbone_lr,
    "head_lr":         args.head_lr,
    "best_mAP@R":      best_map,
})
cfg_path = os.path.join(CFG.results_dir, f"config_{run_name}.json")
with open(cfg_path, "w") as f:
    json.dump(cfg_dict, f, indent=2)

print(f"\nDone. Best val mAP@R: {best_map:.2f}")
print(f"History : {hist_path}")
print(f"Config  : {cfg_path}")

# ── FINAL TEST: reload best ckpt, eval on TEST split, routing analysis ──────
# Lần test cuối cùng: nạp lại checkpoint tốt nhất, đánh giá trên TEST split,
# và (nếu là MoE) chạy phân tích routing -> lưu heatmap + dữ liệu + metrics.
print("\n=== Final evaluation on TEST split ===")
best_ckpt = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
if os.path.exists(best_ckpt):
    state = torch.load(best_ckpt, map_location=device)
    backbone.load_state_dict(state["backbone"])
    head.load_state_dict(state["head"])
    print(f"Loaded best ckpt (val mAP@R={state.get('avg_mAP@R', float('nan')):.2f}) "
          f"from {best_ckpt}")
else:
    print(f"[warn] best ckpt not found ({best_ckpt}); evaluating current weights.")

eval_model   = _EndToEndModel(backbone, head).to(device)
eval_model.eval()
test_loaders = get_image_loaders("test")
test_results = evaluate_all(eval_model, test_loaders, device)
print(f"TEST avg mAP@R = {test_results['avg_mAP@R']:.2f}")

# lưu test metrics ra CSV
test_rows = []
for ds_name in CFG.datasets:
    row_t = {"dataset": ds_name}
    row_t.update(test_results[ds_name])
    test_rows.append(row_t)
test_rows.append({"dataset": "average", "mAP@R": test_results["avg_mAP@R"]})
test_csv = os.path.join(CFG.results_dir, f"test_{run_name}.csv")
pd.DataFrame(test_rows).to_csv(test_csv, index=False)
print(f"Test metrics : {test_csv}")

# routing specialization (chỉ MoE) -> {run}_routing_heatmap.png/.pdf + matrix.csv/.npy + metrics.json
if args.model == "moe":
    run_routing_analysis(eval_model, test_loaders, device, run_name=run_name)
    print(f"Routing plot+data saved under: {CFG.results_dir}/")
