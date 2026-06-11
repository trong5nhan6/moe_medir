"""
GeM (Generalized Mean Pooling) retrieval baseline — TPAMI 2019.

Faithful to original paper: uses ConvNeXt-Base CNN backbone with
GeM pooling applied on the spatial feature map [B, 1024, H, W].

  f_gem = (mean over H×W of x^p)^(1/p)   with learnable p

  p=1  → standard average pooling (GAP)
  p→∞  → max pooling
  p~3  → sweet-spot, typically best for retrieval (learned from data)

Pipeline: raw images → ConvNeXt features(imgs) → GeM → proj → SupCon

Backbone: convnext_base (hardcoded — CNN as in original GeM paper)
Images: loaded via data/image_dataset.py (no pre-extracted features needed)

Usage:
  python baselines/gem_baseline.py
  python baselines/gem_baseline.py --epochs 50 --frozen_epochs 5
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
import pandas as pd
from tqdm import tqdm
from pytorch_metric_learning import losses as pml_losses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from utils import set_seed
from data.image_dataset import get_image_loaders
from eval.metrics import evaluate_all

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",        type=int,   default=CFG.epochs)
parser.add_argument("--frozen_epochs", type=int,   default=5,
                    help="Epochs with backbone frozen (warm-up)")
parser.add_argument("--backbone_lr",   type=float, default=1e-5)
parser.add_argument("--head_lr",       type=float, default=CFG.lr)
args = parser.parse_args()

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_DIM = 1024   # ConvNeXt-Base output channels


class GeMModel(nn.Module):
    """
    ConvNeXt-Base + GeM spatial pooling.
    Applies GeM on the H×W spatial feature map — faithful to TPAMI 2019.
    """
    def __init__(self):
        super().__init__()
        backbone      = tvm.convnext_base(weights="DEFAULT")
        self.features = backbone.features      # [B, 1024, H, W]
        self.gem_p    = nn.Parameter(torch.ones(1) * 3.0)  # learnable p
        self.proj     = nn.Sequential(
            nn.Linear(CNN_DIM, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )

    def forward(self, imgs):
        feat_map = self.features(imgs)                          # [B, 1024, H, W]
        p        = self.gem_p.clamp(min=1.0, max=6.0)
        # GeM: pool spatially — (mean of x^p)^(1/p)
        gem      = feat_map.clamp(min=1e-6).pow(p) \
                          .mean(dim=[-2, -1]).pow(1.0 / p)     # [B, 1024]
        return F.normalize(self.proj(gem), dim=-1), None

    def backbone_parameters(self):
        return list(self.features.parameters())

    def head_parameters(self):
        return list(self.gem_p.parameters()) + list(self.proj.parameters())


def make_optimizer(model, stage: int):
    if stage == 1:
        return torch.optim.AdamW(model.head_parameters(),
                                 lr=args.head_lr, weight_decay=CFG.weight_decay)
    return torch.optim.AdamW([
        {"params": model.head_parameters(),      "lr": args.head_lr},
        {"params": model.backbone_parameters(),  "lr": args.backbone_lr},
    ], weight_decay=CFG.weight_decay)


def main():
    model  = GeMModel().to(device)
    supcon = pml_losses.SupConLoss(temperature=CFG.temperature)

    # Stage 1: freeze backbone
    for p in model.features.parameters():
        p.requires_grad_(False)
    optim = make_optimizer(model, 1)

    train_loader = get_image_loaders("train")
    val_loaders  = get_image_loaders("val")
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.results_dir,    exist_ok=True)

    n_head = sum(p.numel() for p in model.head_parameters())
    n_bb   = sum(p.numel() for p in model.backbone_parameters())
    print(f"GeM (ConvNeXt-Base)  |  head={n_head:,}  backbone={n_bb:,}  |  device={device}")
    print(f"Stage 1 ({args.frozen_epochs} epochs frozen)  →  Stage 2 (backbone_lr={args.backbone_lr})")

    best_map, history, stage = 0.0, [], 1

    for epoch in range(1, args.epochs + 1):

        # Stage transition
        if stage == 1 and epoch > args.frozen_epochs:
            stage = 2
            for p in model.features.parameters():
                p.requires_grad_(True)
            optim = make_optimizer(model, 2)
            n_unf = sum(p.numel() for p in model.features.parameters())
            print(f"\n--- Stage 2 (epoch {epoch}): unfroze {n_unf:,} backbone params ---\n")

        model.features.eval() if stage == 1 else model.features.train()
        model.proj.train()

        total_loss = 0.0
        for imgs, labels, _ in tqdm(train_loader,
                                    desc=f"GeM Ep {epoch:3d} [S{stage}]", leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels, dtype=torch.long, device=device)

            if stage == 1:
                with torch.no_grad():
                    feat_map = model.features(imgs)
                p   = model.gem_p.clamp(min=1.0, max=6.0)
                gem = feat_map.clamp(min=1e-6).pow(p).mean(dim=[-2,-1]).pow(1.0/p)
                embs, _ = F.normalize(model.proj(gem), dim=-1), None
            else:
                embs, _ = model(imgs)

            loss = supcon(embs, labels)
            optim.zero_grad()
            loss.backward()
            all_params = list(model.proj.parameters()) + list(model.gem_p.unsqueeze(0))
            if stage == 2:
                all_params += list(model.features.parameters())
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
            # eval wrapper
            class _EvalWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, imgs): return self.m(imgs)
            results = evaluate_all(_EvalWrapper(model), val_loaders, device)
            avg_map = results["avg_mAP@R"]
            row     = {"epoch": epoch, "stage": stage, "avg_mAP@R": avg_map,
                       "loss": round(total_loss / len(train_loader), 4),
                       "gem_p": round(model.gem_p.item(), 3)}
            for ds in CFG.datasets:
                row[f"{ds}.mAP@R"] = results[ds]["mAP@R"]
            history.append(row)
            per_ds = "  ".join(
                f"{ds[:4]}={results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
            print(f"GeM Ep {epoch:3d} [S{stage}] | loss={row['loss']:.4f} | "
                  f"avg mAP@R={avg_map:.2f} | {per_ds} | p={model.gem_p.item():.2f}")
            if avg_map > best_map:
                best_map = avg_map
                torch.save(model.state_dict(),
                           os.path.join(CFG.checkpoint_dir, "best_gem.pt"))
                print(f"         -> New best: {best_map:.2f}")

    pd.DataFrame(history).to_csv(
        os.path.join(CFG.results_dir, "history_gem.csv"), index=False)
    print(f"\nGeM done. Best val mAP@R: {best_map:.2f}  |  learned p={model.gem_p.item():.3f}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n--- Test set evaluation (best checkpoint) ---")
    model.load_state_dict(torch.load(
        os.path.join(CFG.checkpoint_dir, "best_gem.pt"), map_location=device))
    model.eval()
    test_loaders  = get_image_loaders("test")
    test_results  = evaluate_all(model, test_loaders, device)
    avg_test      = test_results["avg_mAP@R"]
    per_ds        = "  ".join(
        f"{ds[:4]}={test_results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
    print(f"GeM  TEST  avg mAP@R={avg_test:.2f}  |  {per_ds}")
    rows_test = [{"Dataset": ds, **test_results[ds]} for ds in CFG.datasets]
    rows_test.append({"Dataset": "Average", "mAP@R": avg_test})
    pd.DataFrame(rows_test).to_csv(
        os.path.join(CFG.results_dir, "test_gem.csv"), index=False)
    print(f"Saved: {CFG.results_dir}/test_gem.csv")


if __name__ == "__main__":
    main()
