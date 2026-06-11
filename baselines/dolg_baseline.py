"""
DOLG (Dynamic Orthogonal Local-Global) retrieval baseline — ICCV 2021.

Faithful to original paper: uses ConvNeXt-Base CNN backbone.
DOLG is applied on the spatial feature map [B, 1024, H, W]:

  Global branch:  GeM pooling on full feature map         → g [B, 1024]
  Local branch:   1×1 conv attention → weighted sum       → l [B, 1024]
  Orthogonal fusion:
    l_orth = l - (l·ĝ)ĝ      (remove component of l parallel to g)
    fused  = g + l_orth       (global + non-redundant local)

Key insight: l_orth only adds info that global does NOT already have.

Pipeline: raw images → ConvNeXt features(imgs) → DOLG → proj → SupCon

Backbone: convnext_base (hardcoded — CNN as in original DOLG paper)
Images: loaded via data/image_dataset.py (no pre-extracted features needed)

Usage:
  python baselines/dolg_baseline.py
  python baselines/dolg_baseline.py --epochs 50 --frozen_epochs 5
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
parser.add_argument("--frozen_epochs", type=int,   default=5)
parser.add_argument("--backbone_lr",   type=float, default=1e-5)
parser.add_argument("--head_lr",       type=float, default=CFG.lr)
args, _ = parser.parse_known_args()

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_DIM = 1024   # ConvNeXt-Base output channels


class DOLGModel(nn.Module):
    """
    ConvNeXt-Base + DOLG spatial pooling.
    Applies GeM (global) + attention-weighted sum (local) + orthogonal fusion.
    Faithful to ICCV 2021.
    """
    def __init__(self):
        super().__init__()
        backbone      = tvm.convnext_base(weights="DEFAULT")
        self.features = backbone.features           # [B, 1024, H, W]
        d             = CNN_DIM

        # Global branch: learnable GeM power
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        # Local branch: 1×1 conv transform + spatial attention
        self.local_conv = nn.Conv2d(d, d, kernel_size=1, bias=False)
        self.local_attn = nn.Sequential(
            nn.Conv2d(d, d // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, 1, kernel_size=1),
            nn.Softplus(),                           # non-negative attention scores
        )

        self.proj = nn.Sequential(
            nn.Linear(d, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )

    def forward(self, imgs):
        feat_map = self.features(imgs)              # [B, 1024, H, W]
        B, C, H, W = feat_map.shape

        # ── Global branch: GeM ────────────────────────────────────────
        p = self.gem_p.clamp(min=1.0, max=6.0)
        g = feat_map.clamp(min=1e-6).pow(p) \
                    .mean(dim=[-2, -1]).pow(1.0 / p)    # [B, 1024]

        # ── Local branch: attention-weighted spatial sum ───────────────
        l_feat = self.local_conv(feat_map)              # [B, 1024, H, W]
        attn   = self.local_attn(feat_map)              # [B, 1, H, W]
        # Normalize attention across spatial positions
        attn   = attn / (attn.view(B, 1, -1).sum(dim=-1, keepdim=True)
                               .view(B, 1, 1, 1) + 1e-9)
        l      = (l_feat * attn).sum(dim=[-2, -1])      # [B, 1024]

        # ── Orthogonal fusion ─────────────────────────────────────────
        g_norm = F.normalize(g, dim=-1)
        # Remove from l the component that is parallel to g
        l_orth = l - (l * g_norm).sum(dim=-1, keepdim=True) * g_norm
        fused  = g + l_orth                             # [B, 1024]

        return F.normalize(self.proj(fused), dim=-1), None

    def backbone_parameters(self):
        return list(self.features.parameters())

    def head_parameters(self):
        return (list(self.local_conv.parameters()) +
                list(self.local_attn.parameters()) +
                list(self.proj.parameters()) +
                list(self.gem_p.unsqueeze(0)))


def make_optimizer(model, stage: int):
    if stage == 1:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad and
             not any(p is bp for bp in model.backbone_parameters())],
            lr=args.head_lr, weight_decay=CFG.weight_decay)
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters()
                    if p.requires_grad and
                    not any(p is bp for bp in model.backbone_parameters())],
         "lr": args.head_lr},
        {"params": model.backbone_parameters(), "lr": args.backbone_lr},
    ], weight_decay=CFG.weight_decay)


def main():
    model  = DOLGModel().to(device)
    supcon = pml_losses.SupConLoss(temperature=CFG.temperature)

    # Stage 1: freeze backbone
    for p in model.features.parameters():
        p.requires_grad_(False)
    optim = make_optimizer(model, 1)

    train_loader = get_image_loaders("train")
    val_loaders  = get_image_loaders("val")
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.results_dir,    exist_ok=True)

    n_head = sum(p.numel() for p in model.parameters()
                 if p.requires_grad)
    n_bb   = sum(p.numel() for p in model.backbone_parameters())
    print(f"DOLG (ConvNeXt-Base)  |  head≈{n_head:,}  backbone={n_bb:,}  |  device={device}")
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
        model.local_conv.train()
        model.local_attn.train()
        model.proj.train()

        total_loss = 0.0
        for imgs, labels, _ in tqdm(train_loader,
                                    desc=f"DOLG Ep {epoch:3d} [S{stage}]", leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels, dtype=torch.long, device=device)

            if stage == 1:
                with torch.no_grad():
                    feat_map = model.features(imgs)
                B, C, H, W = feat_map.shape
                p      = model.gem_p.clamp(1.0, 6.0)
                g      = feat_map.clamp(1e-6).pow(p).mean([-2,-1]).pow(1/p)
                l_feat = model.local_conv(feat_map)
                attn   = model.local_attn(feat_map)
                attn   = attn / (attn.view(B,1,-1).sum(-1,keepdim=True).view(B,1,1,1)+1e-9)
                l      = (l_feat * attn).sum([-2,-1])
                g_norm = F.normalize(g, dim=-1)
                l_orth = l - (l * g_norm).sum(-1, keepdim=True) * g_norm
                embs   = F.normalize(model.proj(g + l_orth), dim=-1)
            else:
                embs, _ = model(imgs)

            loss = supcon(embs, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
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
            print(f"DOLG Ep {epoch:3d} [S{stage}] | loss={row['loss']:.4f} | "
                  f"avg mAP@R={avg_map:.2f} | {per_ds}")
            if avg_map > best_map:
                best_map = avg_map
                torch.save(model.state_dict(),
                           os.path.join(CFG.checkpoint_dir, "best_dolg.pt"))
                print(f"         -> New best: {best_map:.2f}")

    pd.DataFrame(history).to_csv(
        os.path.join(CFG.results_dir, "history_dolg.csv"), index=False)
    print(f"\nDOLG done. Best val mAP@R: {best_map:.2f}")


if __name__ == "__main__":
    main()
