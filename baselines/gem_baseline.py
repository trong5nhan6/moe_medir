"""
GeM (Generalized Mean Pooling) retrieval baseline — TPAMI 2019.

Adapts GeM to ViT/CNN features: instead of pooling a 2D spatial map,
learns the optimal p to fuse CLS (global) and PatchMean (local) tokens.

  p=1  → arithmetic mean of CLS+PatchMean (same as simple average)
  p→∞  → max of CLS+PatchMean element-wise
  p~3  → typical sweet-spot, learned from data

Requires feature_mode=concat so both CLS and PatchMean are available.

Usage:
  python baselines/gem_baseline.py
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd
from tqdm import tqdm
from pytorch_metric_learning import losses as pml_losses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from utils import set_seed
from data.dataset import get_loaders
from eval.metrics import evaluate_all

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeMHead(nn.Module):
    """
    GeM fusion head: learns p to pool CLS and PatchMean tokens,
    then projects to embed_dim.
    """
    def __init__(self):
        super().__init__()
        self.p    = nn.Parameter(torch.ones(1) * 3.0)   # learnable GeM power
        d         = CFG.backbone_dim
        self.proj = nn.Sequential(
            nn.Linear(d, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )

    def forward(self, x):
        d = CFG.backbone_dim
        if x.shape[-1] == d * 2:
            # concat mode: split CLS[0:d] and PatchMean[d:2d]
            cls   = x[:, :d]
            patch = x[:, d:]
            p = self.p.clamp(min=1.0, max=6.0)
            # GeM across 2 "tokens": ((cls^p + patch^p) / 2)^(1/p)
            gem = ((cls.clamp(min=1e-6).pow(p) +
                    patch.clamp(min=1e-6).pow(p)) / 2).pow(1.0 / p)
        else:
            gem = x   # cls-only mode — no GeM fusion, fallback
        return F.normalize(self.proj(gem), dim=-1), None


def main():
    if CFG.feature_mode != "concat":
        print("[WARN] GeM works best with feature_mode=concat (both CLS+PatchMean). "
              f"Current: {CFG.feature_mode}")

    model  = GeMHead().to(device)
    supcon = pml_losses.SupConLoss(temperature=CFG.temperature)
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
    print(f"GeM  |  backbone={CFG.backbone}  |  params={n_params:,}  |  device={device}")

    best_map, history = 0.0, []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels, _ in tqdm(train_loader,
                                     desc=f"GeM Ep {epoch:3d}", leave=False):
            feats  = feats.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels, dtype=torch.long, device=device)
            feats  = feats + CFG.feat_noise * torch.randn_like(feats)
            embs, _ = model(feats)
            loss    = supcon(embs, labels)
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
                       "loss": round(total_loss / len(train_loader), 4),
                       "gem_p": round(model.p.item(), 3)}
            for ds in CFG.datasets:
                row[f"{ds}.mAP@R"] = results[ds]["mAP@R"]
            history.append(row)
            per_ds = "  ".join(
                f"{ds[:4]}={results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
            print(f"GeM Ep {epoch:3d} | loss={row['loss']:.4f} | "
                  f"avg mAP@R={avg_map:.2f} | {per_ds} | p={model.p.item():.2f}")
            if avg_map > best_map:
                best_map = avg_map
                torch.save(model.state_dict(),
                           os.path.join(CFG.checkpoint_dir, "best_gem.pt"))
                print(f"         -> New best: {best_map:.2f}")

    pd.DataFrame(history).to_csv(
        os.path.join(CFG.results_dir, "history_gem.csv"), index=False)
    print(f"\nGeM done. Best val mAP@R: {best_map:.2f}  |  learned p={model.p.item():.3f}")


if __name__ == "__main__":
    main()
