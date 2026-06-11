"""
DOLG (Dynamic Orthogonal Local-Global) retrieval baseline — ICCV 2021.

Adapts DOLG to ViT/CNN features:
  global g = CLS token  (captures holistic image semantics)
  local  l = PatchMean  (captures spatial/local details)

Orthogonal fusion:
  l_attn = attention_MLP(l)       ← weight local features by relevance
  l_orth = l_attn - (l_attn·ĝ)ĝ  ← remove component parallel to global
  fused  = g + l_orth              ← add non-redundant local info to global

Key insight: l_orth is orthogonal to g, so it only adds information that
the global descriptor does NOT already capture.

Requires feature_mode=concat.

Usage:
  python baselines/dolg_baseline.py
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


class DOLGHead(nn.Module):
    """
    DOLG head: attention-weighted local + orthogonal fusion with global.
    """
    def __init__(self):
        super().__init__()
        d = CFG.backbone_dim
        # Attention MLP for local features
        self.local_attn = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, d),
            nn.Sigmoid(),          # per-dimension attention weights ∈ (0,1)
        )
        self.proj = nn.Sequential(
            nn.Linear(d, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )

    def forward(self, x):
        d = CFG.backbone_dim
        if x.shape[-1] == d * 2:
            g = x[:, :d]           # CLS — global descriptor  [B, d]
            l = x[:, d:]           # PatchMean — local desc.  [B, d]

            # Attention-weighted local
            attn   = self.local_attn(l)   # [B, d], element-wise weights
            l_attn = l * attn             # [B, d]

            # Orthogonal projection: remove component of l parallel to g
            g_norm = F.normalize(g, dim=-1)
            proj_l_on_g = (l_attn * g_norm).sum(dim=-1, keepdim=True) * g_norm
            l_orth = l_attn - proj_l_on_g  # [B, d] — orthogonal to g

            fused = g + l_orth             # DOLG fusion [B, d]
        else:
            fused = x   # cls-only fallback

        return F.normalize(self.proj(fused), dim=-1), None


def main():
    if CFG.feature_mode != "concat":
        print("[WARN] DOLG works best with feature_mode=concat (both CLS+PatchMean). "
              f"Current: {CFG.feature_mode}")

    model  = DOLGHead().to(device)
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
    print(f"DOLG  |  backbone={CFG.backbone}  |  params={n_params:,}  |  device={device}")

    best_map, history = 0.0, []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, labels, _ in tqdm(train_loader,
                                     desc=f"DOLG Ep {epoch:3d}", leave=False):
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
                       "loss": round(total_loss / len(train_loader), 4)}
            for ds in CFG.datasets:
                row[f"{ds}.mAP@R"] = results[ds]["mAP@R"]
            history.append(row)
            per_ds = "  ".join(
                f"{ds[:4]}={results[ds]['mAP@R']:.1f}" for ds in CFG.datasets)
            print(f"DOLG Ep {epoch:3d} | loss={row['loss']:.4f} | "
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
