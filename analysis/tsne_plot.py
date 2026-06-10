"""
Figure: t-SNE of learned embeddings.

Each point = one test image. Color = dataset (modality).
Good retrieval model: tight intra-class clusters, clear inter-class separation.

Usage:
  python analysis/tsne_plot.py               # default: MoE model
  python analysis/tsne_plot.py --model mlp   # compare with MLP baseline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import CFG
from data.dataset import get_loaders
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline

parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="moe", choices=["moe","linear","mlp"])
parser.add_argument("--name",        default=None)
parser.add_argument("--max_samples", default=400, type=int,
                    help="Max samples per dataset (keeps plot readable)")
args = parser.parse_args()

run_name = args.name or args.model
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────
MODEL_MAP = {"moe": MoEMedIR, "linear": LinearBaseline, "mlp": MLPBaseline}
model = MODEL_MAP[args.model]().to(device)
ckpt  = os.path.join(CFG.checkpoint_dir, f"best_{run_name}.pt")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# ── Visual style per dataset ──────────────────────────────────────────────
PALETTE = {
    "pathmnist":  "#e41a1c",
    "dermamnist": "#377eb8",
    "octmnist":   "#4daf4a",
    "bloodmnist": "#ff7f00",
}
MARKERS = {
    "pathmnist":  "o",
    "dermamnist": "s",
    "octmnist":   "^",
    "bloodmnist": "P",
}
DISPLAY = {
    "pathmnist":  "PathMNIST (Histology)",
    "dermamnist": "DermaMNIST (Skin)",
    "octmnist":   "OCTMNIST (Retina)",
    "bloodmnist": "BloodMNIST (Blood)",
}

# ── Collect embeddings ────────────────────────────────────────────────────
test_loaders = get_loaders("test")
all_embs, all_ds_ids = [], []

with torch.no_grad():
    for ds_name, loader in test_loaders.items():
        ds_embs = []
        for feats, _, _ in loader:
            embs, _ = model(feats.to(device))
            ds_embs.append(embs.cpu().numpy())
        embs_np = np.concatenate(ds_embs)

        # Subsample for clean visualisation
        n   = min(args.max_samples, len(embs_np))
        idx = np.random.RandomState(42).choice(len(embs_np), n, replace=False)
        all_embs.append(embs_np[idx])
        all_ds_ids.extend([ds_name] * n)

all_embs   = np.concatenate(all_embs)
all_ds_ids = np.array(all_ds_ids)

print(f"Running t-SNE on {len(all_embs)} embeddings...")
tsne   = TSNE(n_components=2, perplexity=40, n_iter=1000,
              random_state=42, n_jobs=-1)
emb_2d = tsne.fit_transform(all_embs)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

for ds_name in CFG.datasets:
    mask = all_ds_ids == ds_name
    ax.scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        c      = PALETTE[ds_name],
        marker = MARKERS[ds_name],
        s      = 15,
        alpha  = 0.7,
        linewidths = 0,
        label  = DISPLAY[ds_name],
    )

ax.legend(fontsize=9, loc="best", framealpha=0.9)
ax.set_title(f"t-SNE of {run_name.upper()} embeddings (test set)", fontsize=12)
ax.set_xlabel("t-SNE dim 1", fontsize=10)
ax.set_ylabel("t-SNE dim 2", fontsize=10)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()

os.makedirs(CFG.results_dir, exist_ok=True)
out = os.path.join(CFG.results_dir, f"tsne_{run_name}.pdf")
plt.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved: {out}")
plt.show()
