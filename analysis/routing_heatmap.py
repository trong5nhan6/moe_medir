import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from config import CFG
from data.dataset import get_loaders
from models.full_model import MoEMedIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MoEMedIR().to(device)
ckpt   = os.path.join(CFG.checkpoint_dir, "best_moe.pt")
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}\nRun: python train.py")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

test_loaders   = get_loaders("test")
routing_matrix = np.zeros((len(CFG.datasets), CFG.num_experts))

with torch.no_grad():
    for i, (ds_name, loader) in enumerate(test_loaders.items()):
        all_probs = []
        for feats, _, _ in loader:
            _, router_logits = model.moe(feats.to(device))
            probs = F.softmax(router_logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
        routing_matrix[i] = np.concatenate(all_probs).mean(axis=0)

MODALITY_LABELS = {
    "pathmnist":  "PathMNIST\n(Histology)",
    "dermamnist": "DermaMNIST\n(Skin lesion)",
    "octmnist":   "OCTMNIST\n(Retinal OCT)",
    "bloodmnist": "BloodMNIST\n(Blood cell)",
}
y_labels = [MODALITY_LABELS[ds] for ds in CFG.datasets]
x_labels = [f"E{i+1}" for i in range(CFG.num_experts)]

fig, ax = plt.subplots(figsize=(10, 3.5))
sns.heatmap(routing_matrix, annot=True, fmt=".2f",
            xticklabels=x_labels, yticklabels=y_labels,
            cmap="YlGn", ax=ax, linewidths=0.4, vmin=0.0, vmax=0.5,
            cbar_kws={"label": "Mean routing probability"})
ax.set_title("Expert routing probability per medical imaging modality", fontsize=12, pad=10)
ax.set_xlabel("Expert", fontsize=10)
ax.set_ylabel("Dataset (modality)", fontsize=10)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()

os.makedirs(CFG.results_dir, exist_ok=True)
out = os.path.join(CFG.results_dir, "routing_heatmap.pdf")
plt.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved: {out}")
plt.show()
