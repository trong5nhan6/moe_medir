"""
analysis/routing_analysis.py
----------------------------
Phân tích chuyên biệt hóa theo modality cho MoE-MedIR (dùng cho bài báo).

Khác với analysis/routing_heatmap.py (chỉ vẽ PDF cho model feature-based),
module này:
  - Hoạt động với model end-to-end (backbone + head) HOẶC head feature-based,
    miễn là model(...) trả về (embeddings, router_logits).
  - Gộp phân phối router-softmax (clean logits, KHÔNG noise ở eval) theo modality
    -> ma trận (M modalities x Nr routed experts).
  - LƯU vào results_dir:  <run>_routing_heatmap.png/.pdf  (biểu đồ)
                          <run>_routing_matrix.csv/.npy    (DỮ LIỆU để vẽ lại)
                          <run>_routing_metrics.json       (2 chỉ số định lượng)
  - Trả về dict metrics gồm:
        between_modality_variance  = (1/M) Σ_m ||p_m - p_bar||^2   (= -L_aff, Eq.15)
        mutual_information_bits     = I(modality ; top-1 expert)

Dùng cho ABLATION L_aff: chạy 2 lần với run khác nhau (vd. "moe_laff0.1", "moe_laff0"),
đặt 2 heatmap cạnh nhau + so 2 con số metrics.

API chính:
    run_routing_analysis(model, test_loaders, device, run_name, results_dir=None)
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from config import CFG

# matplotlib không cần màn hình (chạy trên server/Kaggle)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── chỉ số định lượng ──────────────────────────────────────────────────────
def between_modality_variance(R: np.ndarray) -> float:
    """R: (M, Nr) hàng đã chuẩn hóa. = (1/M) Σ_m ||p_m - p_bar||^2  (đúng -L_aff)."""
    p_bar = R.mean(axis=0, keepdims=True)
    return float(((R - p_bar) ** 2).sum(axis=1).mean())


def mutual_information_bits(joint_counts: np.ndarray) -> float:
    """I(modality ; top-1 expert) theo bit. joint_counts: (M, Nr) số đếm."""
    P = joint_counts.astype(np.float64)
    tot = P.sum()
    if tot == 0:
        return 0.0
    P /= tot
    Pm = P.sum(axis=1, keepdims=True)
    Pe = P.sum(axis=0, keepdims=True)
    denom = Pm @ Pe
    mask = (P > 0) & (denom > 0)
    return float((P[mask] * np.log2(P[mask] / denom[mask])).sum())


# ── thu phân phối routing từ model ─────────────────────────────────────────
@torch.no_grad()
def collect_routing(model, test_loaders: dict, device):
    """
    model        : callable trả (embeddings, router_logits). router_logits=None -> không phải MoE.
    test_loaders : dict {ds_name: DataLoader}, yield (imgs, labels, ds_ids).
    Trả về (R, joint_counts) hoặc (None, None) nếu head không phải MoE.
    """
    model.eval()
    modalities = list(CFG.datasets)              # giữ đúng thứ tự = modality index
    M = len(modalities)
    Nr = None
    sum_probs = None
    count = np.zeros(M, dtype=np.int64)
    joint = None

    for m_idx, ds_name in enumerate(modalities):
        loader = test_loaders[ds_name]
        for imgs, _labels, _ds_ids in loader:
            imgs = imgs.to(device)
            _, router_logits = model(imgs)
            if router_logits is None:            # Linear/MLP baseline -> bỏ qua
                return None, None
            probs = F.softmax(router_logits.float(), dim=-1).cpu().numpy()  # (B, Nr)
            if Nr is None:
                Nr = probs.shape[1]
                sum_probs = np.zeros((M, Nr), dtype=np.float64)
                joint = np.zeros((M, Nr), dtype=np.int64)
            top1 = probs.argmax(axis=1)
            sum_probs[m_idx] += probs.sum(axis=0)
            count[m_idx] += probs.shape[0]
            for e in top1:
                joint[m_idx, e] += 1

    R = sum_probs / np.clip(count, 1, None)[:, None]
    R = R / R.sum(axis=1, keepdims=True)         # chuẩn hóa hàng
    return R, joint


# ── lưu biểu đồ + dữ liệu + metrics ────────────────────────────────────────
def save_outputs(R, joint, run_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    modalities = list(CFG.datasets)
    M, Nr = R.shape
    experts = [f"E{i+1}" for i in range(Nr)]

    bmv = between_modality_variance(R)
    mi = mutual_information_bits(joint)

    label_map = {
        "pathmnist": "PathMNIST\n(Histology)",
        "dermamnist": "DermaMNIST\n(Skin lesion)",
        "octmnist": "OCTMNIST\n(Retinal OCT)",
        "bloodmnist": "BloodMNIST\n(Blood cell)",
    }
    y_labels = [label_map.get(d, d) for d in modalities]

    fig, ax = plt.subplots(figsize=(1.3 * Nr + 2, 0.95 * M + 1.6))
    im = ax.imshow(R, cmap="YlGn", aspect="auto", vmin=0.0,
                   vmax=max(0.5, float(R.max())))
    ax.set_xticks(range(Nr)); ax.set_xticklabels(experts)
    ax.set_yticks(range(M)); ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Routed expert"); ax.set_ylabel("Modality")
    ax.set_title(f"Per-modality routing distribution  ({run_name})\n"
                 f"between-modality var = {bmv:.4f}   |   "
                 f"MI(modality; top-1 expert) = {mi:.3f} bit", fontsize=10)
    for i in range(M):
        for j in range(Nr):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                    color="black" if R[i, j] < 0.5 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean routing probability")
    fig.tight_layout()
    png = os.path.join(results_dir, f"{run_name}_routing_heatmap.png")
    pdf = os.path.join(results_dir, f"{run_name}_routing_heatmap.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # DỮ LIỆU để vẽ lại biểu đồ
    np.save(os.path.join(results_dir, f"{run_name}_routing_matrix.npy"), R)
    np.savetxt(os.path.join(results_dir, f"{run_name}_routing_matrix.csv"),
               R, delimiter=",", header=",".join(experts), comments="")
    metrics = {"run": run_name,
               "between_modality_variance": bmv,
               "mutual_information_bits": mi,
               "num_modalities": int(M), "num_routed_experts": int(Nr)}
    with open(os.path.join(results_dir, f"{run_name}_routing_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[routing] heatmap -> {png}")
    print(f"[routing] data    -> {run_name}_routing_matrix.csv / .npy")
    print(f"[routing] metrics -> between-modality var={bmv:.4f} | MI={mi:.3f} bit")
    return metrics


def run_routing_analysis(model, test_loaders, device, run_name,
                         results_dir: str = None):
    """Hàm tiện dụng: thu routing -> lưu biểu đồ + dữ liệu + metrics.
    Trả metrics dict, hoặc None nếu head không phải MoE."""
    results_dir = results_dir or CFG.results_dir
    R, joint = collect_routing(model, test_loaders, device)
    if R is None:
        print("[routing] head không phải MoE (router_logits=None) -> bỏ qua.")
        return None
    return save_outputs(R, joint, run_name, results_dir)
