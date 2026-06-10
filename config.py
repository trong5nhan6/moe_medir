from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # ── Datasets ──────────────────────────────────────────────────────────
    datasets: List[str] = field(default_factory=lambda: [
        "pathmnist", "dermamnist", "octmnist", "bloodmnist"
    ])
    dataset_classes: Dict[str, int] = field(default_factory=lambda: {
        "pathmnist":  9,   # colon histology (H&E)
        "dermamnist": 7,   # skin lesion (dermatoscopy)
        "octmnist":   4,   # retinal OCT
        "bloodmnist": 8,   # blood cell microscopy
    })
    # Global class ID offsets (prevents label collision across datasets)
    # pathmnist: 0-8 | dermamnist: 9-15 | octmnist: 16-19 | bloodmnist: 20-27
    dataset_offsets: Dict[str, int] = field(default_factory=lambda: {
        "pathmnist":  0,
        "dermamnist": 9,
        "octmnist":   16,
        "bloodmnist": 20,
    })
    total_classes: int = 28        # 9 + 7 + 4 + 8
    image_size:    int = 224

    # Balanced sampling: same #samples per dataset per epoch
    samples_per_dataset: int = 1000

    # ── Feature extraction ────────────────────────────────────────────────
    feature_dir:  str = "data/features"
    feature_dim:  int = 1024        # CLS[512] + PatchMean[512]
    backbone_dim: int = 512         # CLIP ViT-B/32 hidden dim

    # ── Backbone ──────────────────────────────────────────────────────────
    backbone:            str = "ViT-B-32"   # open_clip model name (~350MB)
    backbone_pretrained: str = "openai"     # pretrained weights

    # ── MoE head ──────────────────────────────────────────────────────────
    num_experts:   int   = 8
    top_k:         int   = 2
    expert_hidden: int   = 512
    embed_dim:     int   = 128      # final L2-normalised retrieval embedding

    # ── Training ──────────────────────────────────────────────────────────
    batch_size:    int   = 256
    epochs:        int   = 50
    lr:            float = 1e-4
    weight_decay:  float = 1e-4
    temperature:   float = 0.07     # SupCon temperature τ
    lambda_lb:     float = 0.01     # load-balance loss weight

    # ── Evaluation ────────────────────────────────────────────────────────
    recall_k: List[int] = field(default_factory=lambda: [1, 5, 10])

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "results/checkpoints"
    results_dir:    str = "results"


CFG = Config()
