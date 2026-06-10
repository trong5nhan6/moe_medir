from dataclasses import dataclass, field
from typing import List, Dict

# ── Backbone registry ─────────────────────────────────────────────────────────
# Switch backbone by changing Config.backbone — everything else auto-updates.
# After switching, re-run: python extract_features.py
#
# loader types:
#   "open_clip"      open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
#   "open_clip_hub"  open_clip.create_model_and_transforms(model_name)  (no pretrained arg)
#   "hf_dinov2"      transformers.AutoModel.from_pretrained(model_name)
#
BACKBONE_REGISTRY: Dict[str, dict] = {
    "clip_vitb32": {
        "dim":        768,
        "loader":     "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "note":       "CLIP ViT-B/32 — general vision-language (OpenAI, ~350 MB)",
    },
    "biomedclip": {
        "dim":        768,
        "loader":     "open_clip_hub",
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": None,
        "note":       "BiomedCLIP ViT-B/16 — 15 M PubMed biomedical images (Microsoft)",
    },
    "dinov2_vitb14": {
        "dim":        768,
        "loader":     "hf_dinov2",
        "model_name": "facebook/dinov2-base",
        "pretrained": None,
        "note":       "DINOv2 ViT-B/14 — self-supervised, strong visual features (Meta)",
    },
}


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

    # ── Backbone ──────────────────────────────────────────────────────────
    # Change ONLY this field to switch backbone (see BACKBONE_REGISTRY above).
    # backbone_dim, feature_dim, feature_dir are auto-set in __post_init__.
    backbone: str = "clip_vitb32"

    # ── Feature extraction ────────────────────────────────────────────────
    # "cls"    → CLS token only            → feature_dim = backbone_dim
    # "concat" → CLS + PatchMean concat    → feature_dim = backbone_dim * 2
    feature_mode: str = "concat"

    # Auto-set by __post_init__ — do NOT edit manually:
    backbone_dim: int = 768
    feature_dim:  int = 1536
    feature_dir:  str = "data/features/clip_vitb32"

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
    lambda_lb:      float = 0.01   # load-balance loss weight (token_choice only)
    lambda_orth:    float = 0.01   # expert weight orthogonality loss weight
    lambda_affinity: float = 0.1  # modality routing diversity loss weight
    warmup_epochs: int   = 5       # linear LR warmup before cosine decay
    feat_noise:    float = 0.01    # Gaussian noise std on input features

    # ── MoE Routing mode ──────────────────────────────────────────────────
    # "token_choice"  : each token selects top-k experts (Switch Transformer)
    # "expert_choice" : each expert selects top-c tokens (NeurIPS 2022)
    #                   → perfect load balance by construction, no lb_loss needed
    routing_mode:    str   = "token_choice"
    capacity_factor: float = 2.0   # expert_choice: slots/expert = capacity_factor*B/K

    # ── Evaluation ────────────────────────────────────────────────────────
    recall_k: List[int] = field(default_factory=lambda: [1, 5, 10])

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "results/checkpoints"
    results_dir:    str = "results"

    def __post_init__(self):
        info = BACKBONE_REGISTRY.get(self.backbone)
        if info is None:
            raise ValueError(
                f"Unknown backbone '{self.backbone}'. "
                f"Valid: {list(BACKBONE_REGISTRY.keys())}"
            )
        self.backbone_dim = info["dim"]
        self.feature_dir  = f"data/features/{self.backbone}"
        self.feature_dim  = (
            self.backbone_dim if self.feature_mode == "cls"
            else self.backbone_dim * 2
        )


CFG = Config()
