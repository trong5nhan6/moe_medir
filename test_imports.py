"""
Quick sanity check — run this BEFORE extract_features.py.
Tests all imports, config, model creation, and a dummy forward pass.
No data download required.

Usage: python test_imports.py
Expected output: all PASS, no errors.
"""
import sys, os
print("Python:", sys.version)
print("="*50)

def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")
        print(f"         {type(e).__name__}: {e}")

# ── 1. Core packages ──────────────────────────────────────────────────────
print("\n[1] Core packages")
check("torch",                    lambda: __import__("torch"))
check("numpy",                    lambda: __import__("numpy"))
check("open_clip",                lambda: __import__("open_clip"))
check("medmnist",                 lambda: __import__("medmnist"))
check("pytorch_metric_learning",  lambda: __import__("pytorch_metric_learning"))
check("sklearn",                  lambda: __import__("sklearn"))
check("pandas",                   lambda: __import__("pandas"))
check("tqdm",                     lambda: __import__("tqdm"))

# ── 2. Project imports ────────────────────────────────────────────────────
print("\n[2] Project modules")
check("config",           lambda: __import__("config"))
check("utils",            lambda: __import__("utils"))
check("data.dataset",     lambda: __import__("data.dataset"))
check("models.moe_module",lambda: __import__("models.moe_module"))
check("models.full_model",lambda: __import__("models.full_model"))
check("losses.load_balance", lambda: __import__("losses.load_balance"))
check("eval.metrics",     lambda: __import__("eval.metrics"))

# ── 3. Config values ──────────────────────────────────────────────────────
print("\n[3] Config values")
def assert_eq(a, b):
    assert a == b, f"Expected {b}, got {a}"

from config import CFG
for name, got, expected in [
    ("backbone",     CFG.backbone,          "ViT-B-32"),
    ("feature_dim",  CFG.feature_dim,       1024),
    ("num_experts",  CFG.num_experts,       8),
    ("seed",         CFG.seed,              42),
    ("datasets len", len(CFG.datasets),     4),
    ("total_classes",CFG.total_classes,     28),
]:
    if got == expected:
        print(f"  [PASS] {name} = {got}")
    else:
        print(f"  [FAIL] {name}: expected {expected}, got {got}")

# ── 4. Model creation + dummy forward pass ────────────────────────────────
print("\n[4] Model forward pass (dummy data, no GPU needed)")
import torch
from models.full_model import MoEMedIR, LinearBaseline, MLPBaseline
from utils import set_seed
set_seed(42)

dummy = torch.randn(8, CFG.feature_dim)  # batch of 8 fake features

for ModelClass in [MoEMedIR, LinearBaseline, MLPBaseline]:
    def _fwd(M=ModelClass):
        m = M()
        emb, router = m(dummy)
        assert emb.shape == (8, CFG.embed_dim), f"Bad emb shape: {emb.shape}"
        # Check L2-normalised
        norms = emb.norm(dim=-1)
        assert (norms - 1.0).abs().max() < 1e-5, "Embeddings not L2-normalised"
    check(f"{ModelClass.__name__} forward", _fwd)

# ── 5. Loss functions ─────────────────────────────────────────────────────
print("\n[5] Loss functions")
from losses.load_balance import load_balance_loss
from models.full_model import MoEMedIR

def _loss():
    m = MoEMedIR()
    _, router_logits = m(dummy)
    loss = load_balance_loss(router_logits)
    assert loss.item() >= 0, "Negative load balance loss"

check("load_balance_loss", _loss)

# ── 6. CLIP backbone loadable ─────────────────────────────────────────────
print("\n[6] CLIP ViT-B/32 (checks model exists, no full download)")
import open_clip
def _clip():
    models = open_clip.list_pretrained()
    matches = [(m, p) for m, p in models if m == "ViT-B-32" and p == "openai"]
    assert len(matches) > 0, "ViT-B-32/openai not found in open_clip registry"
check("ViT-B-32 in open_clip registry", _clip)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Done. Fix any [FAIL] above before running extract_features.py.")
