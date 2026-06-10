#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh — One-command full experiment reproduction
# Usage: bash run_experiment.sh [--skip-extract] [--skip-baselines]
# =============================================================================
set -e

SKIP_EXTRACT=false
SKIP_BASELINES=false

for arg in "$@"; do
  case $arg in
    --skip-extract)   SKIP_EXTRACT=true  ;;
    --skip-baselines) SKIP_BASELINES=true ;;
  esac
done

echo "============================================================"
echo " MoE-MedIR: Full Experiment Pipeline"
echo "============================================================"

# ── Step 1: Environment check ─────────────────────────────────────────────
echo ""
echo "[1/6] Checking environment..."
python -c "import torch, open_clip, medmnist, pytorch_metric_learning; print('  All packages OK')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ── Step 2: Feature extraction ────────────────────────────────────────────
if [ "$SKIP_EXTRACT" = false ]; then
  echo ""
  echo "[2/6] Extracting BiomedCLIP features (one-time, ~10 min)..."
  python extract_features.py
else
  echo ""
  echo "[2/6] Skipping feature extraction (--skip-extract)"
fi

# ── Step 3: Train main model ──────────────────────────────────────────────
echo ""
echo "[3/6] Training MoE-MedIR (main model)..."
python train.py --model moe --name moe --epochs 50

# ── Step 4: Ablation baselines ────────────────────────────────────────────
echo ""
echo "[4/6] Training ablation baselines (linear, mlp)..."
python train.py --model linear --name linear --epochs 50
python train.py --model mlp   --name mlp    --epochs 50

# ── Step 5: Zero-shot baselines ───────────────────────────────────────────
if [ "$SKIP_BASELINES" = false ]; then
  echo ""
  echo "[5/6] Running zero-shot baselines..."
  python baselines/zeroshot.py --model biomedclip
  python baselines/zeroshot.py --model clip
  python baselines/zeroshot.py --model dinov2
else
  echo ""
  echo "[5/6] Skipping baselines (--skip-baselines)"
fi

# ── Step 6: Evaluate + Figures ───────────────────────────────────────────
echo ""
echo "[6/6] Final evaluation and figure generation..."
python eval/evaluate.py --model moe --name moe
python analysis/routing_heatmap.py
python analysis/tsne_plot.py --model moe

echo ""
echo "============================================================"
echo " Done! Results in: results/"
echo "  - test_moe.csv          (main results)"
echo "  - routing_heatmap.pdf   (Figure 3 for paper)"
echo "  - tsne_moe.pdf          (Figure 4 for paper)"
echo "============================================================"
