#!/usr/bin/env bash
# =============================================================================
# run_table.sh — Sinh bảng thực nghiệm chính (Table 1)
#
# Usage:
#   bash run_table.sh                   # full pipeline
#   bash run_table.sh --skip-extract    # skip feature extraction
#   bash run_table.sh --fast            # 20 epochs (debug)
# =============================================================================
set -e

SKIP_EXTRACT=false
EPOCHS=50

for arg in "$@"; do
  case $arg in
    --skip-extract) SKIP_EXTRACT=true ;;
    --fast)         EPOCHS=20         ;;
  esac
done

mkdir -p results/logs

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "======================================"
log " MoE-MedIR  |  epochs=$EPOCHS"
log "======================================"

# ── 1. Feature extraction (one-time) ─────────────────────────────────────
if [ "$SKIP_EXTRACT" = false ]; then
  log "[1] Extracting BiomedCLIP features..."
  python extract_features.py 2>&1 | tee results/logs/extract.log
else
  log "[1] Skipping feature extraction"
fi

# ── 2. Zero-shot baselines (no training needed) ───────────────────────────
log "[2] Zero-shot baselines..."
python baselines/zeroshot.py --model biomedclip
python baselines/zeroshot.py --model clip
python baselines/zeroshot.py --model dinov2

# ── 3. Train HashNet-64 ───────────────────────────────────────────────────
log "[3] Training HashNet-64..."
python baselines/hashnet.py --bits 64 --epochs $EPOCHS \
  2>&1 | tee results/logs/hashnet64.log

# ── 4. Train Linear + MLP ablations ──────────────────────────────────────
log "[4] Training Linear baseline..."
python train.py --model linear --name linear --epochs $EPOCHS \
  2>&1 | tee results/logs/linear.log

log "[4] Training MLP baseline..."
python train.py --model mlp --name mlp --epochs $EPOCHS \
  2>&1 | tee results/logs/mlp.log

# ── 5. Train main MoE model ───────────────────────────────────────────────
log "[5] Training MoE-MedIR..."
python train.py --model moe --name moe --epochs $EPOCHS \
  2>&1 | tee results/logs/moe.log

# ── 6. Test-set evaluation for trained models ─────────────────────────────
log "[6] Evaluating on test set..."
python eval/evaluate.py --model linear --name linear
python eval/evaluate.py --model mlp    --name mlp
python eval/evaluate.py --model moe    --name moe

# ── 7. Compile results table ──────────────────────────────────────────────
log "[7] Compiling table..."
python baselines/compile_table.py

# ── 8. Generate figures ───────────────────────────────────────────────────
log "[8] Generating figures..."
python analysis/routing_heatmap.py
python analysis/tsne_plot.py --model moe

log "======================================"
log " DONE"
log " results/table_main.tex   <- paste into paper"
log " results/routing_heatmap.pdf"
log " results/tsne_moe.pdf"
log "======================================"
