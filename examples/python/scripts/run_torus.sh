#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# run_torus.sh  —  train + visualise CNF on T²  (PyVista paper-grade output)
# =========================================================================
# Usage:  ./scripts/run_torus.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXAMPLE_DIR"

VENV_PYTHON="../../.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

MODEL="torus_model.pt"

echo "=== 1/2  Training CNF on torus  ==========================================="
$VENV_PYTHON train.py \
  --manifold torus \
  --epochs 200 \
  --batch-size 256 \
  --lr 0.01 \
  --dt 0.05 \
  --output "$MODEL"

echo ""
echo "=== 2/2  Rendering still PDF  ============================================"
$VENV_PYTHON infer.py \
  --model "$MODEL" \
  --manifold torus \
  --output "torus_still.pdf" \
  --still-pdf \
  --n-points 500 \
  --elev 25 \
  --azim 30 \
  --point-size 8.0 \
  --manifold-opacity 0.16 \
  --cam-distance 12.5 \
  --pdf-size 2400 \
  --pdf-dpi 300 \
  --title "CNF on T²"

echo ""
echo "Done  →  $EXAMPLE_DIR/torus_still.pdf"
