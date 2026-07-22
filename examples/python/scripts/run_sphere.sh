#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# run_sphere.sh  —  train + visualise CNF on S²  (PyVista paper-grade output)
# =========================================================================
# Usage:  ./scripts/run_sphere.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXAMPLE_DIR"

VENV_PYTHON="../../.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

MODEL="sphere_model.pt"

echo "=== 1/2  Training CNF on sphere  ========================================="
$VENV_PYTHON train.py \
  --manifold sphere \
  --epochs 200 \
  --batch-size 256 \
  --lr 0.01 \
  --dt 0.05 \
  --output "$MODEL"

echo ""
echo "=== 2/2  Rendering still PDF  ============================================"
$VENV_PYTHON infer.py \
  --model "$MODEL" \
  --manifold sphere \
  --output "sphere_still.pdf" \
  --still-pdf \
  --n-points 500 \
  --elev 25 \
  --azim 30 \
  --point-size 8.0 \
  --manifold-opacity 0.16 \
  --cam-distance 5.0 \
  --pdf-size 2400 \
  --pdf-dpi 300

echo ""
echo "Done  →  $EXAMPLE_DIR/sphere_still.pdf"
