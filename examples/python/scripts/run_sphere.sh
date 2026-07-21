#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# run_sphere.sh  —  train + visualise CNF on S²  (PyVista paper-grade output)
# =========================================================================
# Usage:  ./scripts/run_sphere.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXAMPLE_DIR"

VENV_PYTHON="../.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

MODEL="sphere_model.pt"
ANIMATION="sphere_animation.gif"

echo "=== 1/2  Training CNF on sphere  ========================================="
$VENV_PYTHON train.py \
  --manifold sphere \
  --epochs 200 \
  --batch-size 256 \
  --lr 0.01 \
  --dt 0.05 \
  --output "$MODEL"

echo ""
echo "=== 2/2  Rendering animation (PyVista)  =================================="
$VENV_PYTHON infer.py \
  --model "$MODEL" \
  --manifold sphere \
  --output "$ANIMATION" \
  --n-points 500 \
  --n-frames 40 \
  --dt 0.05 \
  --elev 20 \
  --azim 30 \
  --rotate 90 \
  --fps 10 \
  --n-trails 10 \
  --trail-dt 0.02 \
  --trail-width 2.0 \
  --point-size 8.0 \
  --manifold-opacity 0.20 \
  --window-size 800 \
  --cam-distance 3.5

echo ""
echo "Done  →  $EXAMPLE_DIR/$ANIMATION"

