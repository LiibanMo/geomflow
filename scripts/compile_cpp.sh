#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Compiling C++ examples..."
mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"
cmake -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release ..
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" train_cnf infer_cnf

echo ""
echo "Build complete. Binaries:"
echo "  train_cnf:  build/examples/cpp/train_cnf"
echo "  infer_cnf:  build/examples/cpp/infer_cnf"