#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG-2 - Scaling Laws & Manifold Rank Estimation Runner
# Double-Blind Compliant Modular Architecture
# Fits non-parametric retrieval power laws across 24 density points (3 seeds x 8 sizes)
# and evaluates scaling exponents against theoretical bounds: alpha <= 2 / (2 + d_int)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/scaling_laws"
mkdir -p "$OUT_DIR"

echo "=== Starting 24-Point Memory Scaling Laws & Intrinsic Rank Sweeps ==="
echo "Python Executable: $PYTHON"
echo "Output Directory: $OUT_DIR"
echo "====================================================================="

$PYTHON scripts/fit_scaling_laws.py \
    --simulate \
    --metric CRPS \
    --out_file "$OUT_DIR/scaling_law_fits.json" \
    "$@"

echo "=== Scaling Laws Sweeps Completed Successfully ==="
