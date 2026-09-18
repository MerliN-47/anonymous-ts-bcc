#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG-2 - Systems Throughput, Latency & VRAM Profiling Sweep
# Double-Blind Compliant Modular Architecture
# Sweeps across:
#   Batch sizes B in {1, 8, 32, 128}
#   Context lengths L in {512, 1024, 2048}
#   Retrieval budgets k in {1, 10, 25, 50}
#   Fusion strategies: Latent (ARM), Token (In-Context), Output (Vincentization)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/systems_profiling"
mkdir -p "$OUT_DIR"

echo "=== Starting Table 1 Systems Throughput & VRAM Profiling Grid ==="
echo "Python Executable: $PYTHON"
echo "Output Directory: $OUT_DIR"
echo "================================================================="

$PYTHON benchmarks/profile_systems_throughput.py \
    --backbones chronos-bolt chronos-2 moirai-2.0 \
    --fusions latent token output \
    --batch_sizes 1 8 32 128 \
    --context_lengths 512 1024 2048 \
    --k_values 1 10 25 50 \
    --pred_len 64 \
    --warmup 1 \
    --runs 3 \
    --out_file "$OUT_DIR/systems_scaling_grid.json" \
    "$@"

echo "=== Systems Profiling Grid Completed Successfully ==="
