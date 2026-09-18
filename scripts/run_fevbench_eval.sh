#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG-2 - Downstream Aligned Covariate RAG on fev-bench
# Double-Blind Compliant Modular Architecture
# Evaluates TS-RAG-2 against baselines (RAFT, k-NN Weighted, Unaligned Concat)
# across the 30 known-covariate planning tasks in fev-bench
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/fevbench"
mkdir -p "$OUT_DIR"

METHODS=("ts_rag_2" "raft" "knn_weighted" "unaligned_concat")
BACKBONES=("chronos-bolt" "chronos-2" "moirai-2.0")
SEEDS=(42 100 2021)

echo "=== Starting fev-bench Downstream Covariate RAG Sweeps ==="
echo "Python Executable: $PYTHON"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

for METHOD in "${METHODS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ">>> Running fev-bench: Method=$METHOD | Backbone=$BACKBONE | Seed=$SEED"
            $PYTHON evaluate_fev_bench.py \
                --method "$METHOD" \
                --backbone "$BACKBONE" \
                --seed "$SEED" \
                --seq_len 512 \
                --pred_len 64 \
                --top_k 10 \
                "$@" || echo "Warning: Run failed for $METHOD-$BACKBONE-$SEED"
        done
    done
done

echo "=== fev-bench Evaluations Completed Successfully ==="
