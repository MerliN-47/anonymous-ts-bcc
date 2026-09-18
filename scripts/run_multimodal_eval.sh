#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG-2 - Multimodal Event Grounding & Macro Shock Sweeps
# Double-Blind Compliant Modular Architecture
# Evaluates Context-is-Key macroeconomic text event stream grounding
# across nominal and counterfactual macroeconomic shock scenarios
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/multimodal"
mkdir -p "$OUT_DIR"

BACKBONES=("chronos-bolt" "chronos-2" "moirai-2.0")
ROLES=("none" "retrieval_only" "fusion_only" "dual")

echo "=== Starting Context-is-Key Multimodal Event Grounding Sweeps ==="
echo "Python Executable: $PYTHON"
echo "Output Directory: $OUT_DIR"
echo "================================================================="

for BACKBONE in "${BACKBONES[@]}"; do
    for ROLE in "${ROLES[@]}"; do
        SAVE_NAME="multimodal_${BACKBONE}_role_${ROLE}.json"
        echo ">>> Running Multimodal Sweep: Backbone=$BACKBONE | Text Role=$ROLE"

        $PYTHON zeroshot.py \
            --benchmark "context_is_key" \
            --backbone "$BACKBONE" \
            --injection_point "latent" \
            --text_encoder "bge-large-en" \
            --text_role "$ROLE" \
            --seq_len 512 \
            --pred_len 64 \
            --top_k 10 \
            --eval_metrics distributional \
            --save_file_name "$SAVE_NAME" \
            "$@" || echo "Warning: Run failed for $BACKBONE-$ROLE"
    done
done

echo "=== Multimodal Event Grounding Sweeps Completed Successfully ==="
