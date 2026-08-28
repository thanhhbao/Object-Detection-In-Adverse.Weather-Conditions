#!/usr/bin/env bash
set -euo pipefail

# P2-A1 Training Script
# Runs preflight → P2-A0C control training → P2-A1 retrieval training
# DO NOT run without first running setup_p2_a1_vast.sh

REPO="${REPO:-/workspace/Object-Detection}"
PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export PYTHONPATH

A1_CONFIG="${REPO}/configs/ultralytics/p2_a1_rtdetr_bdd_retrieval.yaml"
A0C_CONFIG="${REPO}/configs/ultralytics/p2_a0c_rtdetr_continue_control.yaml"
POOL_STATS="${POOL_STATS:-/workspace/datasets_noleak/bdd_remaining_pool/pool_stats.json}"
RETRIEVAL_STATS="${RETRIEVAL_STATS:-/workspace/datasets_noleak/bdd_active_retrieved/retrieval_stats.json}"

echo "=== P2-A1 TRAINING ==="
echo "Step 0: Preflight check..."
python "${REPO}/scripts/preflight_p2_a1.py" \
  --config "${A1_CONFIG}" \
  --pool-stats "${POOL_STATS}" \
  --retrieval-stats "${RETRIEVAL_STATS}"

echo ""
echo "Step 1: Train P2-A0C (control — same epochs, no retrieved data)..."
python "${REPO}/scripts/train_ultralytics.py" --config "${A0C_CONFIG}"

echo ""
echo "Step 2: Train P2-A1 (retrieval — BDD active retrieved added)..."
python "${REPO}/scripts/train_ultralytics.py" --config "${A1_CONFIG}"

echo ""
echo "=== P2-A1 TRAINING COMPLETE ==="
echo "Compare P2-A0C vs P2-A1 mAP50:95 to measure retrieval benefit."
echo "Held-out test sets remain frozen — do not retrain after evaluating."
