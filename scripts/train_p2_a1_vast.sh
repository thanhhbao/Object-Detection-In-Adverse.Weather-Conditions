#!/usr/bin/env bash
set -euo pipefail

# P2-A1 Training Script
# Requires explicit mode — will NOT train both by default.
#
# Usage:
#   ./scripts/train_p2_a1_vast.sh a0c          # Train control only
#   ./scripts/train_p2_a1_vast.sh a1           # Run preflight + train retrieval only
#   ./scripts/train_p2_a1_vast.sh both         # Run preflight + train both (explicit request)
#   ./scripts/train_p2_a1_vast.sh a0r          # Train A0R random-rare control (no preflight)
#   ./scripts/train_p2_a1_vast.sh a1_dino      # Train A1-DINO DINOv2 retrieval
#
# Run setup_p2_a1_vast.sh (or setup_ablation_vast.sh for A0R/A1-DINO) before this script.

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "ERROR: No mode specified." >&2
  echo "Usage: $0 {a0c|a1|both|a0r|a1_dino}" >&2
  exit 1
fi

case "$MODE" in
  a0c|a1|both|a0r|a1_dino) ;;
  *)
    echo "ERROR: Unknown mode '$MODE'. Must be one of: a0c  a1  both  a0r  a1_dino" >&2
    exit 1
    ;;
esac

REPO="${REPO:-/workspace/Object-Detection}"
PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export PYTHONPATH

A1_CONFIG="${REPO}/configs/ultralytics/p2_a1_rtdetr_bdd_retrieval.yaml"
A0C_CONFIG="${REPO}/configs/ultralytics/p2_a0c_rtdetr_continue_control.yaml"
A0R_CONFIG="${REPO}/configs/ultralytics/p2_a0r_rtdetr_random_rare.yaml"
A1_DINO_CONFIG="${REPO}/configs/ultralytics/p2_a1_dino_rtdetr_retrieval.yaml"
POOL_STATS="${POOL_STATS:-/workspace/datasets_noleak/bdd_remaining_pool/pool_stats.json}"
RETRIEVAL_STATS="${RETRIEVAL_STATS:-/workspace/datasets_noleak/bdd_active_retrieved/retrieval_stats.json}"

echo "=== P2-A1 TRAINING (mode: ${MODE}) ==="

# Preflight is required for a1 and both; optional (skipped) for a0c-only runs.
if [[ "$MODE" == "a1" || "$MODE" == "both" ]]; then
  echo "Step 0: Preflight check (required for A1)..."
  python "${REPO}/scripts/preflight_p2_a1.py" \
    --config "${A1_CONFIG}" \
    --pool-stats "${POOL_STATS}" \
    --retrieval-stats "${RETRIEVAL_STATS}"
  echo ""
fi

if [[ "$MODE" == "a0c" || "$MODE" == "both" ]]; then
  echo "--- Train P2-A0C (control — same epochs, original Phase2 data) ---"
  python "${REPO}/scripts/train_ultralytics.py" --config "${A0C_CONFIG}"
  echo ""
fi

if [[ "$MODE" == "a1" || "$MODE" == "both" ]]; then
  echo "--- Train P2-A1 (retrieval — BDD active retrieved added) ---"
  python "${REPO}/scripts/train_ultralytics.py" --config "${A1_CONFIG}"
  echo ""
fi

if [[ "$MODE" == "a0r" ]]; then
  echo "--- Train P2-A0R (random rare control — no leakage risk, no preflight needed) ---"
  python "${REPO}/scripts/train_ultralytics.py" --config "${A0R_CONFIG}"
  echo ""
fi

if [[ "$MODE" == "a1_dino" ]]; then
  echo "--- Train P2-A1-DINO (DINOv2 retrieval) ---"
  python "${REPO}/scripts/train_ultralytics.py" --config "${A1_DINO_CONFIG}"
  echo ""
fi

echo "=== P2-A1 TRAINING COMPLETE (mode: ${MODE}) ==="
echo "Compare P2-A0C vs P2-A1 vs P2-A0R vs P2-A1-DINO mAP50:95 on XWOD val."
echo "Held-out test sets remain frozen — do not retrain after evaluating."
