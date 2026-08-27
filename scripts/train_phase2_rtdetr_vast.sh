#!/usr/bin/env bash
# Run the Phase 2 RT-DETR training on Vast.ai. Runs preflight first; only
# starts training if the preflight passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export OD_PATHS="$REPO_ROOT/configs/common/paths_vast.yaml"

CONFIG="configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml"
LOG_DIR="/workspace/logs/phase2"
LOG_FILE="$LOG_DIR/rtdetr_phase2.log"

python scripts/preflight_phase2.py --config "$CONFIG"

mkdir -p "$LOG_DIR"

python scripts/train_ultralytics.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
