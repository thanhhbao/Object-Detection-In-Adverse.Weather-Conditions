#!/usr/bin/env bash
# Build the Phase 2 merged dataset on Vast.ai from the no-leak source datasets,
# then run the preflight checks. Does NOT start training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OD_PATHS="$REPO_ROOT/configs/common/paths_vast.yaml"

python scripts/build_phase2_dataset.py \
  --xwod-root /workspace/datasets_noleak/xwod_6cls_yolo \
  --acdc-root /workspace/datasets_noleak/acdc_6cls_yolo \
  --bdd-root /workspace/datasets_noleak/bdd100k_6cls_yolo \
  --out-root /workspace/datasets_noleak/phase2_merged_yolo \
  --bdd-replay-images 2000 \
  --seed 42 \
  --mode symlink \
  --oversample-rare \
  --clean

python scripts/preflight_phase2.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml
