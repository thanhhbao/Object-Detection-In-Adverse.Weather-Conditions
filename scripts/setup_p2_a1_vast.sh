#!/usr/bin/env bash
set -euo pipefail

# Configurable via env or positional args
BDD_FULL_ROOT="${BDD_FULL_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_full_yolo}"
BDD_USED_ROOT="${BDD_USED_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_yolo}"
POOL_ROOT="${POOL_ROOT:-/workspace/datasets_noleak/bdd_remaining_pool}"
RETRIEVED_ROOT="${RETRIEVED_ROOT:-/workspace/datasets_noleak/bdd_active_retrieved}"
PHASE2_MERGED="${PHASE2_MERGED:-/workspace/datasets_noleak/phase2_merged_yolo}"
P2A1_MERGED="${P2A1_MERGED:-/workspace/datasets_noleak/phase2_a1_merged_yolo}"
PHASE2_CKPT="${PHASE2_CKPT:-/workspace/runs/phase2_final_rtdetr/weights/best.pt}"
XWOD_ROOT="${XWOD_ROOT:-/workspace/datasets_noleak/xwod_6cls_yolo}"
ACDC_ROOT="${ACDC_ROOT:-/workspace/datasets_noleak/acdc_6cls_yolo}"
BDD30K_ROOT="${BDD30K_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_yolo}"
TOP_K="${TOP_K:-5000}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.75}"

REPO="/workspace/Object-Detection"
PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export PYTHONPATH

echo "=== P2-A1 DATA PREPARATION ==="
echo "Step 1: Build BDD remaining pool..."
python "${REPO}/scripts/build_bdd_retrieval_pool.py" \
  --full-root "${BDD_FULL_ROOT}" \
  --used-root "${BDD_USED_ROOT}" \
  --out-root "${POOL_ROOT}" \
  --mode symlink --clean

echo "Step 2: Active retrieval (failure-driven)..."
python "${REPO}/scripts/active_retrieval.py" \
  --weights "${PHASE2_CKPT}" \
  --pool-root "${POOL_ROOT}" \
  --query-root "${XWOD_ROOT}/images/train" \
  --query-root "${ACDC_ROOT}/images/train" \
  --out-root "${RETRIEVED_ROOT}" \
  --top-k "${TOP_K}" \
  --similarity-threshold "${SIM_THRESHOLD}" \
  --target-classes 1 3 4 \
  --mode symlink \
  --used-bdd-root "${BDD_USED_ROOT}" \
  --seed 42

echo "Step 3: Build P2-A1 merged dataset..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${RETRIEVED_ROOT}" \
  --out-root "${P2A1_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink --oversample-rare --clean

echo "Step 4: Preflight P2-A1..."
python "${REPO}/scripts/preflight_p2_a1.py" \
  --config "${REPO}/configs/ultralytics/p2_a1_rtdetr_bdd_retrieval.yaml" \
  --pool-stats "${POOL_ROOT}/pool_stats.json" \
  --retrieval-stats "${RETRIEVED_ROOT}/retrieval_stats.json"

echo ""
echo "P2-A1 DATA PREPARATION: PASS"
echo ""
POOL_STATS=$(python3 -c "import json; d=json.load(open('${POOL_ROOT}/pool_stats.json')); print(f'  full_bdd_train={d[\"full_train_pairs\"]}  excluded={d[\"excluded_existing_names\"]}  pool={d[\"candidate_pool\"]}')" 2>/dev/null || echo "  (pool stats unavailable)")
RET_STATS=$(python3 -c "import json; d=json.load(open('${RETRIEVED_ROOT}/retrieval_stats.json')); print(f'  retrieved={d[\"selected_unique\"]}')" 2>/dev/null || echo "  (retrieval stats unavailable)")
echo "  Pool: ${POOL_STATS}"
echo "  Retrieval: ${RET_STATS}"
echo "  Script complete. Run setup_p2_a1_vast.sh before training."
echo "  DO NOT run training automatically — see train_p2_a1_vast.sh"
