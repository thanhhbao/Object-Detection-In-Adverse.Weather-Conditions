#!/usr/bin/env bash
set -euo pipefail

BDD_RAW_ROOT="${BDD_RAW_ROOT:-/workspace/datasets_raw/bdd100k_yolo_raw}"
BDD_FULL_ROOT="${BDD_FULL_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_full_yolo}"
BDD_USED_ROOT="${BDD_USED_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_yolo}"
POOL_ROOT="${POOL_ROOT:-/workspace/datasets_noleak/bdd_remaining_pool}"
RETRIEVED_ROOT="${RETRIEVED_ROOT:-/workspace/datasets_noleak/bdd_active_retrieved}"
P2A1_MERGED="${P2A1_MERGED:-/workspace/datasets_noleak/phase2_a1_merged_yolo}"
PHASE2_CKPT="${PHASE2_CKPT:-/workspace/runs/phase2_final_rtdetr/weights/best.pt}"
XWOD_ROOT="${XWOD_ROOT:-/workspace/datasets_noleak/xwod_6cls_yolo}"
ACDC_ROOT="${ACDC_ROOT:-/workspace/datasets_noleak/acdc_6cls_yolo}"
BDD30K_ROOT="${BDD30K_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_yolo}"
TOP_K="${TOP_K:-5000}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.75}"

REPO="${REPO:-/workspace/Object-Detection}"
PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export PYTHONPATH

echo "=== P2-A1 DATA PREPARATION ==="

# ── Test gate (Vast Python environment) ──────────────────────────────────────
echo "Step -1: Verifying protocol tests pass in Vast Python environment..."
PYTEST_RESULT=0
python -m pytest -q \
  "${REPO}/tests/test_active_retrieval_protocol.py" \
  "${REPO}/tests/test_bdd_retrieval_pool.py" \
  "${REPO}/tests/test_p2_a1_builder.py" \
  "${REPO}/tests/test_preflight_p2_a1.py" \
  2>&1 | tee /tmp/p2a1_pytest.log || PYTEST_RESULT=$?

if [ "$PYTEST_RESULT" -ne 0 ]; then
  echo ""
  echo "ERROR: Protocol tests failed. Fix test failures before running setup."
  echo "Full log: /tmp/p2a1_pytest.log"
  exit 1
fi

# Guard: require torch/numpy (no critical skips due to missing deps)
if grep -q "no module named 'torch'\|no module named 'numpy'\|no module named 'ultralytics'" /tmp/p2a1_pytest.log 2>/dev/null; then
  echo ""
  echo "ERROR: Critical dependency (torch / numpy / ultralytics) is missing in this Python."
  echo "Install required packages before running setup."
  exit 1
fi

echo "  Protocol tests: PASS"

# ── Step 0: Find a real train image and run smoke check ──────────────────────
echo ""
echo "Step 0: Smoke check — verify RT-DETR model and embedding layers..."

SMOKE_IMAGE=""
for ext in jpg jpeg png bmp webp; do
  SMOKE_IMAGE=$(find "${XWOD_ROOT}/images/train" \( -type f -o -type l \) -name "*.${ext}" 2>/dev/null | head -1)
  if [ -n "$SMOKE_IMAGE" ]; then break; fi
done

if [ -z "$SMOKE_IMAGE" ]; then
  echo "ERROR: No train image found in ${XWOD_ROOT}/images/train"
  echo "  The XWOD training set must exist before running setup."
  exit 1
fi
echo "  Using real train image: ${SMOKE_IMAGE}"

python "${REPO}/scripts/smoke_p2_a1_embedding.py" \
  --weights "${PHASE2_CKPT}" \
  --embedding-layers 21 24 27 \
  --sample-image "${SMOKE_IMAGE}"

# ── Step 1: Prepare full BDD 6-class (if not already done) ──────────────────
echo ""
echo "Step 1: Prepare full BDD 6-class (if not already done)..."
if [ ! -f "${BDD_FULL_ROOT}/dataset.yaml" ]; then
  echo "  Full BDD not found at ${BDD_FULL_ROOT} — running prepare_bdd100k_yolo.py --full..."
  python "${REPO}/scripts/prepare_bdd100k_yolo.py" \
    --src "${BDD_RAW_ROOT}" \
    --dst "${BDD_FULL_ROOT}" \
    --full --clean
  echo "  Full BDD prepared."
  FULL_COUNT=$(find "${BDD_FULL_ROOT}/images/train" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')
  echo "  Full BDD train images (files + symlinks): ${FULL_COUNT}"
else
  echo "  Full BDD already exists at ${BDD_FULL_ROOT} — reusing."
  FULL_COUNT=$(find "${BDD_FULL_ROOT}/images/train" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')
  echo "  Full BDD train images (files + symlinks): ${FULL_COUNT}"
fi

# ── Step 2: Build BDD remaining pool ────────────────────────────────────────
echo ""
echo "Step 2: Build BDD remaining pool..."
python "${REPO}/scripts/build_bdd_retrieval_pool.py" \
  --full-root "${BDD_FULL_ROOT}" \
  --used-root "${BDD_USED_ROOT}" \
  --out-root "${POOL_ROOT}" \
  --mode symlink --clean

POOL_COUNT=$(find "${POOL_ROOT}/images/train" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')
echo "  Pool images (files + symlinks): ${POOL_COUNT}"

# ── Step 3: Active retrieval (failure-driven) ────────────────────────────────
echo ""
echo "Step 3: Active retrieval (failure-driven)..."
python "${REPO}/scripts/active_retrieval.py" \
  --weights "${PHASE2_CKPT}" \
  --pool-root "${POOL_ROOT}" \
  --query-root "${XWOD_ROOT}/images/train" \
  --query-root "${ACDC_ROOT}/images/train" \
  --out-root "${RETRIEVED_ROOT}" \
  --top-k "${TOP_K}" \
  --similarity-threshold "${SIM_THRESHOLD}" \
  --target-classes 1 3 4 \
  --candidate-target-classes 1 3 4 \
  --embedding-layers 21 24 27 \
  --mode symlink \
  --used-bdd-root "${BDD_USED_ROOT}" \
  --seed 42 \
  --clean

# ── Step 4: Build P2-A1 merged dataset ──────────────────────────────────────
echo ""
echo "Step 4: Build P2-A1 merged dataset..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${RETRIEVED_ROOT}" \
  --out-root "${P2A1_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink --oversample-rare --clean

# ── Step 5: Preflight P2-A1 ─────────────────────────────────────────────────
echo ""
echo "Step 5: Preflight P2-A1..."
python "${REPO}/scripts/preflight_p2_a1.py" \
  --config "${REPO}/configs/ultralytics/p2_a1_rtdetr_bdd_retrieval.yaml" \
  --pool-stats "${POOL_ROOT}/pool_stats.json" \
  --retrieval-stats "${RETRIEVED_ROOT}/retrieval_stats.json"

echo ""
echo "P2-A1 DATA PREPARATION: PASS"
echo ""
python3 -c "
import json
pool = json.load(open('${POOL_ROOT}/pool_stats.json'))
ret  = json.load(open('${RETRIEVED_ROOT}/retrieval_stats.json'))
print(f'  Pool: full_bdd_train={pool[\"full_train_pairs\"]}  excluded={pool[\"excluded_existing_names\"]}  pool={pool[\"candidate_pool\"]}')
print(f'  Retrieval: pool_after_class_filter={ret.get(\"candidate_pool_after_class_filter\",\"?\")}  selected={ret[\"selected_unique\"]}  duplicates_removed={ret.get(\"duplicate_candidate_hits_removed\",\"?\")}')
print(f'  Embedding: layers={ret.get(\"embedding_layers\",\"?\")}  dim={ret.get(\"embedding_dim\",\"?\")}')
print(f'  Output counts: images={ret.get(\"output_image_count\",\"?\")}  labels={ret.get(\"output_label_count\",\"?\")}  manifest={ret.get(\"manifest_row_count\",\"?\")}')
" 2>/dev/null || echo "  (stats unavailable)"
echo "  DO NOT run training automatically — see train_p2_a1_vast.sh"
