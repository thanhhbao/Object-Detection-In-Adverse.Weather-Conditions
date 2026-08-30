#!/usr/bin/env bash
set -euo pipefail

# Configurable
POOL_ROOT="${POOL_ROOT:-/workspace/datasets_noleak/bdd_remaining_pool}"
XWOD_ROOT="${XWOD_ROOT:-/workspace/datasets_noleak/xwod_6cls_yolo}"
ACDC_ROOT="${ACDC_ROOT:-/workspace/datasets_noleak/acdc_6cls_yolo}"
BDD30K_ROOT="${BDD30K_ROOT:-/workspace/datasets_noleak/bdd100k_6cls_yolo}"
PHASE2_CKPT="${PHASE2_CKPT:-/workspace/runs/phase2_final_rtdetr/weights/best.pt}"

RANDOM_RARE_ROOT="${RANDOM_RARE_ROOT:-/workspace/datasets_noleak/bdd_random_rare}"
DINO_RETRIEVED_ROOT="${DINO_RETRIEVED_ROOT:-/workspace/datasets_noleak/bdd_dinov2_retrieved}"
A0R_MERGED="${A0R_MERGED:-/workspace/datasets_noleak/phase2_a0r_merged_yolo}"
A1_DINO_MERGED="${A1_DINO_MERGED:-/workspace/datasets_noleak/phase2_a1_dino_merged_yolo}"

REPO="${REPO:-/workspace/Object-Detection}"
PYTHONPATH="${PYTHONPATH:-}:${REPO}/src"
export PYTHONPATH

echo "=== ABLATION SETUP: A0R + A1-DINO ==="

# Step 0: Test gate
echo "Step 0: Protocol test gate..."
python -m pytest -q \
  "${REPO}/tests/test_retrieve_dinov2.py" \
  "${REPO}/tests/test_build_random_rare_control.py" \
  2>&1 | tee /tmp/ablation_pytest.log
if grep -qE "^FAILED|^ERROR" /tmp/ablation_pytest.log; then
  echo "ERROR: Protocol tests failed. Fix before running setup."
  exit 1
fi

# Step 1: Verify BDD pool exists (built by setup_p2_a1_vast.sh earlier)
echo ""
echo "Step 1: Verify BDD remaining pool..."
if [ ! -f "${POOL_ROOT}/pool_stats.json" ]; then
  echo "ERROR: BDD pool not found at ${POOL_ROOT}."
  echo "  Run setup_p2_a1_vast.sh first (Steps 1-2) to build the pool."
  exit 1
fi
POOL_COUNT=$(python3 -c "import json; d=json.load(open('${POOL_ROOT}/pool_stats.json')); print(d['candidate_pool'])")
echo "  Pool verified: ${POOL_COUNT} candidates"
python3 -c "
import json
d=json.load(open('${POOL_ROOT}/pool_stats.json'))
assert d['overlap_with_used_train']==0 and d['overlap_with_used_val']==0 and d['overlap_with_used_test']==0, 'POOL OVERLAP INVARIANT VIOLATED'
print('  Pool overlap invariants: OK')
"

# Step 2: Build A0R — random rare control 5K
echo ""
echo "Step 2: Build A0R (random rare 5K)..."
python "${REPO}/scripts/build_random_rare_control.py" \
  --pool-root "${POOL_ROOT}" \
  --out-root "${RANDOM_RARE_ROOT}" \
  --top-k 5000 \
  --candidate-target-classes 1 3 4 \
  --seed 42 \
  --mode symlink \
  --used-bdd-root "${BDD30K_ROOT}" \
  --clean

# Step 3: Build A1-DINO — DINOv2 retrieval 5K
echo ""
echo "Step 3: Build A1-DINO (DINOv2 retrieval 5K)..."
python "${REPO}/scripts/retrieve_dinov2.py" \
  --weights "${PHASE2_CKPT}" \
  --pool-root "${POOL_ROOT}" \
  --query-root "${XWOD_ROOT}/images/train" \
  --query-root "${ACDC_ROOT}/images/train" \
  --out-root "${DINO_RETRIEVED_ROOT}" \
  --top-k 5000 \
  --similarity-threshold 0.70 \
  --target-classes 1 3 4 \
  --candidate-target-classes 1 3 4 \
  --mode symlink \
  --used-bdd-root "${BDD30K_ROOT}" \
  --seed 42 \
  --clean

# Step 4: Build A0R merged dataset
echo ""
echo "Step 4: Build phase2_a0r_merged..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${RANDOM_RARE_ROOT}" \
  --out-root "${A0R_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink --oversample-rare --clean

# Step 5: Build A1-DINO merged dataset
echo ""
echo "Step 5: Build phase2_a1_dino_merged..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${DINO_RETRIEVED_ROOT}" \
  --out-root "${A1_DINO_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink --oversample-rare --clean

# Step 6: Audit — hard-fail unless each arm has exactly 5000 images/labels/manifest and zero overlap
echo ""
echo "Step 6: Audit all ablation arms (exact-5000 + zero-overlap enforcement)..."
python3 -c "
import json, sys

REQUIRED_COUNT = 5000

arms = {
  'A0R':     '${RANDOM_RARE_ROOT}/random_rare_stats.json',
  'A1-DINO': '${DINO_RETRIEVED_ROOT}/retrieval_stats.json',
}

errors = []
for arm, stats_path in arms.items():
    try:
        d = json.load(open(stats_path))
        sel = d.get('selected', d.get('selected_unique', '?'))
        img_c = d.get('output_image_count', '?')
        lbl_c = d.get('output_label_count', '?')
        man_c = d.get('manifest_row_count', '?')
        ovl   = d.get('overlap_with_used_bdd', '?')
        print(f'  {arm}: selected={sel}  images={img_c}  labels={lbl_c}  manifest={man_c}  overlap_bdd={ovl}')

        # Exact count check
        for field, val in [('output_image_count', img_c), ('output_label_count', lbl_c), ('manifest_row_count', man_c)]:
            if val != REQUIRED_COUNT:
                errors.append(f'{arm}: {field}={val}, expected {REQUIRED_COUNT}')

        # Zero overlap check
        if ovl != 0:
            errors.append(f'{arm}: overlap_with_used_bdd={ovl}, expected 0')

        # Cross-arm count parity is enforced by both being REQUIRED_COUNT

    except Exception as e:
        errors.append(f'{arm}: failed to read stats — {e}')

if errors:
    print()
    print('AUDIT FAILED:')
    for e in errors:
        print(f'  ERROR: {e}')
    sys.exit(1)

print()
print('Audit: PASS — both arms have exactly 5000 images/labels/manifest rows, zero overlap with used BDD.')
"

echo ""
echo "=== ABLATION SETUP: COMPLETE ==="
echo ""
echo "Next: share stats with researcher before training."
echo "DO NOT train until stats are reviewed."
echo "Training commands (explicit mode required):"
echo "  ./scripts/train_p2_a1_vast.sh a0r"
echo "  ./scripts/train_p2_a1_vast.sh a1_dino"
