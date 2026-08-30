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
# --exclude-retrieved-from-oversampling ensures only the official Phase2 portion
# (XWOD + ACDC + BDD30K) is oversampled, keeping duplicate count = 13797 for both arms.
echo ""
echo "Step 4: Build phase2_a0r_merged..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${RANDOM_RARE_ROOT}" \
  --out-root "${A0R_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink \
  --oversample-rare --exclude-retrieved-from-oversampling \
  --clean

# Step 5: Build A1-DINO merged dataset
echo ""
echo "Step 5: Build phase2_a1_dino_merged..."
python "${REPO}/scripts/build_phase2_dataset.py" \
  --xwod-root "${XWOD_ROOT}" \
  --acdc-root "${ACDC_ROOT}" \
  --bdd-root "${BDD30K_ROOT}" \
  --retrieved-root "${DINO_RETRIEVED_ROOT}" \
  --out-root "${A1_DINO_MERGED}" \
  --bdd-use-all --seed 42 --mode symlink \
  --oversample-rare --exclude-retrieved-from-oversampling \
  --clean

# Step 6: Full strict audit
# Checks both retrieval stats AND merged dataset stats.
# Hard-fail unless BOTH arms satisfy all invariants.
echo ""
echo "Step 6: Full strict audit (retrieval + merged dataset invariants)..."
python3 -c "
import json, sys

REQUIRED_K     = 5000
EXPECTED_BASE  = 37188   # XWOD(6006) + ACDC(1182) + BDD30K(30000)
EXPECTED_TOTAL_BEFORE_OS = 42188   # base + 5000 retrieved
EXPECTED_DUPS  = 13797   # official Phase2 rare-class duplicates (base only)
EXPECTED_TRAIN = 55985   # 42188 + 13797
EXPECTED_VAL   = 1001    # XWOD val

retrieval_stats = {
  'A0R':     '${RANDOM_RARE_ROOT}/random_rare_stats.json',
  'A1-DINO': '${DINO_RETRIEVED_ROOT}/retrieval_stats.json',
}
merged_stats = {
  'A0R':     '${A0R_MERGED}/stats.json',
  'A1-DINO': '${A1_DINO_MERGED}/stats.json',
}

errors = []

def chk(arm, field, actual, expected):
    if actual != expected:
        errors.append(f'{arm}: {field}={actual!r}, expected {expected!r}')

# ── Retrieval stats ──────────────────────────────────────────────────────────
for arm, path in retrieval_stats.items():
    try:
        d = json.load(open(path))
        sel = d.get('selected', d.get('selected_unique'))
        img = d.get('output_image_count')
        lbl = d.get('output_label_count')
        man = d.get('manifest_row_count')
        ovl = d.get('overlap_with_used_bdd')
        print(f'  [{arm} retrieval] selected={sel} images={img} labels={lbl} manifest={man} overlap={ovl}')
        chk(arm, 'retrieval.selected',            sel, REQUIRED_K)
        chk(arm, 'retrieval.output_image_count',  img, REQUIRED_K)
        chk(arm, 'retrieval.output_label_count',  lbl, REQUIRED_K)
        chk(arm, 'retrieval.manifest_row_count',  man, REQUIRED_K)
        chk(arm, 'retrieval.overlap_with_used_bdd', ovl, 0)
    except Exception as e:
        errors.append(f'{arm}: cannot read retrieval stats — {e}')

# ── Merged dataset stats ─────────────────────────────────────────────────────
train_totals = {}
dup_counts = {}
for arm, path in merged_stats.items():
    try:
        d = json.load(open(path))
        ret_n  = d.get('retrieved_train')
        base_b = d.get('base_train_total_before_retrieval')
        total_b= d.get('train_total_with_retrieval_before_oversampling')
        dups   = d.get('oversample_rare', {}).get('duplicate_images')
        excl   = d.get('oversample_rare', {}).get('exclude_retrieved_from_oversampling')
        train  = d.get('train_total')
        val    = d.get('val_total')
        print(f'  [{arm} merged] retrieved={ret_n} base={base_b} before_os={total_b} '
              f'dups={dups} excl_ret={excl} train={train} val={val}')
        chk(arm, 'merged.retrieved_train',                    ret_n,  REQUIRED_K)
        chk(arm, 'merged.base_train_total_before_retrieval',  base_b, EXPECTED_BASE)
        chk(arm, 'merged.train_total_with_retrieval_before_oversampling', total_b, EXPECTED_TOTAL_BEFORE_OS)
        chk(arm, 'merged.oversample_rare.duplicate_images',   dups,   EXPECTED_DUPS)
        chk(arm, 'merged.oversample_rare.exclude_retrieved',  excl,   True)
        chk(arm, 'merged.train_total',  train, EXPECTED_TRAIN)
        chk(arm, 'merged.val_total',    val,   EXPECTED_VAL)
        train_totals[arm] = train
        dup_counts[arm]   = dups
    except Exception as e:
        errors.append(f'{arm}: cannot read merged stats — {e}')

# ── Cross-arm parity ─────────────────────────────────────────────────────────
if len(train_totals) == 2:
    arms = list(train_totals)
    if train_totals[arms[0]] != train_totals[arms[1]]:
        errors.append(f'cross-arm: train_total mismatch {train_totals}')
    if dup_counts.get(arms[0]) != dup_counts.get(arms[1]):
        errors.append(f'cross-arm: duplicate_images mismatch {dup_counts}')

# ── Result ───────────────────────────────────────────────────────────────────
if errors:
    print()
    print('AUDIT FAILED:')
    for e in errors:
        print(f'  ERROR: {e}')
    sys.exit(1)

print()
print(f'Audit: PASS')
print(f'  Both arms: retrieved=5000  train_total={EXPECTED_TRAIN}  val={EXPECTED_VAL}')
print(f'  Base Phase2 duplicates={EXPECTED_DUPS} (retrieved_bdd_* excluded from oversampling)')
print(f'  Cross-arm train counts identical: A0R == A1-DINO == {EXPECTED_TRAIN}')
"

echo ""
echo "=== ABLATION SETUP: COMPLETE ==="
echo ""
echo "Next: share stats with researcher before training."
echo "DO NOT train until stats are reviewed."
echo "Training commands (explicit mode required):"
echo "  ./scripts/train_p2_a1_vast.sh a0r"
echo "  ./scripts/train_p2_a1_vast.sh a1_dino"
