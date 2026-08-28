# P2-A1 Experiment Protocol: Failure-Driven BDD Retrieval

## Research Question

Does selectively adding BDD100K images that are visually similar to Phase 2 failure
cases (hard samples on rare classes: bicycle, motorcycle, bus) improve mAP50:95 beyond
the Phase 2 baseline, compared with an equal-epochs control run without retrieved data?

---

## Candidate Pool Definition

**BDD original TRAIN minus BDD30K used** = all image/label pairs in the full BDD100K
training split that do NOT appear (by filename basename) in any split of the BDD30K
dataset already included in Phase 2.

Formally:

```
pool = {img ∈ bdd_full/images/train | img.name ∉ (bdd30k_train ∪ bdd30k_val ∪ bdd30k_test)}
```

This is enforced by `build_bdd_retrieval_pool.py` with three hard invariants
(RuntimeError on violation):

- `overlap_with_used_train == 0`
- `overlap_with_used_val == 0`
- `overlap_with_used_test == 0`

---

## Why BDD Val/Test Are Excluded from the Pool

BDD val and BDD test are held-out evaluation splits.  Including their images in
training data — even indirectly via retrieval — would constitute leakage.  The pool is
restricted to `bdd_full/images/train` only.  The overlap check against val and test
names ensures that future BDD dataset revisions (e.g., split re-assignments) do not
silently introduce leakage.

---

## Why Query/Hard Mining Uses Training Data Only

Hard sample mining identifies images where the Phase 2 model fails.  We use XWOD train
and ACDC train as query sets because:

1. They are in-domain (adverse weather) and representative of the model's failure modes.
2. Using val or test images as queries would leak label information into the retrieval
   decision, which influences which training images are added.
3. Separating query (train) from pool (novel BDD) ensures the hard-sample criterion is
   computed only on data already seen during Phase 2 training.

---

## Experimental Conditions

| Condition | Dataset | From checkpoint | Epochs | Purpose |
|-----------|---------|----------------|--------|---------|
| Phase 2 baseline | phase2_merged | stage2_xwod_rtdetr_from_bdd30k | 50 | Main Phase 2 model |
| P2-A0C control | phase2_merged | phase2_final_rtdetr | 20 | Isolates continued-training effect |
| P2-A1 retrieval | phase2_a1_merged | phase2_final_rtdetr | 20 | Tests retrieval benefit |

Both A0C and A1 use **identical** hyperparameters (epochs=20, batch=16, lr0=0.00001,
seed=42, patience=8).  The only difference is the dataset: A0C uses the original
phase2_merged; A1 uses phase2_a1_merged (= phase2_merged + retrieved BDD images).

---

## Why P2-A0C Is Required

Without a control run, any improvement in P2-A1 could be explained by:

- Additional training epochs (20 extra vs. the Phase 2 final model)
- Better convergence from continued fine-tuning on the existing data

P2-A0C controls for both confounders by continuing training for the same 20 epochs
on the same dataset (phase2_merged), using the same checkpoint and hyperparameters.
Only the `dataset` key differs between A0C and A1.

---

## Primary Metric

**mAP50:95** on the frozen XWOD validation set (1 001 images).

Secondary metrics: per-class AP for bicycle (class 1), motorcycle (class 3), bus (class
4) — the rare classes targeted by hard mining.

---

## Held-Out Test Sets Remain Frozen

Final evaluation is done separately on:
- XWOD test
- ACDC test
- DAWN test
- BDD test

None of these sets appear in phase2_a1_merged (enforced by the pool construction hard
invariants and the manifest leakage checks in `preflight_p2_a1.py`).

---

## No Improvement Claimed Before Results Exist

This document describes the experimental protocol and the leakage safeguards.
No claim is made that P2-A1 will outperform P2-A0C or the Phase 2 baseline.
Results will be reported after training completes on the remote GPU environment.

---

## Leakage Invariants Summary

| Check | Where enforced |
|-------|---------------|
| Pool ∩ used_bdd_train == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Pool ∩ used_bdd_val == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Pool ∩ used_bdd_test == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Retrieved ∩ used_bdd == 0 | `active_retrieval.py --used-bdd-root` (RuntimeError) |
| No DAWN in merged train | `preflight_p2_a1.py` manifest check |
| No val/test path leakage | `preflight_p2_a1.py` manifest check |
| A0C == A1 hyperparams | `preflight_p2_a1.py` config parity check |
| Both init from phase2_final_rtdetr | `preflight_p2_a1.py` config parity check |
