# P2-A1 Experiment Protocol: Failure-Driven BDD Retrieval

## Research Question

Does selectively adding BDD100K images that are visually similar to Phase 2 failure
cases (hard samples on rare classes: bicycle, motorcycle, bus) improve mAP50:95 beyond
the Phase 2 baseline, compared with an equal-epochs control run without retrieved data?

This is **failure-driven similarity retrieval**, not classical human-in-the-loop active
learning. There is no oracle labelling step. Hard samples are detected automatically
via GT-aware per-box IoU matching, and retrieved candidates are selected purely by
cosine similarity in the embedding space.

---

## RT-DETR-L Architecture and Multi-Scale Embedding

The Phase 2 checkpoint is RT-DETR-L. The relevant backbone/head structure:

```
backbone:
  0: HGStem [32, 48]
  1: HGBlock [48, 128, 3]      # stage 1
  2: DWConv [128, 3, 2]
  3: HGBlock [96, 512, 3]      # stage 2
  4: DWConv [512, 3, 2]
  5–7: HGBlock [192, 1024, 5]  # stage 3
  8: DWConv [1024, 3, 2]
  9: HGBlock [384, 2048, 5]    # stage 4  ← HGBlock, NOT SPPF

head:
  ...
  21: RepC3 [256]    # X3 / P3 feature  ← hook here
  22: Conv [256, 3, 2]
  23: Concat
  24: RepC3 [256]    # F4 / P4 feature  ← hook here
  25: Conv [256, 3, 2]
  26: Concat
  27: RepC3 [256]    # F5 / P5 feature  ← hook here
  28: RTDETRDecoder [nc]
```

**IMPORTANT:** Layer 9 is `HGBlock` (backbone stage 4), not SPPF. Using layer 9 alone
for embeddings would give a single-scale, backbone-only representation. For RT-DETR-L,
the FPN head fuses multi-scale features, and the P3/P4/P5 RepC3 outputs (layers 21, 24,
27) are the correct embedding points.

### Embedding Construction

1. Register forward hooks on layers 21 (P3), 24 (P4), and 27 (P5).
2. For each layer output `[B, C, H, W]`, apply Global Average Pooling → `[B, 256]`.
3. Concatenate the three GAP outputs along the channel dimension → `[B, 768]`.
4. L2-normalize each row → cosine similarities are computed as dot products.

```
layer 21: [B, 256, H3, W3]  --GAP-->  [B, 256]
layer 24: [B, 256, H4, W4]  --GAP-->  [B, 256]   concat --> [B, 768]  --L2norm-->  [B, 768]
layer 27: [B, 256, H5, W5]  --GAP-->  [B, 256]
```

Final embedding dimension: **768**.

Implemented in `MultiScaleHook` in `scripts/active_retrieval.py`.

---

## Candidate Pool Definition

**BDD original TRAIN minus BDD30K used** = all image/label pairs in the full BDD100K
training split that do NOT appear (by filename basename) in any split of the BDD30K
dataset already included in Phase 2.

```
pool = {img ∈ bdd_full/images/train | img.name ∉ (bdd30k_train ∪ bdd30k_val ∪ bdd30k_test)}
```

Enforced by `build_bdd_retrieval_pool.py` with three hard invariants (RuntimeError on
violation):

- `overlap_with_used_train == 0`
- `overlap_with_used_val == 0`
- `overlap_with_used_test == 0`

### Candidate Class Filter

The pool is further filtered to images containing at least one label with class
1 (bicycle), 3 (motorcycle), or 4 (bus). This reduces the effective pool to images that
can contribute examples for the rare target classes.

Controlled by `--candidate-target-classes 1 3 4` in `active_retrieval.py`.
Set `--candidate-target-classes -1` to disable filtering.

---

## Why BDD Val/Test Are Excluded from the Pool

BDD val and BDD test are held-out evaluation splits. Including their images in training
data — even indirectly via retrieval — would constitute leakage. The pool is restricted
to `bdd_full/images/train` only.

---

## Why Query/Hard Mining Uses Training Data Only

Hard sample mining identifies images where the Phase 2 model fails. We use XWOD train
and ACDC train as query sets because:

1. They are in-domain (adverse weather) and representative of the model's failure modes.
2. Using val or test images as queries would leak label information into the retrieval
   decision, which influences which training images are added.
3. Separating query (train) from pool (novel BDD) ensures the hard-sample criterion is
   computed only on data already seen during Phase 2 training.

`validate_query_root()` in `active_retrieval.py` enforces this at runtime and raises
`SystemExit` if a val or test path is passed.

---

## Hard Sample Definition (GT-Aware Per-Box IoU Matching)

A **GT box is hard** if any of the following hold for the same-class predictions:

- **Missed GT**: No same-class prediction exists with IoU >= `match_iou` (default 0.5).
  `matched_confidence = 0.0`, `hardness = 1.0`.
- **Weak detection**: A same-class prediction with IoU >= `match_iou` exists, but its
  confidence < `conf_hard` (default 0.25).
  `matched_confidence = best_conf`, `hardness = 1.0 - best_conf`.

An **image is hard** if at least one GT box is hard.

**Hardness score** = `max(1.0 - matched_confidence)` over all hard GT boxes in the image.

```python
hardness_score = max(1.0 - matched_conf for each hard GT box)
# completely missed GT → matched_conf=0.0 → hardness=1.0
# weak detection at conf=0.1 → hardness=0.9
```

This replaces the legacy image-level max-confidence approach, which did not distinguish
between missing a GT box vs. predicting a different class at the same location.

Implemented in `is_gt_hard()` and `find_hard_samples_gt_aware()`.

---

## Deduplication Statistics

The retrieval step tracks:

| Statistic | Definition |
|-----------|------------|
| `candidate_hits_above_threshold` | Total (query, pool) pairs with cosine sim >= threshold |
| `unique_candidates_before_top_k` | Number of unique pool images with at least one hit |
| `duplicate_candidate_hits_removed` | `candidate_hits_above_threshold - unique_candidates_before_top_k` |
| `selected_unique` | Final count after top-k selection |

When multiple hard queries match the same pool image, only the hit with the highest
similarity is kept, and the query provenance for that pool image is attributed to the
highest-sim query.

---

## Query Provenance

Each retrieved pool image records which hard query drove its selection:

| Field | Content |
|-------|---------|
| `query_image` | Absolute path to the hard query image |
| `query_dataset` | Inferred dataset name (`xwod` or `acdc` from path) |
| `hardness_score` | Hardness score of the driving query image |
| `similarity` | Cosine similarity between query and retrieved image |

Written to `retrieved_manifest.csv` with columns:
```
retrieved_image, retrieved_label, source_image_name, source_split,
query_image, query_dataset, similarity, hardness_score, rank, selected_reason
```

---

## Embedding Cache Validation

Pool embeddings are cached to `.npz`. On reload, the following metadata fields are
compared against the current run to detect stale caches:

| Field | Why |
|-------|-----|
| `checkpoint` | Path to the weights file |
| `checkpoint_size` | File size — catches replaced checkpoints at same path |
| `checkpoint_sha256_prefix` | SHA-256 of first 64KB — fast partial hash |
| `embedding_layers` | Layer indices used — different layers → different embeddings |
| `imgsz` | Input image size |
| `pool_count` | Number of pool images — catches pool rebuild |
| `version` | Cache format version (currently 2) |

Any mismatch triggers a full rebuild. Implemented in `_cache_fingerprint()`,
`load_pool_cache()`, and `save_pool_cache()`.

---

## Experimental Conditions

| Condition | Dataset | From checkpoint | Epochs | Purpose |
|-----------|---------|----------------|--------|---------|
| Phase 2 baseline | phase2_merged | stage2_xwod_rtdetr_from_bdd30k | 50 | Main Phase 2 model |
| P2-A0C control | phase2_merged | phase2_final_rtdetr | 20 | Isolates continued-training effect |
| P2-A1 retrieval | phase2_a1_merged | phase2_final_rtdetr | 20 | Tests retrieval benefit |

Both A0C and A1 use **identical** hyperparameters (epochs=20, batch=16, lr0=0.00001,
seed=42, patience=8). The only difference is the dataset: A0C uses the original
phase2_merged; A1 uses phase2_a1_merged (= phase2_merged + retrieved BDD images).

---

## Why P2-A0C Is Required

Without a control run, any improvement in P2-A1 could be explained by:

- Additional training epochs (20 extra vs. the Phase 2 final model)
- Better convergence from continued fine-tuning on the existing data

P2-A0C controls for both confounders by continuing training for the same 20 epochs on
the same dataset (phase2_merged), using the same checkpoint and hyperparameters. Only
the `dataset` key differs between A0C and A1.

The claim "retrieved data helps" is only defensible if `P2-A1 mAP > P2-A0C mAP`.

---

## Vast.ai Setup Stages (setup_p2_a1_vast.sh)

| Step | Script | Action |
|------|--------|--------|
| 0 | `smoke_p2_a1_embedding.py` | Load model, register hooks on layers 21/24/27, forward dummy input, verify embedding shapes and L2-norm. Fails fast if model/hooks are wrong. |
| 1 | `prepare_bdd100k_yolo.py --full` | Prepare full BDD100K 6-class YOLO format if not already done. Skipped if `dataset.yaml` exists. |
| 2 | `build_bdd_retrieval_pool.py` | Build BDD remaining pool = full BDD train minus BDD30K used. Enforces overlap=0 invariants. |
| 3 | `active_retrieval.py` | GT-aware hard mining on XWOD/ACDC train, multi-scale embedding retrieval from pool (layers 21/24/27, sim >= 0.75, top-5000). |
| 4 | `build_phase2_dataset.py` | Merge XWOD+ACDC+BDD30K+retrieved into phase2_a1_merged. |
| 5 | `preflight_p2_a1.py` | Config parity check (A0C vs A1), leakage manifest check, pool stats validation. |

**The setup script does NOT train.** Training is triggered separately by `train_p2_a1_vast.sh`.

---

## Primary Metric

**mAP50:95** on the frozen XWOD validation set (1,001 images).

Secondary metrics: per-class AP for bicycle (class 1), motorcycle (class 3), bus (class
4) — the rare classes targeted by hard mining.

---

## Held-Out Test Sets Remain Frozen

Final evaluation is done separately on:
- XWOD test
- ACDC test
- DAWN test
- BDD test

None of these sets appear in phase2_a1_merged (enforced by pool construction hard
invariants and manifest leakage checks in `preflight_p2_a1.py`).

---

## Leakage Invariants Summary

| Check | Where enforced |
|-------|---------------|
| Pool ∩ used_bdd_train == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Pool ∩ used_bdd_val == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Pool ∩ used_bdd_test == 0 | `build_bdd_retrieval_pool.py` (RuntimeError) |
| Retrieved ∩ used_bdd == 0 | `active_retrieval.py --used-bdd-root` (RuntimeError) |
| Query roots are train-only | `active_retrieval.py validate_query_root()` (SystemExit) |
| No DAWN in merged train | `preflight_p2_a1.py` manifest check |
| No val/test path leakage | `preflight_p2_a1.py` manifest check |
| A0C == A1 hyperparams | `preflight_p2_a1.py` config parity check |
| Both init from phase2_final_rtdetr | `preflight_p2_a1.py` config parity check |

---

## No Improvement Claimed Before Results Exist

This document describes the experimental protocol and the leakage safeguards.
No claim is made that P2-A1 will outperform P2-A0C or the Phase 2 baseline.
Results will be reported after training completes on the remote GPU environment.
