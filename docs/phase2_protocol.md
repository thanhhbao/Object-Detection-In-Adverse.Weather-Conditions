# Phase 2 Protocol — RT-DETR-L Final Fine-Tune

## 1. Why RT-DETR-L was selected

Stage 2 (fine-tuned from Stage 1 BDD100K weights on XWOD) was benchmarked across four
detectors. RT-DETR-L had the best mAP50-95 on both the primary adverse-weather test set
(XWOD) and the zero-shot generalization set (DAWN), so it is the single model carried
into Phase 2.

## 2. Stage 2 benchmark summary

XWOD test mAP50-95:

| Model | mAP50-95 |
|---|---|
| RT-DETR-L | 0.500 |
| YOLO11n | 0.454 |
| YOLOv8n | 0.452 |
| Faster R-CNN | 0.381 |

DAWN test mAP50-95:

| Model | mAP50-95 |
|---|---|
| RT-DETR-L | 0.526 |
| Faster R-CNN | 0.449 |
| YOLOv8n | 0.414 |
| YOLO11n | 0.400 |

RT-DETR-L wins on both, so it is the **only** model carried into Phase 2 — the other three
are not re-trained here.

## 3. Phase 2 goal

Fine-tune the Stage 2 RT-DETR-L checkpoint (`stage2_xwod_rtdetr_from_bdd30k/weights/best.pt`)
on a merged training set that combines more adverse-weather data (ACDC) with a small BDD
replay slice, to further improve adverse-weather robustness without erasing what Stage 1/2
already learned on clear-weather driving scenes.

## 4. Training composition

Merged train set (`phase2_merged`), built by `scripts/build_phase2_dataset.py`:

| Source | Split | Images |
|---|---|---|
| XWOD | train | 6006 |
| ACDC | train | 1182 |
| BDD100K | train (replay) | 2000 (seed=42, deterministic) |

## 5. Rare-class oversampling (train only)

Applied only to the merged train split, never to validation:

- bicycle ×2
- motorcycle ×3
- bus ×3
- all other classes ×1

## 6. Validation

Validation = **XWOD val only** (1001 images). No other split is used for early stopping or
model selection during Phase 2.

## 7. Test sets are never used for optimization

None of the following are ever included in Phase 2 training or validation:
XWOD test, ACDC val, ACDC test, DAWN val, DAWN test, BDD val, BDD test.

## 8. Build the merged dataset (Vast.ai)

```bash
bash scripts/build_phase2_vast.sh
```

Equivalent to:

```bash
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
```

## 9. Preflight (run before spending any GPU time)

```bash
export OD_PATHS=configs/common/paths_vast.yaml
python scripts/preflight_phase2.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml
```

`scripts/build_phase2_vast.sh` runs this automatically after the build.

## 10. Train

```bash
bash scripts/train_phase2_rtdetr_vast.sh
```

Equivalent to:

```bash
export OD_PATHS=configs/common/paths_vast.yaml
python scripts/train_ultralytics.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml
```

Config: `configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml`
(`from_run: stage2_xwod_rtdetr_from_bdd30k`, `dataset: phase2_merged`, `batch: 4`,
`lr0: 0.00005`, `patience: 20`; `epochs`, `imgsz`, `optimizer`, `seed`, `deterministic`,
`amp`, `cos_lr` are inherited from `configs/common/train_defaults.yaml`).

## 11. Final evaluation protocol (after Phase 2 training only)

Run separately on all four held-out test sets — none of them were used during training:

- XWOD test
- ACDC test
- DAWN test
- BDD test

```bash
python scripts/evaluate.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml \
  --split test --data /workspace/datasets_noleak/xwod_6cls_yolo/dataset.yaml \
  --metrics-tag xwod_test

python scripts/evaluate.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml \
  --split test --data /workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
  --metrics-tag acdc_test

python scripts/evaluate.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml \
  --split test --data /workspace/datasets_noleak/dawn_6cls_yolo/dataset.yaml \
  --metrics-tag dawn_test

python scripts/evaluate.py \
  --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml \
  --split test --data /workspace/datasets_noleak/bdd100k_6cls_yolo/dataset.yaml \
  --metrics-tag bdd_test
```

## 12. ACDC pre-Phase-2 baseline

Phase 2 trains on ACDC train, so improvement on ACDC test can only be measured against a
**pre-Phase-2 baseline**: the Stage 2 RT-DETR-L checkpoint evaluated on ACDC test, before
Phase 2 ever sees ACDC train.

```bash
export OD_PATHS=configs/common/paths_vast.yaml
python scripts/evaluate.py \
  --config configs/ultralytics/stage2_xwod_rtdetr_from_bdd30k.yaml \
  --split test --data /workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
  --metrics-tag acdc_test_baseline
```

This is a **PRE-PHASE-2 baseline**. It must not use ACDC train — it only evaluates the
Stage 2 checkpoint, which has never seen any ACDC data.

## Notes

- BDD replay (2000 images, train split only) exists to reduce catastrophic forgetting of
  the clear-weather driving domain after Stage 2, without letting BDD dominate the
  adverse-weather signal (XWOD + ACDC train = 7188 images vs. 2000 BDD replay).
- Phase 2 is performed **only** on the selected RT-DETR-L checkpoint, not on YOLOv8n,
  YOLO11n, or Faster R-CNN.
- No results are claimed here — Phase 2 has not been trained yet. This document only
  fixes the reproducible protocol.
