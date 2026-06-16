# Research Protocol

## Research Goal

The thesis focuses on people and vehicle detection under adverse weather
conditions for intelligent traffic surveillance.

The practical objective is not to build a production detector immediately. The
main objective is to run controlled comparisons and ablation studies:

```text
How much does each model or improvement change Precision, Recall, mAP50,
mAP50-95, inference time, FPS, and parameter count?
```

## Target Classes

```text
person, bicycle, car, motorcycle, bus, truck
```

## Training Flow

The canonical flow is:

```text
COCO pretrained weights
        ↓
Stage 1: BDD100K 6-class driving-domain adaptation
        ↓
Stage 2: DAWN 6-class adverse-weather fine-tuning
        ↓
Evaluation and comparison
```

Reason:

```text
COCO -> DAWN directly forces the model to learn driving domain + adverse weather
from a small DAWN dataset.

COCO -> BDD100K -> DAWN separates the problem:
1. learn driving-domain objects first;
2. then adapt to adverse weather.
```

## Benchmark Matrix

Main detector comparison:

| ID | Model | Trainer | Stage 1 Config | Stage 2 Config |
|---|---|---|---|---|
| B0 | YOLOv8n | Ultralytics | `stage1_bdd_yolov8n.yaml` | `stage2_dawn_yolov8n_from_bdd.yaml` |
| B1 | YOLOv8s | Ultralytics | `stage1_bdd_yolov8s.yaml` | `stage2_dawn_yolov8s_from_bdd.yaml` |
| B2 | YOLO11n | Ultralytics | `stage1_bdd_yolo11n.yaml` | `stage2_dawn_yolo11n_from_bdd.yaml` |
| B3 | YOLOv10n | Ultralytics | `stage1_bdd_yolov10n.yaml` | `stage2_dawn_yolov10n_from_bdd.yaml` |
| B4 | RT-DETR | Ultralytics | `stage1_bdd_rtdetr.yaml` | `stage2_dawn_rtdetr_from_bdd.yaml` |
| B5 | Faster R-CNN | TorchVision | `stage1_bdd_faster_rcnn.yaml` | `stage2_dawn_faster_rcnn_from_bdd.yaml` |

Faster R-CNN represents the two-stage detector family against the one-stage YOLO
models. It is trained with `scripts/train_torchvision.py` and evaluated with
`scripts/evaluate_torchvision.py`, which writes the same metric JSON format so
results sit in the same comparison table as the Ultralytics models.

Note on comparability: YOLO `precision`/`recall` come from Ultralytics at its
own operating point, while Faster R-CNN `precision`/`recall` are computed at a
fixed (conf 0.25, IoU 0.5) point. `mAP50` and `mAP50-95` are COCO-style for both
trainers and are the primary comparison metrics.

## Ablation Matrix

Ablations should be run after the main Stage 1 -> Stage 2 benchmark is stable.

| ID | Change | Status | Purpose |
|---|---|---|---|
| A0 | YOLOv8n Stage 2 baseline | implemented | Reference detector |
| A1 | YOLOv8n + CBAM Neck | implemented | Test attention mechanism |
| A2 | YOLOv8n + weather augmentation | planned | Test training-data robustness |
| A3 | YOLOv8n + dehazing preprocessing | planned | Test image-restoration preprocessing |

For a fair ablation, only one factor should change at a time. Dataset split,
pretrained source, image size, optimizer, epochs, early stopping, and evaluation
protocol should stay fixed unless the experiment explicitly studies them.

## Metrics

Report:

```text
Precision
Recall
mAP50
mAP50-95
Ultralytics inference time
Batch-1 inference time
FPS
Parameter count
```

Distinguish these two quantities:

```text
Absolute improvement = improved_mAP - baseline_mAP
Relative improvement (%) = (improved_mAP - baseline_mAP) / baseline_mAP * 100
```

Example:

```text
mAP50-95: 0.400 -> 0.420
absolute improvement = 0.020 = 2.0 percentage points
relative improvement = 5.0%
```

## Recommended Reporting Table

| Model | Precision | Recall | mAP50 | mAP50-95 | Params | ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n Stage 2 | ... | ... | ... | ... | ... | ... | ... |
| YOLOv8s Stage 2 | ... | ... | ... | ... | ... | ... | ... |
| YOLO11n Stage 2 | ... | ... | ... | ... | ... | ... | ... |
| YOLOv8n + CBAM | ... | ... | ... | ... | ... | ... | ... |

If time allows, report AP per class. DAWN is small, so class imbalance can hide
weak performance on bicycle, motorcycle, bus, or truck.

## Threats to Validity

- DAWN is small, so one seed may produce unstable conclusions.
- Test-set tuning makes results too optimistic. Use validation results for model
  decisions and reserve test split for final reporting.
- BDD100K and DAWN annotation quality may differ, which can affect transfer
  learning results.
- FPS depends on hardware, precision mode, batch size, image size, warmup, and
  measurement method.

## Per-Weather and Per-Class Reporting

- `scripts/evaluate.py` saves a `per_class` block (AP50 and AP50-95 for each of
  the 6 classes) so weak rare classes stay visible.
- `scripts/evaluate_by_weather.py` splits the chosen DAWN split by weather
  (fog/rain/sand/snow/night, read from `manifest.csv`) and reports mAP per
  condition. This is the core "adverse weather" result and is more informative
  than a single aggregate mAP.

## Handling Seed Instability

Because DAWN is small, report mean ± std over several seeds for the headline
comparisons (baseline vs CBAM, and across detectors). Use:

```text
python scripts/run_seeds.py --config <config> --seeds 0 1 2 --split val
```

It trains and evaluates each seed in its own run folder and writes a
`<split>_seeds_summary.json` with mean ± std for Precision/Recall/mAP50/mAP50-95.

## Practical Execution Order

1. Train Stage 1 YOLOv8n on BDD100K.
2. Train Stage 2 YOLOv8n on DAWN.
3. Evaluate YOLOv8n Stage 2 and confirm the pipeline works.
4. Repeat Stage 1 and Stage 2 for other Ultralytics models and Faster R-CNN.
5. Run CBAM ablation against YOLOv8n Stage 2.
6. Collect results and compare against the chosen baseline.
7. Report per-class AP, per-weather mAP, and mean ± std over seeds for the
   headline comparisons.
