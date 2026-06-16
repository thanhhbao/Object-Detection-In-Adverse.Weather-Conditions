# Research and Enhancement of Deep Learning Models for People and Vehicle Detection in Adverse Weather Conditions for Intelligent Traffic Surveillance

This repository contains a progressive fine-tuning and ablation-study pipeline
for detecting people and vehicles under adverse weather conditions.

Target classes:

```text
person, bicycle, car, motorcycle, bus, truck
```

Core research flow:

```text
COCO pretrained detector
        ↓
Stage 1: BDD100K 6-class driving-domain adaptation
        ↓
Stage 2: DAWN 6-class adverse-weather fine-tuning
        ↓
Evaluation + model comparison + ablation study
```

## Project Structure

```text
project/
├── configs/
│   ├── ultralytics/       # YOLOv8, YOLO11, YOLOv10, RT-DETR experiments
│   ├── torchvision/       # Faster R-CNN experiments
│   ├── ablation/          # Architecture/preprocessing ablation experiments
│   └── common/            # Shared class names, Colab paths, train defaults
├── scripts/
│   ├── train_ultralytics.py
│   ├── train_torchvision.py
│   ├── train.py           # Custom YOLOv8 ablation trainer, currently CBAM
│   ├── evaluate.py
│   ├── eval_all.py
│   ├── collect_results.py
│   ├── compare_results.py
│   └── dataset preparation scripts
├── datasets/              # Local datasets, ignored by Git except .gitkeep
├── models/                # Custom YOLO YAML architectures
└── runs/                  # Training/evaluation outputs, ignored by Git except .gitkeep
```

Large datasets, checkpoints, and training outputs are intentionally not tracked
by Git.

## Config Logic

Experiment configs are intentionally small. Shared values live in:

```text
configs/common/train_defaults.yaml
configs/common/paths_colab.yaml
configs/common/class_names.yaml
```

Example Stage 1 config:

```yaml
defaults: configs/common/train_defaults.yaml
paths: configs/common/paths_colab.yaml

model: yolov8n.pt
dataset: bdd
name: stage1_bdd_yolov8n
```

Example Stage 2 config:

```yaml
defaults: configs/common/train_defaults.yaml
paths: configs/common/paths_colab.yaml

from_run: stage1_bdd_yolov8n
dataset: dawn
name: stage2_dawn_yolov8n_from_bdd

lr0: 0.0005
patience: 15
```

`from_run` is resolved to:

```text
<project>/<from_run>/weights/best.pt
```

On Colab, `<project>` comes from `configs/common/paths_colab.yaml`.

## Environment

Local:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Google Colab:

```bash
pip install -r requirements-colab.txt
```

Do not reinstall PyTorch on Colab unless necessary; Colab already provides a
CUDA-compatible PyTorch build.

## Dataset Preparation

BDD100K YOLO remapping:

```bash
python scripts/remap_bdd100k_yolo.py \
  --raw-dir datasets/bdd100k_yolo_raw \
  --output-dir datasets/bdd100k_6cls_yolo \
  --max-train-images 5000 \
  --max-val-images 1000 \
  --seed 42 \
  --clean
```

BDD100K class mapping:

```text
person -> person
rider -> person
car -> car
bus -> bus
truck -> truck
bike -> bicycle
motor -> motorcycle
traffic light/sign/train -> ignored
```

DAWN preparation:

```bash
python scripts/prepare_dawn.py \
  --raw-dir data/raw/DAWN \
  --output-dir datasets/dawn_6cls_yolo \
  --imgsz 640 \
  --seed 42 \
  --clean
```

Expected YOLO dataset YAMLs on Colab:

```text
/content/bdd100k_6cls_yolo/dataset.yaml
/content/dawn_6cls_yolo/dataset.yaml
```

## Training

Stage 1, COCO to BDD100K:

```bash
python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolov8n.yaml
```

Stage 2, BDD100K to DAWN:

```bash
python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml
```

Resume a run:

```bash
python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolov8n.yaml \
  --resume
```

Use the same pattern for:

```text
YOLOv8n
YOLOv8s
YOLO11n
YOLOv10n
RT-DETR
```

## CBAM Ablation

Train YOLOv8n + CBAM on DAWN using the same Stage 2 base config:

```bash
python scripts/train.py \
  --config configs/ablation/stage2_dawn_yolov8n_cbam.yaml
```

The CBAM trainer copies compatible pretrained layers from the Stage 1 BDD
checkpoint and leaves inserted CBAM layers randomly initialized.

If your Stage 1 checkpoint has a different run folder, override it:

```bash
python scripts/train.py \
  --config configs/ablation/stage2_dawn_yolov8n_cbam.yaml \
  --weights /content/drive/MyDrive/adverse_weather_project/runs/bdd_yolov8n/weights/best.pt
```

Planned but not implemented yet:

```text
stage2_dawn_yolov8n_weather_aug.yaml
stage2_dawn_yolov8n_dehaze.yaml
```

## Evaluation and Result Collection

Evaluate one run:

```bash
python scripts/evaluate.py \
  --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml \
  --split val
```

Evaluate all Stage 2 Ultralytics runs:

```bash
python scripts/eval_all.py --split val
```

Collect metrics into one CSV:

```bash
python scripts/collect_results.py --split val
```

Compare against a baseline:

```bash
python scripts/compare_results.py \
  --input /content/drive/MyDrive/adverse_weather_project/runs/val_summary.csv \
  --baseline stage2_dawn_yolov8n_from_bdd
```

## Faster R-CNN

Faster R-CNN configs are prepared under:

```text
configs/torchvision/
```

The entrypoint exists, but the actual TorchVision training loop is not
implemented yet:

```bash
python scripts/train_torchvision.py \
  --config configs/torchvision/stage1_bdd_faster_rcnn.yaml
```

## References

- BDD100K: https://arxiv.org/abs/1805.04687
- DAWN: https://arxiv.org/abs/2008.05402
- CBAM: https://arxiv.org/abs/1807.06521
- Ultralytics Model YAML Guide: https://docs.ultralytics.com/guides/model-yaml-config/
