# Research and Enhancement of Deep Learning Models for People and Vehicle Detection in Adverse Weather Conditions for Intelligent Traffic Surveillance

This repository contains a research pipeline for benchmarking and improving deep
learning object detectors for people and vehicle detection under adverse weather
conditions. The target application is intelligent traffic surveillance.

The broader study compares multiple detector families, such as YOLOv8, YOLO11,
YOLOv10, and RT-DETR. The current implementation focuses on a progressive
finetuning pipeline and a YOLOv8-CBAM ablation branch, where the improved model
is evaluated against its baseline under the same data split, pretrained weights,
and hyperparameters.

Target classes:

```text
person, bicycle, car, motorcycle, bus, truck
```

End-to-end Google Colab instructions are available in [docs/COLAB.md](docs/COLAB.md).

## Project Structure

```text
configs/
├── stages/
│   ├── stage1_bdd_yolov8n_colab.yaml
│   ├── stage2_dawn_yolov8n_from_bdd_colab.yaml
│   └── stage3_acdc_dawn_yolov8n_from_bdd_colab.yaml
├── ablation/
│   ├── dawn_cbam_local.yaml
│   └── dawn_cbam_colab.yaml
└── benchmark/
    ├── dawn_yolov8n_colab.yaml
    ├── dawn_yolov8s_colab.yaml
    ├── dawn_yolo11n_colab.yaml
    ├── dawn_yolov10n_colab.yaml
    └── dawn_rtdetr_l_colab.yaml

models/
├── yolov8n_baseline.yaml
└── yolov8n_cbam_neck.yaml

scripts/
├── remap_bdd100k_yolo.py
├── prepare_bdd100k.py
├── prepare_shift.py
├── prepare_acdc_dawn.py
├── prepare_dawn.py
├── train.py
├── evaluate.py
└── compare_results.py

src/dawn_ablation/
├── attention.py
├── common.py
└── data_prep.py
```

## Environment Setup

Python 3.10 or 3.11 is recommended. Python 3.13 may not be compatible with the
full deep learning stack.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Google Colab, use:

```bash
pip install -r requirements-colab.txt
```

Do not reinstall PyTorch on Colab unless necessary; Colab already provides a
CUDA-compatible PyTorch build.

## Multi-Stage Pipeline

### Stage 1: COCO to BDD100K Clear Daytime

Input weights:

```text
yolov8n.pt
```

Dataset:

```text
BDD100K clear daytime subset
```

Output weights:

```text
yolov8n_bdd.pt
```

If you use the Kaggle dataset `a7madmostafa/bdd100k-yolo` with this structure:

```text
bdd100k_yolo_raw/
├── train/images
├── train/labels
├── val/images
├── val/labels
├── test/images
├── test/labels
└── data.yaml
```

remap the original 10-class YOLO labels to the 6 target classes:

```bash
python scripts/remap_bdd100k_yolo.py \
  --raw-dir data/raw/bdd100k_yolo_raw \
  --output-dir data/processed/bdd100k_6cls_yolo \
  --max-train-images 5000 \
  --max-val-images 1000 \
  --seed 42 \
  --clean
```

By default, the script uses symlinks to save disk space. If your environment does
not support symlinks, add:

```bash
--copy-images
```

Class remapping:

```text
person -> person
rider -> person
car -> car
bus -> bus
truck -> truck
bike -> bicycle
motor -> motorcycle
traffic light -> ignored
traffic sign -> ignored
train -> ignored
```

Train Stage 1 with a config file:

```bash
python scripts/train_ultralytics.py --config configs/stages/stage1_bdd_yolov8n_colab.yaml
```

Resume if Colab disconnects:

```bash
python scripts/train_ultralytics.py --config configs/stages/stage1_bdd_yolov8n_colab.yaml --resume
```

If you use the original BDD100K detection JSON instead, run:

```bash
python scripts/prepare_bdd100k.py \
  --images-dir data/raw/BDD100K/images/100k/train \
  --labels-json data/raw/BDD100K/labels/det_train.json \
  --output-dir data/processed/bdd_clear_yolo \
  --max-images 5000 \
  --imgsz 640 \
  --seed 42 \
  --clean
```

### Stage 2: BDD100K to SHIFT Synthetic Weather

Input weights:

```text
yolov8n_bdd.pt
```

Dataset:

```text
SHIFT synthetic fog/rain subset
```

Output weights:

```text
yolov8n_shift.pt
```

Prepare the SHIFT subset:

```bash
python scripts/prepare_shift.py \
  --images-dir data/raw/SHIFT/images \
  --annotations-json data/raw/SHIFT/det_2d.json \
  --output-dir data/processed/shift_yolo \
  --weather fog,rain \
  --max-images 5000 \
  --imgsz 640 \
  --seed 42 \
  --clean
```

The script supports BDD/Scalabel-like JSON and COCO-style JSON.

### Stage 3 and 4: Real Adverse Weather Ablation

Input weights:

```text
yolov8n_shift.pt
```

Dataset:

```text
ACDC + DAWN, stratified by fog/rain/snow/sand
```

Prepare the merged real-weather dataset:

```bash
python scripts/prepare_acdc_dawn.py \
  --acdc-images-dir data/raw/ACDC/images \
  --acdc-annotations-json data/raw/ACDC/instances.json \
  --dawn-raw-dir data/raw/DAWN \
  --output-dir data/processed/acdc_dawn_yolo \
  --weather fog,rain,snow,sand \
  --imgsz 640 \
  --seed 42 \
  --clean
```

The merged dataset is split 70/15/15 into train/val/test using stratified
weather groups.

## DAWN-Only Preparation

For a DAWN-only experiment:

```bash
python scripts/prepare_dawn.py \
  --raw-dir data/raw/DAWN \
  --output-dir data/processed/dawn_yolo \
  --imgsz 640 \
  --seed 42 \
  --clean
```

The script converts Pascal VOC XML annotations to YOLO format, applies letterbox
resizing, and creates `dataset.yaml`.

## Training

Train the original YOLOv8n baseline:

```bash
python scripts/train.py --variant baseline --config configs/ablation/dawn_cbam_local.yaml
```

Train YOLOv8n with CBAM in the Neck:

```bash
python scripts/train.py --variant cbam --config configs/ablation/dawn_cbam_local.yaml
```

Resume a Colab run from `last.pt`:

```bash
python scripts/train.py --variant baseline --config configs/ablation/dawn_cbam_local.yaml --resume
python scripts/train.py --variant cbam --config configs/ablation/dawn_cbam_local.yaml --resume
```

## Evaluation

Evaluate the best checkpoint on the test split:

```bash
python scripts/evaluate.py --variant baseline --config configs/ablation/dawn_cbam_local.yaml
python scripts/evaluate.py --variant cbam --config configs/ablation/dawn_cbam_local.yaml
```

Create the final comparison table:

```bash
python scripts/compare_results.py --config configs/ablation/dawn_cbam_local.yaml
```

The final CSV is saved to:

```text
runs/ablation/ablation_comparison.csv
```

## Ablation Design

The comparison between baseline and CBAM must keep these factors identical:

- dataset split
- pretrained checkpoint
- image size
- batch size
- optimizer
- learning rate
- augmentation settings
- number of epochs
- validation and test protocol

The only intended difference is the architecture:

```text
baseline: YOLOv8n
cbam:     YOLOv8n + CBAMResearch blocks in the Neck
```

Report both accuracy and cost:

- Precision
- Recall
- mAP@50
- mAP@50-95
- inference time
- FPS
- number of parameters

For reliable results, run at least three seeds and report mean ± standard
deviation.

## References

- BDD100K: https://arxiv.org/abs/1805.04687
- ACDC: https://arxiv.org/abs/2104.13395
- DAWN: https://arxiv.org/abs/2008.05402
- CBAM: https://arxiv.org/abs/1807.06521
- Ultralytics Model YAML Guide: https://docs.ultralytics.com/guides/model-yaml-config/
