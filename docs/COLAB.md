# Google Colab Workflow

This guide assumes the repository is cloned to:

```text
/content/Object-Detection
```

and datasets/checkpoints are stored in:

```text
/content/drive/MyDrive/adverse_weather_project
```

## 1. Mount Drive and Check GPU

```python
from google.colab import drive
drive.mount("/content/drive")

!nvidia-smi
```

If `nvidia-smi` fails, switch Colab runtime to GPU before training.

## 2. Enter Project and Install Dependencies

```python
%cd /content/Object-Detection
!pip install -q -r requirements-colab.txt
```

## 3. Prepare Dataset Folders

The configs expect these dataset YAML files:

```text
/content/bdd100k_6cls_yolo/dataset.yaml
/content/dawn_6cls_yolo/dataset.yaml
```

Unzip BDD100K:

```python
!rm -rf /content/bdd100k_6cls_yolo
!unzip -q "/content/drive/MyDrive/adverse_weather_project/datasets/bdd100k_6cls_yolo.zip" -d /content
!cat /content/bdd100k_6cls_yolo/dataset.yaml
```

Unzip DAWN:

```python
!rm -rf /content/dawn_6cls_yolo
!unzip -q "/content/drive/MyDrive/adverse_weather_project/datasets/dawn_6cls_yolo.zip" -d /content
!cat /content/dawn_6cls_yolo/dataset.yaml
```

## 4. Stage 1: Train on BDD100K

YOLOv8n:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolov8n.yaml
```

YOLOv8s:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolov8s.yaml
```

YOLO11n:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolo11n.yaml
```

Resume a disconnected run:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage1_bdd_yolov8n.yaml \
  --resume
```

## 5. Stage 2: Fine-tune on DAWN

After Stage 1 finishes, Stage 2 automatically loads:

```text
/content/drive/MyDrive/adverse_weather_project/runs/<stage1_run>/weights/best.pt
```

YOLOv8n from BDD:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml
```

If your existing Stage 1 run is named `bdd_yolov8n` instead of
`stage1_bdd_yolov8n`, override the checkpoint once:

```python
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml \
  --weights /content/drive/MyDrive/adverse_weather_project/runs/bdd_yolov8n/weights/best.pt \
  --name stage2_dawn_yolov8n_from_bdd
```

## 6. CBAM Ablation

Run this after the Stage 1 YOLOv8n BDD checkpoint exists:

```python
!python scripts/train.py \
  --config configs/ablation/stage2_dawn_yolov8n_cbam.yaml
```

This is an architecture ablation. Weather augmentation and dehazing configs are
registered but not implemented yet.

If your Stage 1 YOLOv8n checkpoint uses the older run name:

```python
!python scripts/train.py \
  --config configs/ablation/stage2_dawn_yolov8n_cbam.yaml \
  --weights /content/drive/MyDrive/adverse_weather_project/runs/bdd_yolov8n/weights/best.pt
```

## 7. Evaluate

Evaluate one run on validation split:

```python
!python scripts/evaluate.py \
  --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml \
  --split val
```

Evaluate CBAM:

```python
!python scripts/evaluate.py \
  --config configs/ablation/stage2_dawn_yolov8n_cbam.yaml \
  --split val
```

Evaluate all Stage 2 Ultralytics runs:

```python
!python scripts/eval_all.py --split val
```

Collect results:

```python
!python scripts/collect_results.py --split val
```

Compare against YOLOv8n Stage 2:

```python
!python scripts/compare_results.py \
  --input /content/drive/MyDrive/adverse_weather_project/runs/val_summary.csv \
  --baseline stage2_dawn_yolov8n_from_bdd
```

## 8. Faster R-CNN

Faster R-CNN config files are present, but the TorchVision trainer is still a
placeholder:

```python
!python scripts/train_torchvision.py \
  --config configs/torchvision/stage1_bdd_faster_rcnn.yaml
```
