# Google Colab Workflow

Pipeline 3 giai đoạn huấn luyện + ablation study.

```
COCO pretrained
    ↓ Stage 1 (cả 4 model)
BDD_balanced_30K  (traffic domain)
    ↓ Stage 2 (cả 4 model)
DAWN_train        (adverse weather)
    → đánh giá DAWN_test + XWOD_held-out → chọn winner
    ↓ Stage 3 (winner only)
XWOD_filtered     (+ optional DAWN mixed)
    → ablation study
```

Datasets và checkpoints lưu tại Drive:
```
/content/drive/MyDrive/adverse_weather_project/
```

---

## 0. Setup (mỗi session)

```python
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi

%cd /content/Object-Detection
!pip install -q -r requirements-colab.txt
```

---

## Bước 1 — Chuẩn hóa dữ liệu (6 class)

### BDD100K — 30K ảnh clear-daytime

```python
# Nếu chưa có bdd100k_6cls_yolo_30k.zip trên Drive:
!python scripts/prepare_bdd100k.py \
  --images-dir /content/bdd100k_raw/bdd100k/images/100k/train \
  --labels-json /content/bdd100k_raw/bdd100k/labels/det_20/det_train.json \
  --output-dir /content/bdd100k_6cls_yolo \
  --per-condition 6000 --seed 42
# → ~30K ảnh cân bằng: clear/overcast, rainy, foggy, night, dawn/dusk (~6K mỗi nhóm)
!zip -q -r "$DRIVE/datasets/bdd100k_6cls_yolo_30k.zip" /content/bdd100k_6cls_yolo

# Các session sau (zip đã có):
DRIVE="/content/drive/MyDrive/adverse_weather_project"
!unzip -q "$DRIVE/datasets/bdd100k_6cls_yolo_30k.zip" -d /content
```

### DAWN — split 70/15/15 train/val/test (theo weather)

```python
!python scripts/prepare_dawn.py \
  --raw-dir /content/DAWN_raw \
  --output-dir /content/dawn_6cls_yolo \
  --imgsz 640 --seed 42
# Lưu lên Drive một lần:
!zip -q -r "$DRIVE/datasets/dawn_6cls_yolo.zip" /content/dawn_6cls_yolo

# Các session sau:
!unzip -q "$DRIVE/datasets/dawn_6cls_yolo.zip" -d /content
!cat /content/dawn_6cls_yolo/dataset.yaml
```

### XWOD — giữ tất cả điều kiện (train/val/test có sẵn)

```python
!unzip -q "$DRIVE/datasets/XWOD.zip" -d /content
!python scripts/prepare_xwod.py \
  --src /content/XWOD/dataset \
  --dst /content/xwod_6cls_yolo \
  --weather all   # fog, rain, snow, sand, flooding, tornado, wildfire
!cat /content/xwod_6cls_yolo/dataset.yaml
```

XWOD_filtered (chỉ fog/rain/snow/sand — dùng cho Stage 3 fine-tune):
```python
!python scripts/prepare_xwod.py \
  --src /content/XWOD/dataset \
  --dst /content/xwod_filtered_6cls_yolo \
  --weather fog,rain,snow,sand
```

---

## Bước 2 — Train tất cả 4 model (Stage 1 BDD → Stage 2 DAWN)

Mỗi model chạy 2 bước tuần tự. Dùng `--resume` khi session bị ngắt.

### YOLOv8n

```python
# Stage 1: COCO → BDD
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_yolov8n.yaml
# Stage 2: BDD → DAWN_train
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml
```

### YOLO11n

```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_yolo11n.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_dawn_yolo11n_from_bdd.yaml
```

### RT-DETR-l

```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_rtdetr.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_dawn_rtdetr_from_bdd.yaml
```

### Faster R-CNN (ResNet50-FPN)

```python
!python scripts/train_torchvision.py --config configs/torchvision/stage1_bdd_faster_rcnn.yaml
!python scripts/train_torchvision.py --config configs/torchvision/stage2_dawn_faster_rcnn_from_bdd.yaml
```

Resume nếu ngắt giữa chừng:
```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml --resume
```

---

## Bước 3 — Đánh giá để chọn model

### Trên DAWN_test (held-out)

```python
for cfg in [
    "configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml",
    "configs/ultralytics/stage2_dawn_yolo11n_from_bdd.yaml",
    "configs/ultralytics/stage2_dawn_rtdetr_from_bdd.yaml",
]:
    !python scripts/evaluate.py --config {cfg} --split test
    !python scripts/evaluate_by_weather.py --config {cfg} --split test

!python scripts/evaluate_torchvision.py \
  --config configs/torchvision/stage2_dawn_faster_rcnn_from_bdd.yaml --split test
!python scripts/evaluate_torchvision_by_weather.py \
  --config configs/torchvision/stage2_dawn_faster_rcnn_from_bdd.yaml --split test
```

### Trên XWOD_held-out (XWOD test, zero-shot từ DAWN model)

```python
from ultralytics import YOLO
RUNS = "/content/drive/MyDrive/adverse_weather_project/runs"
xwod_data = "/content/xwod_6cls_yolo/dataset.yaml"

for run_name in ["stage2_dawn_yolov8n_from_bdd", "stage2_dawn_yolo11n_from_bdd",
                 "stage2_dawn_rtdetr_from_bdd"]:
    model = YOLO(f"{RUNS}/{run_name}/weights/best.pt")
    model.val(data=xwod_data, split="test", imgsz=640, batch=16,
              project=f"{RUNS}/{run_name}_on_xwod", name="zeroshot")
```

### Trên BDD_val

```python
for cfg in [
    "configs/ultralytics/stage1_bdd_yolov8n.yaml",
    "configs/ultralytics/stage1_bdd_yolo11n.yaml",
    "configs/ultralytics/stage1_bdd_rtdetr.yaml",
]:
    !python scripts/evaluate.py --config {cfg} --split val
```

Thu thập và so sánh:
```python
!python scripts/collect_results.py --split test
!python scripts/compare_results.py \
  --input "$DRIVE/runs/val_summary.csv" \
  --baseline stage2_dawn_yolov8n_from_bdd
```

---

## Bước 4 — Chọn model tốt nhất

Tiêu chí (theo mức độ ưu tiên):
1. **mAP@50-95 trên DAWN_test + XWOD_held-out**
2. **Recall trong điều kiện thời tiết xấu** (rain, sand, fog)
3. FPS (inference T4, batch=1)
4. Số tham số (params)
5. Độ ổn định trên video thực tế

Dự kiến: YOLOv8n dẫn đầu (đã xác nhận TN1). Nếu ranking đổi với 30K BDD, điều chỉnh stage3 configs tương ứng.

---

## Bước 5 — Fine-tune sâu model thắng trên XWOD (Stage 3)

Ví dụ với YOLOv8n (thay tên nếu model khác thắng):

```python
# Stage 3: DAWN_model → XWOD_filtered_train
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/stage3_xwod_yolov8n_from_dawn.yaml
```

**Nếu muốn mixed DAWN+XWOD** (tránh forgetting DAWN):
```python
import yaml, os

# Tạo dataset gộp tại runtime
merged = {
    "path": "/content/mixed_adverse",
    "train": ["/content/xwod_filtered_6cls_yolo/images/train",
              "/content/dawn_6cls_yolo/images/train"],
    "val":   ["/content/xwod_filtered_6cls_yolo/images/val",
              "/content/dawn_6cls_yolo/images/val"],
    "test":  "/content/xwod_6cls_yolo/images/test",
    "names": {0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"bus",5:"truck"},
}
os.makedirs("/content/mixed_adverse", exist_ok=True)
with open("/content/mixed_adverse/dataset.yaml","w") as f:
    yaml.dump(merged, f, sort_keys=False)

# Train với dataset gộp (override dataset trong config)
from ultralytics import YOLO
RUNS = "/content/drive/MyDrive/adverse_weather_project/runs"
ckpt = f"{RUNS}/stage2_dawn_yolov8n_from_bdd/weights/best.pt"
model = YOLO(ckpt)
model.train(data="/content/mixed_adverse/dataset.yaml", epochs=30,
            imgsz=640, lr0=0.0001, project=RUNS, name="stage3_mixed_yolov8n")
```

Đánh giá Stage 3 trên DAWN_test + XWOD_test:
```python
!python scripts/evaluate.py \
  --config configs/ultralytics/stage3_xwod_yolov8n_from_dawn.yaml --split test
!python scripts/evaluate_by_weather.py \
  --config configs/ultralytics/stage3_xwod_yolov8n_from_dawn.yaml --split test
```

---

## Bước 6 — Ablation study

| ID | Cấu hình | Config |
|---|---|---|
| A0 | BDD → DAWN (baseline) | `stage2_dawn_yolov8n_from_bdd` |
| A1 | A0 → XWOD fine-tune | `stage3_xwod_yolov8n_from_dawn` |
| A2 | A1 + weather augmentation | `ablation/stage3_xwod_yolov8n_weather_aug` |
| A3 | A1 + CBAM module | `ablation/stage3_xwod_yolov8n_cbam` |

```python
# A2 — Weather aug
!python scripts/train_ultralytics.py \
  --config configs/ablation/stage3_xwod_yolov8n_weather_aug.yaml

# A3 — CBAM neck
!python scripts/train_ultralytics.py \
  --config configs/ablation/stage3_xwod_yolov8n_cbam.yaml
```

Đánh giá tất cả trên cùng test set:
```python
for cfg in [
    "configs/ultralytics/stage2_dawn_yolov8n_from_bdd.yaml",       # A0
    "configs/ultralytics/stage3_xwod_yolov8n_from_dawn.yaml",      # A1
    "configs/ablation/stage3_xwod_yolov8n_weather_aug.yaml",       # A2
    "configs/ablation/stage3_xwod_yolov8n_cbam.yaml",              # A3
]:
    !python scripts/evaluate.py --config {cfg} --split test
    !python scripts/evaluate_by_weather.py --config {cfg} --split test

!python scripts/compare_results.py \
  --input "$DRIVE/runs/val_summary.csv" \
  --baseline stage2_dawn_yolov8n_from_bdd
```

---

## Ghi chú

- DAWN_test (15% = ~210 ảnh) là tập held-out chính — không dùng trong bất kỳ bước train nào.
- XWOD test là held-out thứ hai — không dùng trong train Stage 3.
- Stage 3 chỉ train model thắng; các model còn lại dừng ở Stage 2.
- Ablation (A2, A3) đều khởi đầu từ Stage 2 DAWN checkpoint, không phải Stage 3.
- `--resume` an toàn cho mọi bước; checkpoint lưu vào Drive sau mỗi epoch.
