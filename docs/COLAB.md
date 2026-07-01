# Google Colab Workflow

Pipeline 2-phase: fair model selection → final model training.

```
PHASE 1 — Fair Model Selection (cả 4 model)
  COCO pretrained → BDD100K → XWOD_train
  → Evaluate: XWOD_test + DAWN_test + ACDC_test + BDD_val + video
  → Rank → chọn winner

PHASE 2 — Final Model Training (winner only)
  Winner → XWOD_train + ACDC_train + optional DAWN_train + BDD replay 20–30%
  → Final eval: XWOD_test + DAWN_test + ACDC_test + video
```

6 class: person(0), bicycle(1), car(2), motorcycle(3), bus(4), truck(5)

Drive root:
```
DRIVE=/content/drive/MyDrive/adverse_weather_project
```

---

## 0. Setup (mỗi session)

```python
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi

%cd /content
!git clone https://github.com/thanhhbao/Object-Detection-In-Adverse.Weather-Conditions.git Object-Detection
%cd /content/Object-Detection
!pip install -q -r requirements-colab.txt

DRIVE="/content/drive/MyDrive/adverse_weather_project"
```

---

## 1. Chuẩn hóa dataset (chỉ làm 1 lần, zip lưu Drive)

### BDD100K — 30K ảnh cân bằng 5 điều kiện

```python
# Cần: BDD100K raw (images + det_train.json)
# Tải từ Kaggle:
#   !kaggle datasets download -d <bdd100k-detection-dataset> -p /content/bdd100k_raw --unzip
# hoặc nuImages (nuscenes.org) → dùng prepare_bdd100k.py tương tự

!python scripts/prepare_bdd100k.py \
  --images-dir /content/bdd100k_raw/bdd100k/images/100k/train \
  --labels-json /content/bdd100k_raw/bdd100k/labels/det_20/det_train.json \
  --output-dir /content/bdd100k_6cls_yolo \
  --per-condition 6000 --seed 42
# → ~30K ảnh: clear/overcast, rainy, foggy, night, dawn/dusk (~6K mỗi nhóm)

!zip -q -r "$DRIVE/datasets/bdd100k_6cls_yolo_30k.zip" /content/bdd100k_6cls_yolo

# Các session sau:
!unzip -q "$DRIVE/datasets/bdd100k_6cls_yolo_30k.zip" -d /content
```

### XWOD — tất cả điều kiện

```python
!unzip -q "$DRIVE/datasets/XWOD.zip" -d /content

!python scripts/prepare_xwod.py \
  --src /content/XWOD/dataset \
  --dst /content/xwod_6cls_yolo \
  --weather all
!cat /content/xwod_6cls_yolo/dataset.yaml

!zip -q -r "$DRIVE/datasets/xwod_6cls_yolo.zip" /content/xwod_6cls_yolo
# Các session sau: !unzip -q "$DRIVE/datasets/xwod_6cls_yolo.zip" -d /content
```

### ACDC — fog, rain, snow, night (panoptic → YOLO bbox)

```python
# Tải từ acdc.vision.ee.ethz.ch (đăng ký miễn phí)
# Upload zip lên Drive rồi:
!unzip -q "$DRIVE/datasets/ACDC.zip" -d /content/acdc_raw

!python scripts/prepare_acdc.py \
  --raw-dir /content/acdc_raw \
  --output-dir /content/acdc_6cls_yolo \
  --conditions all --seed 42
!cat /content/acdc_6cls_yolo/dataset.yaml

!zip -q -r "$DRIVE/datasets/acdc_6cls_yolo.zip" /content/acdc_6cls_yolo
# Các session sau: !unzip -q "$DRIVE/datasets/acdc_6cls_yolo.zip" -d /content
```

### DAWN — chỉ cần test split (held-out, không train)

```python
# DAWN chỉ dùng để evaluate — cần split 70/15/15 để có test set
!unzip -q "$DRIVE/datasets/DAWN_raw.zip" -d /content/DAWN_raw

!python scripts/prepare_dawn.py \
  --raw-dir /content/DAWN_raw \
  --output-dir /content/dawn_6cls_yolo \
  --imgsz 640 --seed 42
# → train(70%) / val(15%) / test(15%) theo weather stratified

!zip -q -r "$DRIVE/datasets/dawn_6cls_yolo.zip" /content/dawn_6cls_yolo
# Các session sau: !unzip -q "$DRIVE/datasets/dawn_6cls_yolo.zip" -d /content
```

---

## 2. Phase 1 — Train Stage 1: BDD (cả 4 model)

```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_yolov8n.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_yolo11n.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_rtdetr.yaml
!python scripts/train_torchvision.py --config configs/torchvision/stage1_bdd_faster_rcnn.yaml
```

Resume nếu ngắt:
```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd_yolov8n.yaml --resume
```

---

## 3. Phase 1 — Train Stage 2: XWOD (cả 4 model)

```python
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_yolo11n_from_bdd.yaml
!python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_rtdetr_from_bdd.yaml
!python scripts/train_torchvision.py --config configs/torchvision/stage2_xwod_faster_rcnn_from_bdd.yaml
```

---

## 4. Phase 1 — Evaluate để chọn model

### XWOD_test

```python
for cfg in [
    "configs/ultralytics/stage2_xwod_yolov8n_from_bdd.yaml",
    "configs/ultralytics/stage2_xwod_yolo11n_from_bdd.yaml",
    "configs/ultralytics/stage2_xwod_rtdetr_from_bdd.yaml",
]:
    !python scripts/evaluate.py --config {cfg} --split test
    !python scripts/evaluate_by_weather.py --config {cfg} --split test

!python scripts/evaluate_torchvision.py \
  --config configs/torchvision/stage2_xwod_faster_rcnn_from_bdd.yaml --split test
```

### DAWN_test (held-out — zero-shot từ XWOD model)

```python
RUNS = f"{DRIVE}/runs"

for run_name in [
    "stage2_xwod_yolov8n_from_bdd",
    "stage2_xwod_yolo11n_from_bdd",
    "stage2_xwod_rtdetr_from_bdd",
]:
    from ultralytics import YOLO
    model = YOLO(f"{RUNS}/{run_name}/weights/best.pt")
    model.val(data="/content/dawn_6cls_yolo/dataset.yaml", split="test",
              imgsz=640, batch=16, project=f"{RUNS}/{run_name}_on_dawn", name="zeroshot")
```

### ACDC_test (zero-shot)

```python
for run_name in [
    "stage2_xwod_yolov8n_from_bdd",
    "stage2_xwod_yolo11n_from_bdd",
    "stage2_xwod_rtdetr_from_bdd",
]:
    model = YOLO(f"{RUNS}/{run_name}/weights/best.pt")
    model.val(data="/content/acdc_6cls_yolo/dataset.yaml", split="test",
              imgsz=640, batch=16, project=f"{RUNS}/{run_name}_on_acdc", name="zeroshot")
```

### BDD_val

```python
for cfg in [
    "configs/ultralytics/stage1_bdd_yolov8n.yaml",
    "configs/ultralytics/stage1_bdd_yolo11n.yaml",
    "configs/ultralytics/stage1_bdd_rtdetr.yaml",
]:
    !python scripts/evaluate.py --config {cfg} --split val
```

### Thu thập và so sánh

```python
!python scripts/collect_results.py --split test
!python scripts/compare_results.py \
  --input "$DRIVE/runs/val_summary.csv" \
  --baseline stage2_xwod_yolov8n_from_bdd
```

### Tiêu chí chọn model (theo thứ tự ưu tiên)

1. **mAP@50-95 trên XWOD_test + DAWN_test + ACDC_test** (tổng hợp adverse weather)
2. **Recall trong điều kiện xấu** (fog, rain, snow, night)
3. FPS (batch=1, T4)
4. Số tham số
5. Độ ổn định trên video thực tế

---

## 5. Phase 2 — Final model training (winner only)

Ví dụ với YOLOv8n (thay nếu model khác thắng):

```python
import yaml, os

# Tạo combined dataset: XWOD + ACDC + BDD replay 20–30%
merged = {
    "path": "/content/mixed_final",
    "train": [
        "/content/xwod_6cls_yolo/images/train",
        "/content/acdc_6cls_yolo/images/train",
        # optional DAWN train:
        # "/content/dawn_6cls_yolo/images/train",
    ],
    "val": [
        "/content/xwod_6cls_yolo/images/val",
        "/content/acdc_6cls_yolo/images/val",
    ],
    "test": "/content/xwod_6cls_yolo/images/test",
    "names": {0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"bus",5:"truck"},
}
os.makedirs("/content/mixed_final", exist_ok=True)
with open("/content/mixed_final/dataset.yaml", "w") as f:
    yaml.dump(merged, f, sort_keys=False)
```

```python
# BDD replay: lấy ~20% BDD training set
import random, shutil
from pathlib import Path

bdd_imgs = sorted(Path("/content/bdd100k_6cls_yolo/images/train").glob("*.jpg"))
random.seed(42)
replay = random.sample(bdd_imgs, int(len(bdd_imgs) * 0.25))

replay_img_dir = Path("/content/mixed_final/images/replay")
replay_lbl_dir = Path("/content/mixed_final/labels/replay")
replay_img_dir.mkdir(parents=True, exist_ok=True)
replay_lbl_dir.mkdir(parents=True, exist_ok=True)

for img in replay:
    shutil.copy(img, replay_img_dir / img.name)
    lbl = Path("/content/bdd100k_6cls_yolo/labels/train") / img.with_suffix(".txt").name
    if lbl.exists():
        shutil.copy(lbl, replay_lbl_dir / lbl.name)

# Thêm replay vào merged dataset
merged["train"].append(str(replay_img_dir))
with open("/content/mixed_final/dataset.yaml", "w") as f:
    yaml.dump(merged, f, sort_keys=False)
```

```python
# Train Phase 2
!python scripts/train_ultralytics.py \
  --config configs/ultralytics/phase2_final_yolov8n_from_xwod.yaml \
  --data /content/mixed_final/dataset.yaml
```

---

## 6. Phase 2 — Final evaluation

```python
# XWOD_test
!python scripts/evaluate.py \
  --config configs/ultralytics/phase2_final_yolov8n_from_xwod.yaml --split test
!python scripts/evaluate_by_weather.py \
  --config configs/ultralytics/phase2_final_yolov8n_from_xwod.yaml --split test

# DAWN_test (zero-shot)
model = YOLO(f"{RUNS}/phase2_final_yolov8n/weights/best.pt")
model.val(data="/content/dawn_6cls_yolo/dataset.yaml", split="test",
          imgsz=640, batch=16, project=f"{RUNS}/phase2_on_dawn", name="final")

# ACDC_test (zero-shot)
model.val(data="/content/acdc_6cls_yolo/dataset.yaml", split="test",
          imgsz=640, batch=16, project=f"{RUNS}/phase2_on_acdc", name="final")
```

---

## 7. Ablation study (winner = YOLOv8n, từ Phase 2 checkpoint)

| ID | Cấu hình | Config |
|---|---|---|
| A0 | Phase 1 winner (BDD→XWOD) | `stage2_xwod_yolov8n_from_bdd` |
| A1 | Phase 2 full (XWOD+ACDC+replay) | `phase2_final_yolov8n_from_xwod` |
| A2 | A1 + weather augmentation | `ablation/phase2_yolov8n_weather_aug` |
| A3 | A1 + CBAM module | `ablation/phase2_yolov8n_cbam` |

```python
# A2 — Weather augmentation
!python scripts/train_ultralytics.py \
  --config configs/ablation/phase2_yolov8n_weather_aug.yaml

# A3 — CBAM neck
!python scripts/train_ultralytics.py \
  --config configs/ablation/phase2_yolov8n_cbam.yaml
```

Đánh giá tất cả ablation trên cùng test set:
```python
for cfg in [
    "configs/ultralytics/stage2_xwod_yolov8n_from_bdd.yaml",      # A0
    "configs/ultralytics/phase2_final_yolov8n_from_xwod.yaml",    # A1
    "configs/ablation/phase2_yolov8n_weather_aug.yaml",            # A2
    "configs/ablation/phase2_yolov8n_cbam.yaml",                   # A3
]:
    !python scripts/evaluate.py --config {cfg} --split test
    !python scripts/evaluate_by_weather.py --config {cfg} --split test
```

---

## Metrics cần report

```
mAP@50, mAP@50-95
Precision, Recall
FPS (batch=1, T4), ms/img
Params
Per-class AP (6 class)
Per-weather mAP (fog, rain, snow, night, sand)
```

Seed stability (vì DAWN nhỏ):
```python
!python scripts/run_seeds.py \
  --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd.yaml \
  --seeds 0 1 2 --split test
```

---

## Ghi chú

- **DAWN_test**: held-out chính — không train bất kỳ bước nào, chỉ evaluate
- **XWOD_test / ACDC_test**: held-out thứ hai — không dùng trong Phase 2 train
- Phase 2 chỉ chạy model thắng; 3 model còn lại dừng ở Stage 2
- Ablation (A2, A3) đều khởi từ Phase 2 checkpoint
- `--resume` an toàn cho mọi bước; checkpoint tự lưu vào Drive mỗi epoch
- Xóa run cũ trước khi train lại: `!rm -rf "$DRIVE/runs/stage1_bdd_* $DRIVE/runs/stage2_dawn_*"`
