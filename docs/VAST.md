# Vast.ai Training Workflow

Pipeline 2-phase trên Vast.ai GPU instance.

Điểm khác so với Colab:
- Không timeout — training chạy liên tục trong tmux
- SSH access — không cần browser session
- `/workspace` là persistent storage trong instance
- Dùng `rclone` thay Drive mount

---

## 0. Chọn instance trên Vast

Recommended: **RTX 3090 / 4090 / A100**, image `pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime`

Bật port **8888** (Jupyter) trong Vast dashboard trước khi thuê.

---

## 1. One-time setup (lần đầu tiên)

SSH vào instance, chạy:

```bash
# System packages
apt-get update -qq && apt-get install -y fuse3 tmux unzip git rclone

# Clone repo
git clone https://github.com/thanhhbao/Object-Detection-In-Adverse.Weather-Conditions.git \
    /workspace/Object-Detection
cd /workspace/Object-Detection

# Python deps — KHÔNG cài lại torch nếu image đã có CUDA torch
pip install ultralytics==8.3.* pycocotools pyyaml opencv-python-headless
# Nếu cần torchvision (Faster R-CNN):
# pip install torchvision

# Thư mục dataset và runs
mkdir -p /workspace/datasets /workspace/runs

# Báo cho scripts dùng paths_vast.yaml thay vì paths_colab.yaml
echo 'export OD_PATHS=configs/common/paths_vast.yaml' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. Cấu hình rclone (Google Drive)

```bash
rclone config
# → chọn "n" (new remote)
# → name: gdrive
# → type: drive (số 13 hoặc gõ "drive")
# → client_id/secret: để trống (Enter)
# → scope: 1 (full access)
# → Cuối cùng: "Use auto config? n"
# → Copy link, mở trên máy local, dán auth token vào terminal
```

Test:
```bash
rclone ls gdrive:adverse_weather_project/datasets
```

---

## 3. Copy dataset từ Drive về /workspace

**Không train trực tiếp từ Drive mount** — copy về local trước:

```bash
# Mount Drive (chỉ để copy, không để train)
mkdir -p /tmp/gdrive
rclone mount gdrive: /tmp/gdrive --daemon --vfs-cache-mode writes

# Copy 3 dataset zip về local
cp /tmp/gdrive/adverse_weather_project/datasets/BDD100K.zip /workspace/datasets/
cp /tmp/gdrive/adverse_weather_project/datasets/XWOD.zip    /workspace/datasets/
cp /tmp/gdrive/adverse_weather_project/datasets/dawn_6cls_yolo.zip /workspace/datasets/

# Unmount sau khi copy xong
fusermount -u /tmp/gdrive
```

Hoặc dùng `rclone copy` (không cần mount):
```bash
rclone copy "gdrive:adverse_weather_project/datasets/BDD100K.zip" \
    /workspace/datasets/ --progress
rclone copy "gdrive:adverse_weather_project/datasets/XWOD.zip" \
    /workspace/datasets/ --progress
rclone copy "gdrive:adverse_weather_project/datasets/dawn_6cls_yolo.zip" \
    /workspace/datasets/ --progress
```

---

## 4. Chuẩn hóa dataset

```bash
cd /workspace/Object-Detection

# BDD100K → 30K subset (condition-aware)
python scripts/prepare_bdd100k_json.py \
  --zip-path /workspace/datasets/BDD100K.zip \
  --output-dir /workspace/datasets/bdd100k_6cls_30k_yolo \
  --subset-size 30000 --subset-mode condition_aware \
  --seed 42 --clean

# XWOD
unzip -q /workspace/datasets/XWOD.zip -d /workspace/datasets/XWOD_raw
python scripts/prepare_xwod.py \
  --src /workspace/datasets/XWOD_raw/dataset \
  --dst /workspace/datasets/xwod_6cls_yolo \
  --weather all

# DAWN — held-out test only (cần split 70/15/15)
unzip -q /workspace/datasets/dawn_6cls_yolo.zip -d /workspace/datasets/
# Nếu zip chưa có test split, prepare lại:
# python scripts/prepare_dawn.py \
#   --raw-dir /workspace/datasets/DAWN_raw \
#   --output-dir /workspace/datasets/dawn_6cls_yolo \
#   --imgsz 640 --seed 42
```

---

## 5. Phase 1 — Train trong tmux (không timeout)

```bash
# Tạo tmux session
tmux new -s train

# Trong session, đảm bảo OD_PATHS đã set
export OD_PATHS=configs/common/paths_vast.yaml
cd /workspace/Object-Detection

# Stage 1 — BDD (chạy tuần tự, mỗi model ~2-4h tùy GPU)
python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd30k_yolov8n.yaml
python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd30k_yolo11n.yaml
python scripts/train_ultralytics.py --config configs/ultralytics/stage1_bdd30k_rtdetr.yaml
python scripts/train_torchvision.py --config configs/torchvision/stage1_bdd30k_faster_rcnn.yaml

# Stage 2 — XWOD
python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd30k.yaml
python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_yolo11n_from_bdd30k.yaml
python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_rtdetr_from_bdd30k.yaml
python scripts/train_torchvision.py --config configs/torchvision/stage2_xwod_faster_rcnn_from_bdd30k.yaml
```

Detach khỏi tmux: `Ctrl+B` rồi `D`
Reattach: `tmux attach -t train`
Resume nếu ngắt giữa chừng:
```bash
python scripts/train_ultralytics.py --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd30k.yaml --resume
```

---

## 6. Backup runs về Drive (chạy định kỳ)

```bash
# Sync toàn bộ /workspace/runs lên Drive
rclone copy /workspace/runs \
    "gdrive:adverse_weather_project/runs" \
    --progress --transfers 4

# Hoặc chạy tự động mỗi 15 phút trong tmux session riêng
tmux new -s backup
watch -n 900 rclone copy /workspace/runs \
    "gdrive:adverse_weather_project/runs" --progress
```

---

## 7. Evaluate

```bash
export OD_PATHS=configs/common/paths_vast.yaml

# XWOD_test
python scripts/evaluate.py \
  --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd30k.yaml --split test
python scripts/evaluate_by_weather.py \
  --config configs/ultralytics/stage2_xwod_yolov8n_from_bdd30k.yaml --split test

# DAWN_test (zero-shot)
python - <<'EOF'
from ultralytics import YOLO
model = YOLO("/workspace/runs/stage2_xwod_yolov8n_from_bdd30k/weights/best.pt")
model.val(data="/workspace/datasets/dawn_6cls_yolo/dataset.yaml", split="test",
          imgsz=640, batch=16, project="/workspace/runs/zeroshot_dawn", name="yolov8n")
EOF

# Thu thập kết quả
python scripts/collect_results.py --split test
python scripts/compare_results.py \
  --input /workspace/runs/val_summary.csv \
  --baseline stage2_xwod_yolov8n_from_bdd30k
```

---

## 8. Vẽ biểu đồ trực tiếp trên Vast (Jupyter)

```bash
# Trong tmux session mới
tmux new -s jupyter
cd /workspace/Object-Detection
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Truy cập: `http://<vast-instance-ip>:8888` (token in terminal output)

Hoặc chạy script plot thẳng, lưu PNG:
```bash
python scripts/plot_torchvision_history.py \
  --run /workspace/runs/stage1_bdd30k_faster_rcnn \
  --output /workspace/runs/plots/
```

Download về máy:
```bash
# Từ máy local
scp -P <port> root@<vast-ip>:/workspace/runs/plots/*.png ./
```

Hoặc sync toàn bộ về Drive rồi mở trên Colab:
```bash
rclone copy /workspace/runs/plots \
    "gdrive:adverse_weather_project/plots" --progress
```

---

## 9. Khi xong — lưu kết quả cuối

```bash
# Sync toàn bộ: runs, datasets đã processed
rclone copy /workspace/runs \
    "gdrive:adverse_weather_project/runs" --progress --transfers 4

rclone copy /workspace/datasets/bdd100k_6cls_30k_yolo \
    "gdrive:adverse_weather_project/datasets/bdd100k_6cls_30k_yolo" --progress

# Chỉ cần giữ best.pt + metrics JSON — không cần weights/last.pt
# Nếu muốn tiết kiệm Drive space:
find /workspace/runs -name "last.pt" -delete
rclone copy /workspace/runs \
    "gdrive:adverse_weather_project/runs" --progress
```

---

## Ghi chú

- `OD_PATHS=configs/common/paths_vast.yaml` phải được export trước khi chạy bất kỳ script nào.
  Đã thêm vào `~/.bashrc` ở bước setup, nhưng tmux cần `source ~/.bashrc` hoặc đặt lại trong session.
- Vast instance bị xóa sau khi terminate — backup về Drive trước khi dừng instance.
- `/workspace` persist khi stop instance (không terminate), mất khi terminate.
- Dùng Stop (không Terminate) nếu muốn giữ data để train tiếp sau.
