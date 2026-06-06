# Chạy nghiên cứu trên Google Colab

Chọn `Runtime > Change runtime type > T4 GPU` hoặc GPU tốt hơn trước khi chạy.
Không train trực tiếp trên ảnh nằm trong Google Drive vì tốc độ đọc file nhỏ
chậm. Hãy copy/giải nén dữ liệu vào `/content`, nhưng lưu kết quả vào Drive.

## Chuẩn bị trên Google Drive

Đặt repository và file dữ liệu nén theo cấu trúc:

```text
MyDrive/
├── Object-Detection/              # repository này
├── DAWN.zip                       # chứa các ảnh và Pascal VOC XML
└── YOLOv8_DAWN_Thesis/            # kết quả được tạo tự động
```

Tên thư mục repository có thể khác. Điều chỉnh lệnh `cd` tương ứng.

## Cell 1: Mount Drive và kiểm tra GPU

```python
from google.colab import drive
drive.mount("/content/drive")

!nvidia-smi
```

## Cell 2: Cài dependency

```python
%cd /content/drive/MyDrive/Object-Detection
!pip install -q -r requirements-colab.txt

import torch, ultralytics
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("ultralytics:", ultralytics.__version__)
```

Không cài lại `torch` bằng `requirements.txt`; Colab đã cung cấp bản PyTorch phù
hợp với CUDA của runtime.

## Cell 3: Copy và chuẩn bị DAWN ở local runtime

```python
%cd /content
!unzip -q -o "/content/drive/MyDrive/DAWN.zip" -d /content/DAWN_RAW

%cd /content/drive/MyDrive/Object-Detection
!python scripts/prepare_dawn.py \
  --raw-dir /content/DAWN_RAW \
  --output-dir /content/dawn_yolo \
  --imgsz 640 \
  --seed 42 \
  --clean
```

Sau bước này, kiểm tra số lượng split được in ra và file
`/content/dawn_yolo/manifest.csv`. Khi runtime bị ngắt, `/content/dawn_yolo` sẽ
mất; chạy lại Cell 3 với cùng seed sẽ tạo đúng split cũ.

## Cell 4: Huấn luyện baseline

```python
%cd /content/drive/MyDrive/Object-Detection
!python scripts/train.py --variant baseline --config configs/experiment_colab.yaml
```

Kết quả được lưu tại:

```text
/content/drive/MyDrive/YOLOv8_DAWN_Thesis/ablation/baseline/
```

Nếu runtime bị ngắt, chạy lại Cell 1-3 rồi tiếp tục:

```python
%cd /content/drive/MyDrive/Object-Detection
!python scripts/train.py \
  --variant baseline \
  --config configs/experiment_colab.yaml \
  --resume
```

## Cell 5: Huấn luyện CBAM

```python
%cd /content/drive/MyDrive/Object-Detection
!python scripts/train.py --variant cbam --config configs/experiment_colab.yaml
```

Resume khi bị ngắt:

```python
!python scripts/train.py \
  --variant cbam \
  --config configs/experiment_colab.yaml \
  --resume
```

Không chạy baseline và CBAM đồng thời trên cùng một GPU.

## Cell 6: Đánh giá và tạo bảng ablation

Phải chạy lại Cell 3 trước nếu runtime mới chưa có `/content/dawn_yolo`.

```python
%cd /content/drive/MyDrive/Object-Detection
!python scripts/evaluate.py --variant baseline --config configs/experiment_colab.yaml
!python scripts/evaluate.py --variant cbam --config configs/experiment_colab.yaml
!python scripts/compare_results.py --config configs/experiment_colab.yaml
```

Bảng cuối:

```text
/content/drive/MyDrive/YOLOv8_DAWN_Thesis/ablation/ablation_comparison.csv
```

## Lưu ý thực nghiệm

- Giữ nguyên loại GPU giữa hai mô hình khi so sánh FPS.
- `T4`, `L4` và `A100` có tốc độ khác nhau; ghi rõ GPU trong khóa luận.
- Nếu T4 hết VRAM, giảm `batch` của **cả hai mô hình** xuống cùng giá trị, ví dụ
  `8`. Không dùng batch khác nhau giữa baseline và CBAM.
- Chỉ đánh giá test set sau khi đã chốt kiến trúc bằng validation set.
- Để báo cáo đáng tin cậy, lặp lại với ít nhất ba training seed.

