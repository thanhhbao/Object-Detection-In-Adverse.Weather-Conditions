# Experiment Results Log

Nhật ký kết quả thực nghiệm để trích vào luận văn. Mỗi mục ghi rõ: split đánh
giá, số seed, phần cứng, và nguồn số liệu (file JSON trong thư mục run).

> ⚠️ Lưu ý khi viết luận văn:
> - Các số dưới đây là trên **validation split**, **1 seed (seed=42)**.
>   Số dùng để báo cáo cuối cùng nên lấy trên **test split** và (lý tưởng) trung
>   bình ± độ lệch chuẩn qua nhiều seed (`scripts/run_seeds.py`).
> - mAP50 / mAP50-95 theo chuẩn COCO. Precision/Recall ở operating point của
>   Ultralytics.

## Cấu hình chung

| Hạng mục | Giá trị |
|---|---|
| Pipeline | COCO → BDD100K (Stage 1) → DAWN (Stage 2) |
| Lớp đối tượng | person, bicycle, car, motorcycle, bus, truck (6 lớp) |
| Ảnh đầu vào | 640×640 |
| Tối ưu | AdamW, lr0 theo config, cos_lr, AMP, seed 42 |
| Phần cứng đánh giá | Google Colab, NVIDIA Tesla T4 (15 GB) |
| Ultralytics / PyTorch | 8.3.159 / 2.11.0+cu128 |

---

## B0 — YOLOv8n Stage 2 (BDD100K → DAWN)

- Run: `stage2_dawn_yolov8n_from_bdd`
- Tham số: 3,006,818 (≈3.0M), 8.1 GFLOPs, 72 layers (fused)
- Tốc độ (Ultralytics, T4): 0.8 ms preprocess + 4.4 ms inference + 3.2 ms
  postprocess / ảnh
- Nguồn: `runs/stage2_dawn_yolov8n_from_bdd/val_metrics.json`,
  `val_metrics_by_weather.json`

### Tổng thể (val, 206 ảnh, 1806 đối tượng)

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.662 | 0.579 | 0.639 | 0.426 |

### Theo từng lớp (val)

| Lớp | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 55 | 111 | 0.783 | 0.694 | 0.739 | 0.439 |
| bicycle | 6 | 6 | 0.709 | 0.418 | 0.688 | 0.572 |
| car | 204 | 1522 | 0.746 | 0.864 | 0.875 | 0.588 |
| motorcycle | 14 | 23 | 0.717 | 0.652 | 0.702 | 0.389 |
| bus | 26 | 43 | 0.527 | 0.302 | 0.311 | 0.212 |
| truck | 62 | 101 | 0.488 | 0.545 | 0.519 | 0.354 |

Nhận xét: `car` tốt nhất (nhiều dữ liệu); `bus`/`truck` yếu nhất; `bicycle` chỉ
6 mẫu nên số liệu không đáng tin — minh chứng cho mất cân bằng lớp trên DAWN.

### Theo điều kiện thời tiết (val)

| Thời tiết | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| fog | 60 | 424 | 0.732 | 0.717 | 0.742 | 0.490 |
| snow | 41 | 561 | 0.643 | 0.666 | 0.680 | 0.457 |
| sand | 65 | 508 | 0.669 | 0.518 | 0.589 | 0.397 |
| rain | 40 | 313 | 0.560 | 0.621 | 0.632 | 0.382 |

Nhận xét: `rain` và `sand` khó nhất (mAP50-95 thấp nhất), `fog` dễ nhất. Đây là
mốc để đánh giá CBAM / weather-augmentation có cải thiện điều kiện khó không.

---

## B2 — YOLO11n Stage 2 (BDD100K → DAWN)

- Run: `stage2_dawn_yolo11n_from_bdd`
- Tham số: 2,583,322 (≈2.6M), 6.3 GFLOPs, 100 layers (fused)
- Tốc độ (T4): batch-1 14.8 ms → **67.4 FPS**; Ultralytics inference 6.6 ms
  (151 FPS chế độ batch)
- Nguồn: `runs/stage2_dawn_yolo11n_from_bdd/val_metrics.json`,
  `val_metrics_by_weather.json`

### Tổng thể (val, 206 ảnh, 1806 đối tượng)

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.784 | 0.504 | 0.602 | 0.369 |

### Theo từng lớp (val)

| Lớp | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.830 | 0.586 | 0.707 | 0.406 |
| bicycle | 1.000 | 0.328 | 0.415 | 0.215 |
| car | 0.867 | 0.793 | 0.875 | 0.590 |
| motorcycle | 0.861 | 0.541 | 0.654 | 0.351 |
| bus | 0.579 | 0.321 | 0.465 | 0.350 |
| truck | 0.564 | 0.455 | 0.495 | 0.302 |

### Theo điều kiện thời tiết (val)

| Thời tiết | mAP50 | mAP50-95 |
|---|---:|---:|
| fog | 0.687 | 0.434 |
| rain | 0.637 | 0.435 |
| snow | 0.687 | 0.433 |
| sand | 0.573 | 0.365 |

### So sánh với B0 (YOLOv8n)

| | Params | mAP50 | mAP50-95 | Recall |
|---|---:|---:|---:|---:|
| YOLOv8n (B0) | 3.0M | **0.639** | **0.426** | **0.579** |
| YOLO11n (B2) | 2.6M | 0.602 | 0.369 | 0.504 |

**Phát hiện đáng chú ý:** YOLO11n tuy mới hơn và ít tham số hơn nhưng **kém hơn
YOLOv8n** trên DAWN (mAP50-95 −0.057, recall −0.075). Recall thấp = bỏ sót nhiều
đối tượng hơn, dù precision cao hơn (0.784 vs 0.662) — model "thận trọng" hơn,
phát hiện ít nhưng chắc. Đây là material tốt: *kiến trúc mới hơn không đảm bảo
tốt hơn trên tập dữ liệu nhỏ/đặc thù như thời tiết xấu*.

---

## B5 — Faster R-CNN (ResNet50-FPN) Stage 2 (BDD100K → DAWN)

- Run: `stage2_dawn_faster_rcnn_from_bdd` (TorchVision)
- Tham số: 41,324,786 (≈41,3M), mô hình hai giai đoạn
- Tốc độ (T4): batch-1 36,7 ms → **27,2 FPS**
- Dừng sớm ở epoch 17 (patience 10)
- Nguồn: `runs/stage2_dawn_faster_rcnn_from_bdd/val_metrics.json`,
  `val_metrics_by_weather.json`, `val_confusion_matrix.png`, `val_pr_curve.png`

### Tổng thể (val, 206 ảnh, 1806 đối tượng)

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.576 | 0.808 | 0.514 | 0.310 |

### Theo từng lớp (mAP50-95, val)

| Lớp | mAP50-95 |
|---|---:|
| person | 0.387 |
| bicycle | 0.190 |
| car | 0.493 |
| motorcycle | 0.128 |
| bus | 0.344 |
| truck | 0.316 |

### Theo điều kiện thời tiết (val)

| Thời tiết | mAP50 | mAP50-95 |
|---|---:|---:|
| fog | 0.749 | 0.442 |
| snow | 0.723 | 0.461 |
| rain | 0.624 | 0.431 |
| sand | 0.366 | 0.225 |

**Phát hiện đáng chú ý:** Faster R-CNN cho mAP50-95 thấp nhất trong ba model đã
có (0.310 so với 0.426 của YOLOv8n và 0.369 của YOLO11n), dù nặng hơn ~14 lần
(41,3M tham số) và chậm hơn nhiều (27 FPS). Đặc biệt, recall rất cao (0.808 —
phát hiện được nhiều đối tượng) nhưng precision thấp (0.576 — nhiều cảnh báo
sai), ngược hẳn với YOLO11n. Điều kiện bão cát (sand) là điểm yếu rõ rệt nhất
(mAP50-95 chỉ 0.225). Kết quả củng cố nhận định: *mô hình hai giai đoạn nặng và
"đói" dữ liệu không phải lựa chọn tốt cho tập thời tiết bất lợi quy mô nhỏ, cả
về độ chính xác lẫn tốc độ.*

---

## B4 — RT-DETR-l Stage 2 (BDD100K → DAWN)

- Run: `stage2_dawn_rtdetr_from_bdd` (Ultralytics)
- Tham số: 31,996,070 (≈32,0M), 103.5 GFLOPs, transformer (không cần NMS)
- Tốc độ (T4): inference 23.1 ms (~43 FPS chế độ batch)

### Tổng thể (val, 206 ảnh, 1806 đối tượng)

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.712 | 0.463 | 0.516 | 0.327 |

### Theo từng lớp (mAP50-95, val)

| Lớp | mAP50-95 |
|---|---:|
| person | 0.425 |
| bicycle | 0.146 |
| car | 0.566 |
| motorcycle | 0.328 |
| bus | 0.140 |
| truck | 0.355 |

(Per-weather: chờ output `evaluate_by_weather` để bổ sung.)

**Nhận xét:** RT-DETR (32M, transformer) cho mAP50-95 0.327 — **thấp hơn rõ so với
YOLOv8n (0.426)**, chỉ nhỉnh hơn Faster R-CNN (0.310). Recall thấp (0.463). Mô hình
transformer "đói dữ liệu" cũng không phát huy trên DAWN nhỏ, củng cố nhận định
chung của TN1.

---

## KẾT LUẬN TN1 — Benchmark trên dữ liệu nhỏ (DAWN)

| Hạng | Model | Hướng | Params | mAP50 | mAP50-95 | Recall | FPS |
|---|---|---|---:|---:|---:|---:|---:|
| 🥇 | **YOLOv8n** | 1-stage | 3.0M | **0.639** | **0.426** | 0.579 | — |
| 🥈 | YOLO11n | 1-stage | 2.6M | 0.602 | 0.369 | 0.504 | 67.4 |
| 🥉 | RT-DETR-l | transformer | 32.0M | 0.516 | 0.327 | 0.463 | ~43 |
| 4 | Faster R-CNN | 2-stage | 41.3M | 0.514 | 0.310 | 0.808 | 27.2 |

**Kết luận TN1:** Trên tập thời tiết bất lợi nhỏ (DAWN), **YOLOv8n (1-stage, nhẹ
nhất) thắng rõ rệt** cả về độ chính xác lẫn tốc độ. Các mô hình nặng/"đói dữ liệu"
— RT-DETR (transformer) và Faster R-CNN (2-stage) — **xếp cuối** dù nhiều tham số
gấp 10–14 lần. Đây là cơ sở đặt ra **TN2**: liệu khi bổ sung dữ liệu thật (XWOD),
các mô hình nặng có bứt lên và đổi thứ hạng không.

---

## Bảng tổng hợp benchmark (điền dần)

| ID | Model | Trainer | Params | mAP50 | mAP50-95 | FPS (b1) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---|
| B0 | YOLOv8n | Ultralytics | 3.0M | 0.639 | 0.426 | — | val, seed 42 |
| B1 | YOLOv8s | Ultralytics | | | | | chưa train |
| B2 | YOLO11n | Ultralytics | 2.6M | 0.602 | 0.369 | 67.4 | val, seed 42 |
| B3 | YOLOv10n | Ultralytics | | | | | chưa train |
| B4 | RT-DETR-l | Ultralytics | 32.0M | 0.516 | 0.327 | ~43 | val, seed 42 |
| B5 | Faster R-CNN | TorchVision | 41.3M | 0.514 | 0.310 | 27.2 | val, seed 42 |

## Bảng ablation (điền dần)

| ID | Cấu hình | mAP50 | mAP50-95 | Δ vs B0 | Ghi chú |
|---|---|---:|---:|---:|---|
| A0 | YOLOv8n baseline | 0.639 | 0.426 | — | = B0 |
| A1 | YOLOv8n + CBAM Neck | | | | chưa train |
| A2 | YOLOv8n + weather aug | | | | dự kiến |
| A3 | YOLOv8n + dehaze | | | | dự kiến |
