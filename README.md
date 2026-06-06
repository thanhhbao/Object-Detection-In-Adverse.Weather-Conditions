# YOLOv8-CBAM Ablation Study on DAWN

Pipeline nghiên cứu cho đề tài cải tiến YOLOv8 bằng CBAM trên tập DAWN. Thiết kế
thực nghiệm chỉ thay đổi kiến trúc attention; seed, dữ liệu, pretrained weights và
siêu tham số được giữ giống nhau giữa baseline và mô hình cải tiến.

Hướng dẫn chạy end-to-end trên Google Colab: `docs/COLAB.md`.

## 1. Cấu trúc dữ liệu

Giải nén DAWN vào `data/raw/DAWN`. Script hỗ trợ ảnh và Pascal VOC XML nằm ở bất
kỳ thư mục con nào, miễn ảnh và XML có cùng tên file (ví dụ `rain_001.jpg` và
`rain_001.xml`).

```text
data/
├── raw/DAWN/
│   ├── Fog/
│   ├── Rain/
│   ├── Sand/
│   └── Snow/
└── processed/dawn_yolo/          # được tạo tự động
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── dataset.yaml
    └── manifest.csv
```

Sáu lớp mặc định: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
Các alias phổ biến như `pedestrian` và `motorbike` được chuẩn hóa tự động.

## 2. Cài đặt

Khuyến nghị Python 3.10 hoặc 3.11 và CUDA GPU. Python 3.13 thường chưa tương
thích đồng đều với toàn bộ stack học sâu.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu

### Stage 1: BDD100K clear daytime

Nếu dùng Kaggle `a7madmostafa/bdd100k-yolo` như cấu trúc
`bdd100k_yolo_raw/{train,val,test}/images|labels`, chạy script remap YOLO sẵn:

```bash
python scripts/remap_bdd100k_yolo.py \
  --raw-dir data/raw/bdd100k_yolo_raw \
  --output-dir data/processed/bdd100k_6cls_yolo \
  --max-train-images 5000 \
  --max-val-images 1000 \
  --seed 42 \
  --clean
```

Mặc định script dùng symlink ảnh để tiết kiệm dung lượng. Nếu môi trường không
hỗ trợ symlink, thêm `--copy-images`.

Nếu dùng BDD100K JSON gốc, chạy script convert JSON:

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

Script lọc `weather=clear`, `timeofday=daytime`, map về 6 lớp mục tiêu và tạo
dataset YOLO train/val cho Stage 1.

### Stage 2: SHIFT synthetic adverse weather

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

Script hỗ trợ annotation dạng BDD/Scalabel JSON hoặc COCO JSON, sau đó chia
train/val có phân tầng theo weather.

### Stage 3/4: ACDC + DAWN real adverse weather

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

Script gộp ACDC và DAWN trước, rồi chia stratified 70/15/15 theo weather để dùng
chung cho baseline và CBAM.

### DAWN riêng lẻ

```bash
python scripts/prepare_dawn.py \
  --raw-dir data/raw/DAWN \
  --output-dir data/processed/dawn_yolo \
  --imgsz 640 \
  --seed 42 \
  --clean
```

Script dùng letterbox về 640x640, cập nhật bbox tương ứng, chia có phân tầng theo
thư mục thời tiết với tỷ lệ 70/15/15, và tạo `dataset.yaml`.
`--clean` ngăn file cũ từ lần chia trước gây rò rỉ giữa các split.

Kiểm tra `manifest.csv`, số ảnh mỗi split và vài ảnh/nhãn thủ công trước khi
huấn luyện. Không thay đổi test set sau khi bắt đầu thực nghiệm.

## 4. Chạy ablation

Chạy hai thí nghiệm với cùng cấu hình:

```bash
python scripts/train.py --variant baseline --config configs/experiment.yaml
python scripts/train.py --variant cbam --config configs/experiment.yaml
```

Đánh giá checkpoint tốt nhất trên test set:

```bash
python scripts/evaluate.py --variant baseline --config configs/experiment.yaml
python scripts/evaluate.py --variant cbam --config configs/experiment.yaml
python scripts/compare_results.py --config configs/experiment.yaml
```

Kết quả cuối nằm tại `runs/ablation/ablation_comparison.csv`. Mỗi thí nghiệm nên
được chạy tối thiểu 3 seed và báo cáo mean ± standard deviation. Chỉ kết luận
CBAM hiệu quả nếu mức tăng ổn định qua các seed, không chỉ ở một lần chạy.
Giao thức báo cáo học thuật chi tiết nằm tại `docs/RESEARCH_PROTOCOL.md`.

## Thiết kế đối chứng

- Baseline và CBAM dùng cùng kiến trúc YOLOv8n resolved; CBAM chỉ được thêm sau
  bốn khối C2f ở Neck.
- Cùng split, seed, epoch, batch size, augmentation, optimizer và pretrained
  checkpoint.
- Chọn `best.pt` bằng validation set; chỉ dùng test set để báo cáo cuối.
- Báo cáo cả độ chính xác và chi phí: Precision, Recall, mAP50, mAP50-95,
  inference ms/image, FPS và số tham số.
- Để so sánh thời gian công bằng, benchmark trên cùng máy, batch=1, cùng imgsz,
  warmup và số vòng lặp.
# Object-Detection
