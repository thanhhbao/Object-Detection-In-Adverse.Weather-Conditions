# Giao thức nghiên cứu và đối chứng

## Câu hỏi và giả thuyết

**RQ:** CBAM tại Neck của YOLOv8n có cải thiện khả năng nhận diện trên ảnh thời
tiết bất lợi của DAWN hay không, và chi phí tính toán tăng bao nhiêu?

- `H0`: mAP50-95 của YOLOv8n-CBAM không cao hơn baseline một cách ổn định.
- `H1`: mAP50-95 của YOLOv8n-CBAM cao hơn baseline qua nhiều seed.

CBAM được đặt sau bốn đầu ra C2f trong Neck vì đây là các feature map đã trải qua
fusion đa tỉ lệ. Channel attention chọn đặc trưng hữu ích; spatial attention tập
trung vùng ảnh quan trọng khi độ tương phản bị giảm bởi sương mù, mưa hoặc tuyết.

## Ma trận ablation tối thiểu

| ID | Kiến trúc | Pretrained | Split | Hyperparameters |
|---|---|---|---|---|
| A0 | YOLOv8n gốc | yolov8n.pt | cố định | cố định |
| A1 | YOLOv8n + CBAM tại Neck | yolov8n.pt | giống A0 | giống A0 |

Không thay augmentation, epoch, optimizer hoặc ảnh đầu vào giữa A0 và A1. File
`pretrained_transfer.json` phải cho thấy cùng số tensor/layer YOLO được truyền
vào hai mô hình. Các tham số CBAM mới được khởi tạo ngẫu nhiên.

Mở rộng sau thí nghiệm tối thiểu:

| ID | Thay đổi duy nhất | Mục đích |
|---|---|---|
| A2 | CBAM chỉ tại Backbone | Xác định vị trí attention tốt hơn |
| A3 | SE tại Neck | Tách tác động channel và spatial attention |
| A4 | CBAM reduction 8/16/32 | Đánh giá độ nhạy cấu hình |

Chỉ thực hiện A2-A4 sau khi A0-A1 đã chạy ổn định.

## Quy trình

1. Chia DAWN một lần với seed 42. Lưu `manifest.csv` và không thay test set.
2. Kiểm tra trực quan bbox sau letterbox và thống kê số instance theo lớp/split.
3. Huấn luyện A0 và A1 với cùng cấu hình. Chọn checkpoint bằng validation set.
4. Đánh giá `best.pt` đúng một lần trên test set.
5. Lặp lại huấn luyện với tối thiểu ba training seed, nhưng giữ nguyên data split.
6. Báo cáo trung bình, độ lệch chuẩn và kết quả theo từng loại thời tiết nếu có.

## Cách báo cáo mức tăng

Hai đại lượng sau khác nhau và cần ghi rõ:

```text
Mức tăng tuyệt đối (percentage points) = mAP_cải_tiến - mAP_gốc
Mức tăng tương đối (%) = (mAP_cải_tiến - mAP_gốc) / mAP_gốc * 100
FPS = 1000 / inference_time_ms
```

Ví dụ mAP50-95 tăng từ `0.400` lên `0.420` nghĩa là tăng `0.020`, hay `2.0`
điểm phần trăm, tương đương tăng tương đối `5.0%`.

## Bảng kết quả đề xuất

| Model | Precision | Recall | mAP50 | mAP50-95 | Params | ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | mean ± std | mean ± std | mean ± std | mean ± std | ... | ... | ... |
| YOLOv8n-CBAM | mean ± std | mean ± std | mean ± std | mean ± std | ... | ... | ... |

Nên bổ sung AP theo lớp và theo `fog/rain/sand/snow`. Một mức tăng overall có thể
che giấu việc mô hình tốt hơn ở một điều kiện nhưng kém hơn đáng kể ở điều kiện
khác.

## Threats to validity

- DAWN nhỏ, vì vậy kết quả một seed có phương sai cao.
- Chia ngẫu nhiên có thể làm ảnh gần giống nhau xuất hiện ở nhiều split; cần kiểm
  tra nguồn ảnh và loại bỏ duplicate nếu có.
- Test-set tuning làm kết quả lạc quan giả tạo. Mọi quyết định kiến trúc phải dựa
  trên validation set.
- FPS phụ thuộc phần cứng và cách đo. Chỉ so sánh khi cùng máy, precision, batch,
  kích thước ảnh, warmup và số vòng lặp.

## Tài liệu nền

- DAWN paper: https://arxiv.org/abs/2008.05402
- CBAM paper: https://arxiv.org/abs/1807.06521
- Ultralytics model YAML guide:
  https://docs.ultralytics.com/guides/model-yaml-config/

