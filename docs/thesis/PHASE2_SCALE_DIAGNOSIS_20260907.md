# Chẩn đoán nút thắt sau Phase 2 — nhật ký thực nghiệm 2026-09-07

- Mô hình: RT-DETR-L, khởi tạo từ `phase2_final_rtdetr`
- Nhánh mã: `feat/p2-a1-bdd-retrieval`
- Máy: VAST C.50053805, RTX 5090 32 GB
- Phạm vi: huấn luyện A1-DINO v2, dò ngưỡng theo lớp, chẩn đoán nguyên nhân lớp yếu

Tài liệu này ghi lại thông số đo được, trình tự thực hiện, các lỗi gặp phải và
hướng đã quyết, để chương 4 có nguồn tra cứu và để phiên làm việc sau nối tiếp
được.

---

## 1. Bối cảnh

Vòng ablation trước (Phase 2 / A0R / A1-DINO v1) cho kết quả nằm trong khoảng
nhiễu ở cả bốn tập test, và A1-DINO v1 còn thấp hơn A0R trên F1 tổng thể. Phân
tích chuyển trạng thái cho thấy v1 sửa được 35 đối tượng nhưng làm hỏng 51 —
ròng **−16**. Chi tiết ở `docs/f1_ablation_report.md`.

Ba lỗi phương pháp của v1 được xác định: nhúng đặc trưng ở mức toàn ảnh thay vì
vùng cắt, không có hạn ngạch theo lớp, và ACDC bị XWOD lấn át trong tập truy vấn.
Phiên này sửa cả ba rồi huấn luyện lại thành v2.

---

## 2. Thông số đo được

### 2.1 A1-DINO v2 so với Phase 2 (F1 trên tập val)

| Tập | Phase 2 | A1-DINO v2 | Δ |
|---|---:|---:|---:|
| XWOD | 0.8109 | 0.8149 | +0.0040 |
| BDD | 0.6231 | 0.6306 | +0.0075 |
| DAWN | 0.7512 | 0.7711 | +0.0199 |
| ACDC | 0.4639 | 0.4698 | +0.0059 |

v2 thắng ở cả bốn tập, trong khi v1 thua ở XWOD và DAWN. Ba sửa đổi phương pháp
có tác dụng, nhưng biên độ vẫn nhỏ: +0.004 đến +0.020, một seed, không có khoảng
tin cậy.

Trên các ô lớp hiếm đủ mẫu (≥100 hộp), mức tăng trung bình là **+0.0127**, trong
khi khoảng cách còn lại tới mức 0.80 là khoảng **0.25**. Truy hồi lấp được chừng
5% khoảng cách.

### 2.2 Hạn ngạch truy hồi và trần của nguồn dữ liệu

Pool BDD còn lại sau khi trừ 30.000 ảnh đã dùng: **39.863** ảnh.

| Lớp | Ảnh có sẵn trong pool | Hạn ngạch | Thực nhận |
|---|---:|---:|---:|
| motorcycle | 1.391 | 2.000 | 1.391 (hết nguồn) |
| bicycle | 2.550 | 2.000 | 2.000 |
| bus | 4.946 | 1.000 | 1.000 |

Với motorcycle, truy hồi suy biến thành lấy toàn bộ — thiếu 609 so với hạn ngạch.
Đây là trần cứng của nguồn BDD, cần nêu trong phần hạn chế.

### 2.3 Ngưỡng tin cậy theo lớp

Ultralytics đọc mọi lớp tại một ngưỡng chung tối đa hoá F1 trung bình. Dò ngưỡng
riêng cho từng lớp trên val, rồi áp lên test để kiểm tra khả năng tổng quát hoá:

| Tập | Δ khi dò (val) | Δ trên test | Giữ lại |
|---|---:|---:|---:|
| ACDC | +0.0216 | **+0.0103** | 48% |
| BDD | +0.0067 | +0.0013 | 19% |
| XWOD | +0.0051 | **−0.0027** | âm |

Mức tăng thật khoảng **+0.003 trung bình, và có thể âm**. Con số đo trên val bị
thổi lên do khớp nhiễu.

Quy luật rút ra: ngưỡng riêng chỉ có lợi khi phân phối điểm giữa các lớp lệch rõ
rệt. Ở ACDC, motorcycle tối ưu tại 0.38 còn bus tại 0.66 so với ngưỡng chung
0.52 — độ lệch lớn và có hệ thống nên chuyển giao được. Ở XWOD mọi lớp chỉ lệch
0.01–0.10 quanh ngưỡng chung, tức đường cong F1 đã phẳng, và mức tăng đo trên val
chỉ là nhiễu.

### 2.4 Chẩn đoán lớp yếu trên ACDC val

**Kích thước hộp ground truth, tính bằng pixel gốc:**

| Lớp | ACDC | XWOD | Tỉ lệ | % nhỏ ACDC | % nhỏ XWOD |
|---|---:|---:|---:|---:|---:|
| person | 12,1 | 82,8 | 6,8× | 91,7% | 5,6% |
| bicycle | 13,9 | 205,0 | 14,7× | 79,1% | 2,6% |
| car | 14,8 | 86,3 | 5,8× | 80,7% | 10,4% |
| motorcycle | 14,1 | 75,0 | 5,3× | 82,4% | 11,0% |
| bus | 30,9 | 175,7 | 5,7× | 50,0% | 3,8% |
| truck | 20,0 | 141,8 | 7,1× | 74,3% | 2,6% |

ACDC có 80–92% vật thể nhỏ (dưới 32×32 px), XWOD chỉ 3–11%. Ảnh ACDC là
1920×1080, vật thể trung vị 14 px — chiếm 0,7% chiều rộng ảnh.

**Recall theo dải kích thước:**

| Lớp | nhỏ | vừa | lớn |
|---|---:|---:|---:|
| person | 0,511 | 0,895 | 1,000 |
| bicycle | 0,294 | 0,500 | 1,000 |
| car | 0,696 | 0,940 | 1,000 |
| motorcycle | 0,286 | 0,667 | — |
| bus | 0,000 | 0,750 | 1,000 |
| truck | 0,564 | 0,533 | 1,000 |

Vật thể lớn đạt recall 1,000 ở cả sáu lớp. Bus nhỏ bỏ sót toàn bộ 10/10 nhưng
bus vừa đạt 0,750 và bus lớn 1,000. Mô hình phát hiện tốt các lớp này khi chúng
đủ lớn.

**Phân loại lỗi bỏ sót:**

| Lớp | Không phát hiện gì | Có hộp lớp khác đè lên |
|---|---:|---:|
| bicycle | 85,7% | 14,3% |
| motorcycle | 77,3% | 22,7% |
| bus | 100,0% | 0% |

Không phải nhầm lớp, mà là không nhìn thấy.

**Số instance trong tập huấn luyện:**

| Lớp | ACDC train | XWOD train |
|---|---:|---:|
| person | 1.495 | 6.287 |
| bicycle | 299 | 917 |
| car | 5.263 | 13.651 |
| motorcycle | 193 | 1.049 |
| bus | 148 | 378 |
| truck | 382 | 3.157 |

### 2.5 Quét độ phân giải lúc suy luận

Cùng checkpoint, không huấn luyện lại, F1 trên val:

| imgsz | ACDC | Δ | XWOD | Δ |
|---|---:|---:|---:|---:|
| 640 | 0.4696 | — | 0.8149 | — |
| 960 | 0.4924 | +0.0227 | 0.7765 | −0.0384 |
| 1024 | 0.4549 | −0.0148 | 0.7438 | −0.0712 |
| 1280 | 0.3710 | −0.0987 | 0.5189 | −0.2961 |
| 1600 | 0.2388 | −0.2308 | 0.1910 | −0.6239 |
| 1920 | 0.1736 | −0.2960 | 0.0770 | −0.7380 |

Kết quả âm và giải thích được: mô hình huấn luyện ở 640 nên bị khoá vào tỉ lệ đầu
vào đó. XWOD sụp mạnh hơn ACDC gấp đôi vì vật thể XWOD vốn đã lớn, phóng to càng
đẩy chúng ra xa dải quen thuộc. ACDC được lợi nhỏ tại 960 vì vật thể quá nhỏ nên
lợi ích còn thắng thiệt hại, nhưng chỉ trong một khoảng hẹp.

---

## 3. Trình tự thực hiện

1. Sửa `retrieve_dinov2.py`: nhúng theo vùng cắt, hạn ngạch theo lớp, ACDC nhân
   đôi trọng số truy vấn, thứ tự lấp hạn ngạch theo độ khan hiếm.
2. Chuyển nhãn BDD100K chính thức sang định dạng YOLO sáu lớp (69.863 ảnh train).
3. Dựng pool truy hồi: 69.863 − 30.000 đã dùng = **39.863** ảnh, chồng lấn với
   train/val/test đã dùng đều bằng 0.
4. Chạy truy hồi và ghép tập dữ liệu; kiểm định bất biến đạt: cả hai nhánh
   5.000 ảnh truy hồi, 55.985 ảnh train, 1.001 ảnh val.
5. Huấn luyện A1-DINO v2, 20 epoch, lr 1e-5, batch 16, imgsz 640.
6. Đo F1 theo lớp trên val, so với Phase 2.
7. Dò ngưỡng theo lớp trên val, kiểm chứng chuyển giao sang test.
8. Chẩn đoán lớp yếu theo bốn trục: điều kiện, kích thước, kiểu bỏ sót, số mẫu.
9. Quét độ phân giải lúc suy luận.

---

## 4. Nguyên nhân thực sự

Giả thuyết ban đầu là **thời tiết**: ACDC yếu vì sương mù, đêm, tuyết. Giả thuyết
này **sai**.

Bảng per-weather sẵn có cho thấy trên ACDC, sương mù đạt 0.388 và mưa 0.250, gần
như trùng khớp với XWOD (0.387 và 0.243). Hai điều kiện này không còn khoảng
trống nào. Chỉ đêm (0.121) và tuyết (0.234) là thấp.

Nhưng khi tách theo lớp, số mẫu ở mỗi ô điều kiện quá nhỏ để kết luận: bicycle
trong sương mù chỉ có 2 hộp, motorcycle trong sương mù có 1 hộp. Ngay cả car —
lớp duy nhất đủ mẫu — cũng chỉ chênh 0,09 giữa đêm và tuyết.

Ngược lại, hiệu ứng kích thước lớn và nhất quán ở cả sáu lớp: recall 1,000 cho
vật thể lớn so với 0,00–0,70 cho vật thể nhỏ. Và ACDC có 80–92% vật thể nhỏ trong
khi XWOD chỉ 3–11%.

Kết luận: **nút thắt là tỉ lệ vật thể, không phải điều kiện thời tiết.** Phần lớn
khoảng cách "ban đêm" đo bằng mAP50-95 cũng là hệ quả gián tiếp, vì mAP50-95 phạt
nặng sai lệch định vị mà vật thể nhỏ thì định vị luôn kém.

Điểm quan trọng: XWOD **chính là** dữ liệu thời tiết xấu, và mô hình đạt F1 0.815
ở đó, với bicycle 0.904 — cao hơn cả car. Thời tiết xấu tự nó không khó; vật thể
14 pixel mới khó.

---

## 5. Vấn đề gặp phải và cách xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| Pool chỉ có 531 ảnh thay vì ~40.000 | `BDD100K.zip` là bản chính thức (ảnh + JSON), không phải bản Kaggle định dạng YOLO mà `prepare_bdd100k_yolo.py` mong đợi | Viết `convert_bdd100k_json_to_yolo.py`, ánh xạ 10 lớp BDD sang 6 lớp dự án |
| Convert chỉ tìm thấy 1.154/69.863 ảnh | Thư mục train chứa bốn thư mục con (`trainA/B`, `testA/B`), script chỉ quét phẳng | Lập chỉ mục đệ quy theo tên tệp |
| Hạn ngạch motorcycle không đạt | Pool chỉ có 1.391 ảnh chứa motorcycle | Ghi nhận là trần của nguồn; thêm `class_availability` và `class_quota_shortfall` vào thống kê |
| Lớp khan hiếm bị lớp dồi dào lấy mất ứng viên | Thứ tự lấp hạn ngạch dựa trên **kích thước hạn ngạch**, trong khi điểm số ưu tiên ảnh nhiều lớp hiếm nên lớp dồi dào đi trước sẽ vét mất | Đổi sang lấp theo **độ khan hiếm** của nguồn |
| Hết VRAM khi dò ngưỡng trên BDD | Truyền cả danh sách 3.000 ảnh vào `model.predict` khiến batch bằng số ảnh, tensor warmup 36 GB | Chia lô cố định, mặc định 8 ảnh |
| Hai chế độ tính F1 lệch nhau ở ca sát ngưỡng | Một bên ép `float32`, một bên dùng `float64` | Thống nhất `float64` |
| Ảnh `flooding_train_0007` bị đọc thành điều kiện `rain` | Tìm chuỗi con: `"train"` chứa `"rain"` | Tách token và loại trừ các từ chỉ split |
| Khối metrics rỗng nhưng không báo lỗi | Thiếu tệp nguồn bị bỏ qua im lặng | Dừng hẳn khi thiếu; một khối trống trông y hệt một kết quả |
| Ô thiếu dữ liệu hiển thị thành `0.0000` | Script tóm tắt dùng giá trị mặc định 0 | Hiển thị `THIẾU`; thêm `coverage.json` ghi số ô kỳ vọng và thực có |

Ba mục cuối cùng một dạng: **thiếu dữ liệu bị trình bày như một giá trị**. Nếu
không phát hiện, bảng trong luận văn sẽ chứa số bịa.

---

## 6. Hướng đã quyết

Bốn hướng đã xét, hai bị loại bằng thực nghiệm:

| Hướng | Trạng thái | Căn cứ |
|---|---|---|
| Tăng cường thời tiết bằng mô hình sinh ảnh | **Loại** | Nút thắt là kích thước; sinh sương mù không làm vật thể to lên |
| Phóng to ảnh lúc suy luận | **Loại** | Đã thử sáu mức; mô hình khoá ở tỉ lệ 640, cả hai tập đều giảm |
| Suy luận theo ô cắt (SAHI) | **Làm trước** | Giữ đầu vào ở đúng 640; vật thể 14 px không còn co xuống 4,6 px |
| Huấn luyện lại ở độ phân giải cao hơn | Sau đó | Mô hình học lại tiên nghiệm tỉ lệ; tốn một lượt huấn luyện |

Suy luận theo ô cắt được làm trước vì nó là phép thử rẻ nhất cho giả thuyết mà cả
hai hướng còn lại cùng dựa vào. Nếu có hiệu quả, giả thuyết tỉ lệ được xác nhận
và việc bỏ GPU huấn luyện lại là có cơ sở. Nếu không, giả thuyết sai và phải quay
lại xét khả năng thiếu dữ liệu — ACDC train chỉ có 193 instance motorcycle và 148
instance bus.

Ước lượng mức lợi, suy từ bảng recall theo kích thước: cắt ô loại bỏ bước thu nhỏ
0,33 lần, tức phóng đại hiệu dụng 3 lần, đưa các vật thể đang ở dải "nhỏ" sang
hành xử như dải "vừa" — nơi recall là 0,50–0,94 thay vì 0,00–0,70. Đây là ước
lượng có căn cứ, không phải dự báo chắc chắn; ánh xạ 3 lần không chính xác tuyệt
đối và cắt ô có cái giá riêng là vật thể bị cắt ở biên ô cùng chi phí suy luận
tăng 6–9 lần.

---

## 7. Hạn chế cần nêu trong luận văn

- Mỗi cấu hình chỉ chạy một seed, không có khoảng tin cậy. Biên độ giữa các cấu
  hình (0.004–0.020) nhỏ hơn sai số mong đợi của một lần chạy.
- A1-DINO v1 đã được đánh giá trên tập test, và kết quả đó dẫn tới việc sửa
  phương pháp rồi huấn luyện v2. Số test của v2 vì vậy đã qua một vòng lặp có
  thông tin test, không còn là ước lượng giữ lại thuần tuý.
- Nhiều ô lớp hiếm trên val có dưới 100 hộp (ACDC bus 20, motorcycle 34,
  bicycle 43; DAWN bicycle 7). F1 ở các ô này không đủ tin cậy để xếp hạng.
- Hạn ngạch motorcycle không đạt do pool BDD cạn ở 1.391 ảnh.
- Phân tích theo điều kiện thời tiết ở mức từng lớp không kết luận được vì số
  mẫu mỗi ô quá nhỏ.

---

## 8. Tệp và mã liên quan

| Đường dẫn | Vai trò |
|---|---|
| `scripts/convert_bdd100k_json_to_yolo.py` | Chuyển nhãn BDD100K chính thức sang YOLO sáu lớp |
| `scripts/retrieve_dinov2.py` | Truy hồi theo vùng cắt, hạn ngạch, mật độ lớp hiếm |
| `scripts/eval_f1_matrix.py` | F1 theo lớp trên nhiều nhánh và nhiều tập |
| `scripts/tune_conf_thresholds.py` | Dò ngưỡng theo lớp và kiểm chứng chuyển giao |
| `scripts/diagnose_weak_classes.py` | Chẩn đoán theo điều kiện, kích thước, kiểu bỏ sót, số mẫu |
| `scripts/report_f1_ablation.py` | Sinh báo cáo F1 từ bộ metrics chính thức |
| `scripts/add_ablation_sheet.py` | Thêm tab kết quả vào workbook |
| `docs/f1_ablation_report.md` | Báo cáo vòng ablation trước |
| `docs/adverse_weather_model_results_google_sheets.xlsx` | Tab `Ablation — Retrieval + Conf` |

Lưu ý: `docs/thesis/experiment-progress.md` mang ngày 2026-08-09 và mô tả pipeline
chọn YOLOv8n cùng 2.000 ảnh BDD replay. Nội dung đó đã lỗi thời so với pipeline
RT-DETR-L hiện hành; khi có mâu thuẫn, lấy tài liệu này làm chuẩn.
