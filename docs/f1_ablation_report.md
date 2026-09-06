# Báo cáo F1 — so sánh Phase 2, A0R và A1-DINO

Đánh giá trên tập test giữ lại của bốn bộ dữ liệu. Ba cấu hình dùng chung
kiến trúc RT-DETR-L và cùng số ảnh huấn luyện (55.985), chỉ khác ở cách
chọn 5.000 ảnh bổ sung: A0R lấy ngẫu nhiên trong nhóm ảnh chứa lớp hiếm,
A1-DINO truy hồi bằng độ tương đồng đặc trưng DINOv2.

## 1. F1 tổng thể

F1 tính từ precision và recall trung bình theo lớp: `F1 = 2PR/(P+R)`.

| Tập test | Vai trò | Phase 2 | A0R | A1-DINO | Tốt nhất | Biên độ |
|---|---|---:|---:|---:|---|---:|
| XWOD | cùng miền | 0.7743 | **0.7796** | 0.7670 | A0R | 0.0126 |
| BDD | miền nguồn | 0.6202 | **0.6244** | 0.6241 | A0R | 0.0043 |
| DAWN | ngoài miền | **0.7990** | 0.7960 | 0.7924 | Phase 2 | 0.0066 |
| ACDC | giữ lại | 0.4684 | **0.4717** | 0.4712 | A0R | 0.0033 |

| Tập test | A0R − Phase 2 | A1-DINO − Phase 2 | A1-DINO − A0R |
|---|---:|---:|---:|
| XWOD | +0.0053 | -0.0073 | -0.0126 |
| BDD | +0.0043 | +0.0039 | -0.0003 |
| DAWN | -0.0030 | -0.0066 | -0.0036 |
| ACDC | +0.0033 | +0.0028 | -0.0005 |

Biên độ giữa ba cấu hình lớn nhất là 0.0126 và nhỏ nhất là
0.0033. Việc bổ sung 5.000 ảnh — khoảng 9% tập huấn luyện —
làm F1 thay đổi dưới một điểm phần trăm rưỡi ở mọi tập. A0R, tức lấy mẫu
ngẫu nhiên, đạt F1 cao hơn A1-DINO ở cả bốn tập.

## 2. Kết quả theo lớp

Bộ metrics lưu lại chỉ có precision và recall ở mức tổng thể, nên phần theo
lớp dùng mAP50 và mAP50-95. Lớp hiếm được in đậm.

### XWOD

| Lớp | Phase 2 mAP50 | A0R mAP50 | A1-DINO mAP50 | A1 − A0R |
|---|---:|---:|---:|---:|
| person | 0.7943 | 0.7914 | 0.7828 | -0.0086 |
| **bicycle** | 0.9122 | 0.9266 | 0.8998 | -0.0269 |
| car | 0.8659 | 0.8679 | 0.8617 | -0.0062 |
| **motorcycle** | 0.7494 | 0.7573 | 0.7305 | -0.0268 |
| **bus** | 0.5462 | 0.5429 | 0.5344 | -0.0086 |
| truck | 0.8035 | 0.8029 | 0.8007 | -0.0022 |

### BDD

| Lớp | Phase 2 mAP50 | A0R mAP50 | A1-DINO mAP50 | A1 − A0R |
|---|---:|---:|---:|---:|
| person | 0.6603 | 0.6662 | 0.6521 | -0.0140 |
| **bicycle** | 0.5063 | 0.5084 | 0.5009 | -0.0075 |
| car | 0.8122 | 0.8137 | 0.8080 | -0.0057 |
| **motorcycle** | 0.4415 | 0.4584 | 0.4677 | +0.0094 |
| **bus** | 0.5715 | 0.5820 | 0.5695 | -0.0125 |
| truck | 0.5712 | 0.5651 | 0.5625 | -0.0026 |

### DAWN

| Lớp | Phase 2 mAP50 | A0R mAP50 | A1-DINO mAP50 | A1 − A0R |
|---|---:|---:|---:|---:|
| person | 0.8356 | 0.8438 | 0.8197 | -0.0241 |
| **bicycle** | 0.7027 | 0.6926 | 0.7659 | +0.0733 |
| car | 0.9286 | 0.9279 | 0.9242 | -0.0037 |
| **motorcycle** | 0.8022 | 0.7964 | 0.8889 | +0.0926 |
| **bus** | 0.7380 | 0.7415 | 0.7568 | +0.0153 |
| truck | 0.7652 | 0.7572 | 0.7483 | -0.0089 |

### ACDC

| Lớp | Phase 2 mAP50 | A0R mAP50 | A1-DINO mAP50 | A1 − A0R |
|---|---:|---:|---:|---:|
| person | 0.4771 | 0.4790 | 0.4871 | +0.0081 |
| **bicycle** | 0.2184 | 0.2148 | 0.2087 | -0.0061 |
| car | 0.7010 | 0.7068 | 0.6985 | -0.0083 |
| **motorcycle** | 0.2319 | 0.2656 | 0.2184 | -0.0472 |
| **bus** | 0.2567 | 0.2888 | 0.2587 | -0.0301 |
| truck | 0.5554 | 0.5578 | 0.5557 | -0.0022 |

### Các ô yếu nhất

Xếp theo mAP50 của Phase 2, thấp nhất trước:

| Tập | Lớp | mAP50 |
|---|---|---:|
| ACDC | bicycle | 0.2184 |
| ACDC | motorcycle | 0.2319 |
| ACDC | bus | 0.2567 |
| BDD | motorcycle | 0.4415 |
| ACDC | person | 0.4771 |
| BDD | bicycle | 0.5063 |
| XWOD | bus | 0.5462 |
| ACDC | truck | 0.5554 |

## 3. Phân tích lỗi của A1-DINO

Theo dõi từng đối tượng ground truth qua hai mô hình, trên 7188 ảnh truy vấn XWOD và ACDC, ngưỡng phát hiện 0.25:

| Chuyển trạng thái | Số lượng |
|---|---:|
| Phát hiện được ở cả hai | 2470 |
| Sai ở cả hai | 428 |
| A1-DINO **sửa được** | 35 |
| A1-DINO **làm hỏng** | 51 |

Ròng: **-16** đối tượng. A1-DINO làm hỏng nhiều hơn số nó sửa được.

| Lớp | GT | Phase 2 bỏ sót | A1 sửa | A1 làm hỏng | Ròng | Tỉ lệ phục hồi |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 1216 | 195 | 10 | 20 | -10 | 5.1% |
| motorcycle | 1242 | 164 | 20 | 22 | -2 | 12.2% |
| bus | 526 | 104 | 5 | 9 | -4 | 4.8% |

Cả ba lớp hiếm đều hỏng nhiều hơn sửa.

### Độ tin cậy trên chính những đối tượng Phase 2 bỏ sót

| Lớp | Phase 2 | A1-DINO | Thay đổi | Ngưỡng |
|---|---:|---:|---:|---:|
| bicycle | 0.176 | 0.193 | +0.018 | 0.25 |
| motorcycle | 0.170 | 0.220 | +0.050 | 0.25 |
| bus | 0.160 | 0.161 | +0.001 | 0.25 |

Mô hình **có** nhận ra các đối tượng này, nhưng ở mức tin cậy dưới
ngưỡng. Dữ liệu bổ sung đẩy độ tin cậy lên — rõ nhất ở motorcycle —
song vẫn chưa đủ vượt ngưỡng. Nguyên nhân hỏng vì vậy không phải là
mô hình chưa từng học lớp đó, mà là học rồi nhưng không đủ chắc chắn.

Các giá trị trên là điểm tin cậy sau NMS, không phải xác suất đã hiệu chuẩn.

## 4. Nhận định

1. Khác biệt giữa ba cấu hình nằm trong khoảng nhiễu. Truy hồi bằng DINOv2
   không vượt được lấy mẫu ngẫu nhiên; trên F1 tổng thể nó còn thấp hơn ở
   cả bốn tập test.
2. Phân tích chuyển trạng thái cho thấy A1-DINO tác động gần như ngẫu nhiên
   lên chính những lỗi mà nó nhắm tới, với xu hướng lệch nhẹ về phía xấu.
3. Nguồn dữ liệu bổ sung là BDD100K — ảnh đô thị điều kiện quang. Bổ sung
   thêm ảnh trời quang không thu hẹp được khoảng cách ở điều kiện sương mù,
   ban đêm và tuyết, vốn là chỗ ACDC yếu nhất.
4. Khoảng trống lớn nhất không nằm giữa các cấu hình mà nằm giữa các miền:
   ACDC đạt F1 0.4684 với recall 0.384,
   trong khi XWOD đạt 0.7743. Chênh lệch 0.3058 lớn hơn nhiều lần
   mọi khác biệt giữa ba cấu hình.

## 5. Hạn chế của báo cáo

Bộ metrics đã lưu chỉ chứa precision và recall ở mức tổng thể, nên **F1 theo
từng lớp chưa tính được** — phần theo lớp trong báo cáo này dùng mAP thay thế.
Muốn có F1 theo lớp, cùng phép kiểm định val ↔ test, cần chạy lại đánh giá
bằng `scripts/eval_f1_matrix.py` với ba checkpoint đã lưu.

Ngoài ra mọi số ở đây đo trên một lần huấn luyện cho mỗi cấu hình, không có
nhiều seed, nên chưa có khoảng tin cậy. Với biên độ nhỏ như đã thấy, đây là
lý do chính để không diễn giải thứ hạng giữa ba cấu hình như một kết luận.
