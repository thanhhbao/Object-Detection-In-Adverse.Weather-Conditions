# -*- coding: utf-8 -*-
"""Nội dung khóa luận (tách riêng để dễ cập nhật). fill_thesis.py đọc file này.

Mỗi chương: {"title", "body": [(kind, text), ...]} với kind = h2 | h3 | b.
Số liệu Chương 4 lấy từ docs/RESULTS.md (val, seed 42, Tesla T4).
"""

LOI_MO_DAU = [
    "Giám sát giao thông thông minh ngày càng đóng vai trò quan trọng trong quản "
    "lý đô thị, an toàn giao thông và điều phối phương tiện. Một thành phần cốt lõi "
    "là khả năng phát hiện chính xác người và phương tiện từ hình ảnh camera. Tuy "
    "nhiên, trong điều kiện thời tiết bất lợi như sương mù, mưa, bão cát hay tuyết, "
    "chất lượng hình ảnh suy giảm khiến độ chính xác của mô hình phát hiện giảm sút.",
    "Khóa luận tập trung thiết kế và phát triển mô hình học sâu phát hiện người và "
    "phương tiện trong điều kiện thời tiết bất lợi, đồng thời so sánh có hệ thống ba "
    "hướng tiếp cận hiện đại nhằm tìm giải pháp cân bằng giữa độ chính xác và tốc độ, "
    "phù hợp cho ứng dụng giám sát giao thông thực tế.",
]

LOI_CAM_ON = [
    "Em xin chân thành cảm ơn giảng viên hướng dẫn đã tận tình định hướng và góp ý "
    "trong suốt quá trình thực hiện khóa luận. Em cũng xin cảm ơn quý thầy cô Khoa "
    "Công nghệ Thông tin, Trường Đại học Nguyễn Tất Thành, cùng gia đình và bạn bè "
    "đã hỗ trợ, động viên để em hoàn thành đề tài này.",
]

ABBREVIATIONS = [
    ("AP", "Average Precision"),
    ("BDD100K", "Berkeley DeepDrive 100K Dataset"),
    ("CBAM", "Convolutional Block Attention Module"),
    ("CNN", "Convolutional Neural Network"),
    ("DAWN", "Detection in Adverse Weather Nature Dataset"),
    ("FPN", "Feature Pyramid Network"),
    ("FPS", "Frames Per Second"),
    ("IoU", "Intersection over Union"),
    ("mAP", "mean Average Precision"),
    ("NMS", "Non-Maximum Suppression"),
    ("RT-DETR", "Real-Time Detection Transformer"),
    ("YOLO", "You Only Look Once"),
]

# --------------------------------------------------------------------------- #

CHAPTER_1 = {
    "title": "CHƯƠNG 1. GIỚI THIỆU CHUNG",
    "body": [
        ("h2", "1.1. Lý do chọn đề tài"),
        ("b", "Hệ thống giám sát giao thông thông minh dựa trên thị giác máy tính đang "
              "được triển khai rộng rãi nhằm giám sát mật độ, phát hiện vi phạm và hỗ trợ "
              "điều phối giao thông. Phát hiện người và phương tiện là bài toán nền tảng "
              "của các hệ thống này."),
        ("b", "Trong thực tế, camera giám sát phải hoạt động ở mọi điều kiện thời tiết. "
              "Sương mù làm giảm độ tương phản; mưa gây nhiễu và che khuất; bão cát và "
              "tuyết làm biến đổi màu sắc và che lấp đối tượng. Những yếu tố này khiến mô "
              "hình được huấn luyện trên dữ liệu thời tiết tốt suy giảm đáng kể. Do đó, "
              "nghiên cứu mô hình phát hiện bền vững với thời tiết bất lợi có ý nghĩa thực "
              "tiễn cao."),
        ("h2", "1.2. Mục tiêu nghiên cứu"),
        ("b", "Khóa luận hướng tới: (1) xây dựng quy trình huấn luyện mô hình phát hiện "
              "người và phương tiện thích nghi với thời tiết bất lợi; (2) so sánh có hệ "
              "thống ba hướng tiếp cận tiêu biểu gồm mô hình một giai đoạn (YOLO), mô hình "
              "hai giai đoạn (Faster R-CNN) và mô hình dựa trên transformer (RT-DETR); "
              "(3) khảo sát tác động của cơ chế chú ý CBAM đối với hiệu năng phát hiện."),
        ("h2", "1.3. Đối tượng và phạm vi nghiên cứu"),
        ("b", "Đối tượng nghiên cứu là sáu lớp đối tượng giao thông: người (person), xe "
              "đạp (bicycle), ô tô (car), xe máy (motorcycle), xe buýt (bus) và xe tải "
              "(truck)."),
        ("b", "Phạm vi nghiên cứu sử dụng hai bộ dữ liệu công khai: BDD100K cho giai đoạn "
              "thích nghi miền lái xe và DAWN cho giai đoạn thời tiết bất lợi. Toàn bộ "
              "thực nghiệm thực hiện trên nền tảng Google Colab với GPU NVIDIA Tesla T4."),
        ("h2", "1.4. Phương pháp nghiên cứu"),
        ("b", "Khóa luận áp dụng chiến lược học chuyển giao lũy tiến: khởi tạo từ trọng số "
              "huấn luyện trên COCO, tinh chỉnh trên BDD100K để học miền lái xe, sau đó "
              "tinh chỉnh tiếp trên DAWN để thích nghi thời tiết bất lợi. Mô hình được "
              "đánh giá bằng Precision, Recall, mAP50, mAP50-95, tốc độ suy luận (FPS) và "
              "số tham số, kèm phân tích theo từng lớp và từng điều kiện thời tiết."),
        ("h2", "1.5. Bố cục khóa luận"),
        ("b", "Khóa luận gồm năm chương. Chương 1 giới thiệu chung. Chương 2 trình bày cơ "
              "sở lý luận và các nghiên cứu liên quan. Chương 3 mô tả mô hình lý thuyết và "
              "giải pháp đề xuất. Chương 4 trình bày thực nghiệm, kết quả và phân tích. "
              "Chương 5 kết luận và đề xuất hướng phát triển."),
    ],
}

CHAPTER_2 = {
    "title": "CHƯƠNG 2. CƠ SỞ LÝ LUẬN",
    "body": [
        ("h2", "2.1. Tổng quan bài toán phát hiện đối tượng"),
        ("b", "Phát hiện đối tượng (object detection) là bài toán xác định vị trí (hộp bao "
              "- bounding box) và phân loại các đối tượng trong ảnh. Khác với phân loại ảnh "
              "chỉ trả về một nhãn, phát hiện đối tượng phải đồng thời định vị và phân loại "
              "nhiều đối tượng. Các độ đo phổ biến gồm Precision, Recall, và đặc biệt là "
              "mAP (mean Average Precision) tính trên các ngưỡng IoU."),
        ("h2", "2.2. Các hướng tiếp cận phát hiện đối tượng"),
        ("h3", "2.2.1. Mô hình hai giai đoạn (two-stage)"),
        ("b", "Tiêu biểu là họ R-CNN và Faster R-CNN. Mô hình trước tiên sinh các vùng đề "
              "xuất (region proposals) bằng mạng RPN, sau đó phân loại và tinh chỉnh hộp "
              "bao cho từng vùng. Hướng này thường cho độ chính xác cao nhưng tốc độ chậm "
              "và đòi hỏi nhiều dữ liệu."),
        ("h3", "2.2.2. Mô hình một giai đoạn (one-stage)"),
        ("b", "Tiêu biểu là họ YOLO và SSD. Mô hình dự đoán trực tiếp lớp và hộp bao trên "
              "lưới đặc trưng trong một lần suy luận, không qua bước sinh vùng đề xuất. "
              "Hướng này nhanh, phù hợp ứng dụng thời gian thực, là lựa chọn phổ biến cho "
              "giám sát giao thông."),
        ("h3", "2.2.3. Mô hình dựa trên transformer"),
        ("b", "RT-DETR là mô hình phát hiện thời gian thực dựa trên kiến trúc DETR, loại bỏ "
              "bước hậu xử lý NMS bằng cơ chế so khớp tập hợp (set prediction). Đây là "
              "hướng hiện đại, cân bằng giữa độ chính xác và tốc độ."),
        ("h2", "2.3. Các kỹ thuật nền tảng"),
        ("b", "Mạng nơ-ron tích chập (CNN) là xương sống trích xuất đặc trưng. Mạng kim tự "
              "tháp đặc trưng (FPN) kết hợp đặc trưng đa tỉ lệ giúp phát hiện đối tượng ở "
              "nhiều kích thước. Học chuyển giao (transfer learning) tận dụng trọng số đã "
              "huấn luyện trên tập lớn để cải thiện hiệu quả trên tập nhỏ."),
        ("h3", "2.3.1. Cơ chế chú ý CBAM"),
        ("b", "CBAM (Convolutional Block Attention Module) là mô-đun chú ý nhẹ, lần lượt "
              "áp dụng chú ý theo kênh (channel attention) và theo không gian (spatial "
              "attention) để mô hình tập trung vào đặc trưng quan trọng. CBAM được kỳ vọng "
              "giúp mô hình bền vững hơn với nhiễu do thời tiết."),
        ("h2", "2.4. Các bộ dữ liệu"),
        ("b", "BDD100K là bộ dữ liệu lái xe quy mô lớn với nhiều điều kiện đường phố, dùng "
              "cho giai đoạn thích nghi miền. DAWN là bộ dữ liệu chuyên về thời tiết bất "
              "lợi gồm bốn điều kiện: sương mù (fog), mưa (rain), bão cát (sand) và tuyết "
              "(snow), dùng cho giai đoạn tinh chỉnh chính."),
        ("h2", "2.5. Vấn đề đặt ra và giải pháp đề xuất"),
        ("b", "Việc huấn luyện trực tiếp từ COCO sang DAWN buộc mô hình học đồng thời miền "
              "lái xe và đặc trưng thời tiết từ một tập DAWN nhỏ, dễ dẫn tới quá khớp. "
              "Khóa luận đề xuất chiến lược tinh chỉnh lũy tiến hai giai đoạn (COCO → "
              "BDD100K → DAWN) để tách biệt việc học miền và việc thích nghi thời tiết, "
              "kết hợp khảo sát cơ chế chú ý CBAM."),
    ],
}

CHAPTER_3 = {
    "title": "CHƯƠNG 3. MÔ HÌNH LÝ THUYẾT",
    "body": [
        ("h2", "3.1. Kiến trúc tổng thể của giải pháp"),
        ("b", "Giải pháp gồm hai giai đoạn huấn luyện nối tiếp. Giai đoạn 1 tinh chỉnh mô "
              "hình (khởi tạo từ COCO) trên BDD100K để học các đối tượng miền lái xe. Giai "
              "đoạn 2 tiếp tục tinh chỉnh trên DAWN để thích nghi điều kiện thời tiết bất "
              "lợi. Cùng quy trình này được áp dụng cho các mô hình so sánh."),
        ("h2", "3.2. Các mô hình được khảo sát"),
        ("b", "Khóa luận khảo sát các mô hình đại diện ba hướng: YOLOv8n và YOLO11n (một "
              "giai đoạn), Faster R-CNN (hai giai đoạn) và RT-DETR (transformer). Các mô "
              "hình nhẹ được ưu tiên do yêu cầu suy luận thời gian thực của bài toán giám "
              "sát giao thông."),
        ("h2", "3.3. Mô-đun chú ý CBAM"),
        ("b", "CBAM gồm hai thành phần nối tiếp. Chú ý theo kênh tổng hợp thông tin toàn "
              "cục bằng gộp trung bình và gộp cực đại, qua một mạng MLP dùng chung để tạo "
              "trọng số cho từng kênh. Chú ý theo không gian ghép bản đồ trung bình và cực "
              "đại theo kênh rồi qua một tích chập để tạo trọng số cho từng vị trí. Hai "
              "trọng số này nhân lần lượt vào đặc trưng đầu vào, giữ nguyên kích thước "
              "tensor nên có thể chèn vào phần cổ (neck) của mạng phát hiện."),
        ("h2", "3.4. Quy trình tiền xử lý dữ liệu"),
        ("b", "Ảnh được đưa về kích thước 640x640 bằng kỹ thuật letterbox giữ nguyên tỉ lệ "
              "và đệm viền. Nhãn được chuẩn hóa về sáu lớp mục tiêu theo định dạng YOLO "
              "(class, tâm x, tâm y, rộng, cao - chuẩn hóa về [0,1]). Riêng DAWN được chia "
              "tập theo phân tầng điều kiện thời tiết để mỗi tập con cân bằng về phân bố "
              "thời tiết."),
        ("h2", "3.5. Thiết kế thí nghiệm"),
        ("b", "Thí nghiệm gồm hai trục. Trục so sánh (benchmark) đánh giá các mô hình trên "
              "cùng quy trình và cùng bộ dữ liệu. Trục khảo sát (ablation) cố định kiến "
              "trúc nền YOLOv8n và lần lượt thay đổi một yếu tố (CBAM, tăng cường dữ liệu "
              "thời tiết) để cô lập tác động. Mọi cấu hình giữ cố định kích thước ảnh, bộ "
              "tối ưu, số epoch và giao thức đánh giá để bảo đảm tính công bằng."),
    ],
}

CHAPTER_4 = {
    "title": "CHƯƠNG 4. MÔ HÌNH THỰC NGHIỆM",
    "body": [
        ("h2", "4.1. Môi trường và công cụ"),
        ("b", "Thực nghiệm chạy trên Google Colab với GPU NVIDIA Tesla T4 (15 GB), sử dụng "
              "PyTorch 2.x, thư viện Ultralytics cho YOLO/RT-DETR và TorchVision cho Faster "
              "R-CNN. Cấu hình huấn luyện chung: ảnh 640x640, bộ tối ưu AdamW, lịch học cos, "
              "huấn luyện hỗn hợp độ chính xác (AMP), seed 42."),
        ("h2", "4.2. Chuẩn bị dữ liệu và cấu hình huấn luyện"),
        ("b", "Dữ liệu BDD100K và DAWN được chuyển về định dạng YOLO sáu lớp. Tập DAWN dùng "
              "để đánh giá gồm 206 ảnh kiểm định với 1806 đối tượng, phân bố không cân bằng: "
              "ô tô chiếm đa số (1522 đối tượng), trong khi xe đạp chỉ có 6, xe máy 23 và "
              "xe buýt 43 đối tượng."),
        ("h2", "4.3. Kết quả so sánh các mô hình"),
        ("b", "Trên tập kiểm định DAWN, YOLOv8n đạt Precision 0,662; Recall 0,579; mAP50 "
              "0,639 và mAP50-95 0,426 với khoảng 3,0 triệu tham số. YOLO11n đạt Precision "
              "0,784; Recall 0,504; mAP50 0,602 và mAP50-95 0,369 với 2,6 triệu tham số và "
              "tốc độ 67,4 FPS (batch-1)."),
        ("b", "Đáng chú ý, YOLO11n tuy mới hơn và ít tham số hơn nhưng cho mAP50-95 thấp "
              "hơn YOLOv8n (0,369 so với 0,426) và Recall thấp hơn (0,504 so với 0,579), "
              "dù Precision cao hơn. Điều này cho thấy kiến trúc mới hơn không đảm bảo tốt "
              "hơn trên tập dữ liệu nhỏ và đặc thù như thời tiết bất lợi. (Kết quả Faster "
              "R-CNN và RT-DETR sẽ được bổ sung sau khi hoàn tất huấn luyện.)"),
        ("h2", "4.4. Phân tích theo từng lớp"),
        ("b", "Với YOLOv8n, lớp ô tô đạt mAP50-95 cao nhất (0,588) nhờ nhiều dữ liệu; các "
              "lớp xe buýt (0,212) và xe tải (0,354) yếu nhất. Lớp xe đạp tuy có chỉ số "
              "cao nhưng chỉ dựa trên 6 mẫu nên không đáng tin cậy. Kết quả cho thấy mất "
              "cân bằng lớp ảnh hưởng rõ rệt và cần được lưu ý khi diễn giải."),
        ("h2", "4.5. Phân tích theo điều kiện thời tiết"),
        ("b", "Với YOLOv8n, điều kiện sương mù cho kết quả tốt nhất (mAP50-95 0,490), kế "
              "đến là tuyết (0,457), bão cát (0,397) và thấp nhất là mưa (0,382). Như vậy "
              "mưa và bão cát là hai điều kiện khó nhất, là cơ sở để đánh giá hiệu quả của "
              "các cải tiến (CBAM, tăng cường dữ liệu) trong các bước tiếp theo."),
        ("h2", "4.6. Kết quả khảo sát CBAM"),
        ("b", "(Phần này sẽ trình bày so sánh giữa YOLOv8n nền và YOLOv8n + CBAM, báo cáo "
              "trung bình và độ lệch chuẩn qua nhiều seed, sau khi hoàn tất huấn luyện "
              "khảo sát.)"),
        ("h2", "4.7. Nhận xét và thảo luận"),
        ("b", "Kết quả ban đầu cho thấy mô hình một giai đoạn nhẹ (YOLOv8n) đạt cân bằng "
              "tốt giữa độ chính xác và tốc độ trên tập thời tiết bất lợi nhỏ. Việc đánh "
              "giá theo từng lớp và từng điều kiện thời tiết cung cấp góc nhìn sâu hơn so "
              "với chỉ số mAP tổng, đặc biệt quan trọng cho bài toán giám sát giao thông."),
    ],
}

CHAPTER_5 = {
    "title": "CHƯƠNG 5. KẾT LUẬN VÀ KIẾN NGHỊ",
    "body": [
        ("h2", "5.1. Kết luận"),
        ("b", "Khóa luận đã xây dựng quy trình huấn luyện lũy tiến hai giai đoạn (COCO → "
              "BDD100K → DAWN) cho bài toán phát hiện người và phương tiện trong điều kiện "
              "thời tiết bất lợi, đồng thời thiết lập khung so sánh có hệ thống giữa các "
              "hướng tiếp cận và công cụ đánh giá theo lớp, theo thời tiết."),
        ("h2", "5.2. Đóng góp"),
        ("b", "Đóng góp chính gồm: quy trình huấn luyện và đánh giá tái lập được; bộ công "
              "cụ đo lường chi tiết theo lớp và theo điều kiện thời tiết; và phân tích so "
              "sánh các mô hình đại diện một giai đoạn, hai giai đoạn và transformer."),
        ("h2", "5.3. Hạn chế"),
        ("b", "Bộ dữ liệu DAWN tương đối nhỏ và mất cân bằng lớp, khiến một số chỉ số theo "
              "lớp chưa đáng tin cậy và kết quả từ một seed có thể dao động. Một số mô hình "
              "nặng cần nhiều dữ liệu hơn để phát huy."),
        ("h2", "5.4. Hướng phát triển"),
        ("b", "Các hướng tiếp theo gồm: bổ sung tăng cường dữ liệu mô phỏng thời tiết và "
              "tiền xử lý khử nhiễu (dehazing); đánh giá đa seed để báo cáo trung bình và "
              "độ lệch chuẩn; mở rộng bộ dữ liệu; và tối ưu mô hình cho triển khai trên "
              "thiết bị biên phục vụ giám sát giao thông thời gian thực."),
    ],
}

CHAPTERS = [CHAPTER_1, CHAPTER_2, CHAPTER_3, CHAPTER_4, CHAPTER_5]
