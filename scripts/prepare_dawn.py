#!/usr/bin/env python3
"""Convert DAWN Pascal VOC annotations to a stratified YOLO dataset.

Mục đích của file này:
1. Đọc tập DAWN thô, gồm ảnh và file nhãn Pascal VOC XML.
2. Tìm đúng từng cặp ảnh/XML.
3. Chuẩn hóa tên lớp về bộ lớp nghiên cứu.
4. Resize ảnh theo kiểu letterbox để giữ tỉ lệ ảnh gốc.
5. Chuyển bbox từ Pascal VOC `xmin ymin xmax ymax` sang YOLO
   `class_id x_center y_center width height`, tất cả đã được chuẩn hóa về [0, 1].
6. Chia dữ liệu thành train/val/test theo tỉ lệ 70/15/15, có phân tầng theo thời
   tiết để mỗi split có phân phối thời tiết gần nhau hơn.
7. Tạo cấu trúc thư mục và file `dataset.yaml` để Ultralytics YOLO có thể train.
"""

# Cho phép dùng cú pháp type hint mới như `list[Sample]` ổn định hơn giữa các
# phiên bản Python.
from __future__ import annotations

# argparse: đọc tham số dòng lệnh, ví dụ --raw-dir, --output-dir.
import argparse
# csv: ghi manifest.csv để kiểm tra ảnh nào được đưa vào split nào.
import csv
# hashlib: tạo mã hash ngắn từ đường dẫn ảnh, tránh trùng tên file khi copy.
import hashlib
# random: trộn dữ liệu theo seed cố định để chia train/val/test tái lập được.
import random
# shutil: xóa thư mục output cũ khi người dùng truyền --clean.
import shutil
# ElementTree: parser XML Pascal VOC.
import xml.etree.ElementTree as ET
# Counter đếm lớp bị bỏ qua; defaultdict gom ảnh theo weather hoặc XML theo stem.
from collections import Counter, defaultdict
# dataclass giúp định nghĩa object Sample ngắn gọn, rõ ràng.
from dataclasses import dataclass
# Path xử lý đường dẫn an toàn hơn nối chuỗi thủ công.
from pathlib import Path

# cv2 đọc, resize và ghi ảnh.
import cv2
# yaml ghi file dataset.yaml cho Ultralytics.
import yaml

# Các định dạng ảnh được script nhận diện khi quét thư mục raw.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Danh sách lớp cuối cùng của bài toán. Thứ tự ở đây chính là class_id trong YOLO.
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# DAWN hoặc các bản annotation khác nhau có thể ghi tên lớp không thống nhất.
# ALIASES chuẩn hóa các tên đó về đúng tên trong CLASSES.
ALIASES = {
    "pedestrian": "person",
    "people": "person",
    "person": "person",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motorbike": "motorcycle",
    "motor cycle": "motorcycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}


@dataclass(frozen=True)
class Sample:
    """Một mẫu dữ liệu hợp lệ gồm ảnh, file XML và nhãn thời tiết."""

    # Đường dẫn tới file ảnh gốc.
    image: Path
    # Đường dẫn tới file Pascal VOC XML tương ứng.
    annotation: Path
    # Nhóm thời tiết suy ra từ đường dẫn/tên file: fog, rain, sand, snow, unknown.
    weather: str


def infer_weather(path: Path) -> str:
    """Suy ra điều kiện thời tiết từ các thành phần trong đường dẫn ảnh."""

    # Chuyển toàn bộ folder/file name về chữ thường để tìm không phân biệt hoa/thường.
    text_parts = [part.lower() for part in path.parts]

    # Nếu bất kỳ phần nào của đường dẫn chứa từ khóa weather thì trả về weather đó.
    for weather in ("fog", "rain", "sand", "snow"):
        if any(weather in part for part in text_parts):
            return weather

    # Nếu không tìm được, vẫn giữ ảnh nhưng đưa vào nhóm unknown.
    return "unknown"


def parse_args() -> argparse.Namespace:
    """Khai báo và đọc các tham số dòng lệnh."""

    parser = argparse.ArgumentParser()

    # Thư mục chứa dữ liệu DAWN thô sau khi giải nén.
    parser.add_argument("--raw-dir", type=Path, required=True)

    # Thư mục output theo format YOLO: images/train, labels/train, ...
    parser.add_argument("--output-dir", type=Path, required=True)

    # Kích thước ảnh vuông sau letterbox. YOLOv8 thường dùng 640.
    parser.add_argument("--imgsz", type=int, default=640)

    # Seed cố định để mỗi lần chia dữ liệu cho cùng kết quả.
    parser.add_argument("--seed", type=int, default=42)

    # Nếu bật --clean, xóa output cũ trước khi tạo lại để tránh lẫn split cũ.
    parser.add_argument("--clean", action="store_true", help="Delete output first.")
    return parser.parse_args()


def find_samples(raw_dir: Path) -> list[Sample]:
    """Tìm tất cả cặp ảnh/XML hợp lệ trong thư mục raw."""

    # Tạo index: tên file không đuôi -> danh sách XML có cùng stem.
    # Ví dụ rain_001.xml có stem là rain_001.
    xml_by_stem: dict[str, list[Path]] = defaultdict(list)
    for xml_path in raw_dir.rglob("*.xml"):
        xml_by_stem[xml_path.stem].append(xml_path)

    samples: list[Sample] = []

    # Quét đệ quy toàn bộ ảnh trong raw_dir, sau đó tìm XML tương ứng.
    for image in sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        # Ưu tiên XML nằm cùng thư mục và cùng tên với ảnh.
        # Ví dụ Fog/0001.jpg -> Fog/0001.xml.
        same_dir = image.with_suffix(".xml")

        # Nếu XML cùng thư mục không tồn tại, tìm XML có cùng stem ở nơi khác.
        # Cách này hỗ trợ dataset tách riêng thư mục Images và Annotations.
        candidates = [same_dir] if same_dir.exists() else xml_by_stem.get(image.stem, [])

        # Nếu không có XML hoặc có nhiều XML cùng stem, bỏ qua để tránh gán nhãn sai.
        if len(candidates) != 1:
            print(f"SKIP: expected one XML for {image}, found {len(candidates)}")
            continue

        # relative dùng để suy luận weather và tạo hash ổn định theo vị trí ảnh.
        relative = image.relative_to(raw_dir)
        weather = infer_weather(relative)
        samples.append(Sample(image, candidates[0], weather))

    # Nếu không tìm được cặp nào thì dừng sớm, vì train YOLO sẽ không có dữ liệu.
    if not samples:
        raise RuntimeError(f"No image/XML pairs found under {raw_dir}")
    return samples


def stratified_split(samples: list[Sample], seed: int) -> dict[Path, str]:
    """Split each weather group independently to approximate 70/15/15."""

    # Dùng object Random riêng để việc shuffle không ảnh hưởng random global.
    rng = random.Random(seed)

    # Gom mẫu theo weather. Mục tiêu: fog/rain/snow/sand đều có mặt tương đối đều
    # trong train, val và test.
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.weather].append(sample)

    # assignment ánh xạ từ đường dẫn ảnh gốc sang split: train, val hoặc test.
    assignment: dict[Path, str] = {}
    for weather, group in sorted(groups.items()):
        # Shuffle từng nhóm weather bằng seed cố định.
        rng.shuffle(group)

        # Tính số lượng ảnh cho train và val. Phần còn lại tự động là test.
        n = len(group)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        # Gán split theo thứ tự sau khi shuffle.
        for index, sample in enumerate(group):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            assignment[sample.image] = split

        # In thống kê để người dùng biết split mỗi weather có bao nhiêu ảnh.
        print(f"{weather}: total={n}, train={n_train}, val={n_val}, test={n-n_train-n_val}")
    return assignment


def parse_voc(xml_path: Path, width: int, height: int) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    """Đọc một file Pascal VOC XML và trả về bbox dạng absolute xyxy."""

    # Mỗi box có dạng: (class_id, xmin, ymin, xmax, ymax).
    boxes: list[tuple[int, float, float, float, float]] = []

    # unknown dùng để thống kê các class không nằm trong bộ lớp nghiên cứu.
    unknown: Counter = Counter()

    # Parse XML và lấy node gốc <annotation>.
    root = ET.parse(xml_path).getroot()

    # Pascal VOC lưu mỗi đối tượng trong một thẻ <object>.
    for obj in root.findall("object"):
        # Lấy tên lớp, bỏ khoảng trắng, chuyển chữ thường để chuẩn hóa.
        raw_name = (obj.findtext("name") or "").strip().lower()

        # Ánh xạ tên gốc về tên chuẩn. Ví dụ motorbike -> motorcycle.
        name = ALIASES.get(raw_name)
        if name is None:
            # Nếu class không thuộc phạm vi nghiên cứu thì bỏ qua bbox này.
            unknown[raw_name or "<empty>"] += 1
            continue

        # bndbox chứa bốn tọa độ Pascal VOC: xmin, ymin, xmax, ymax.
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        # Đọc tọa độ và clamp vào biên ảnh để tránh lỗi annotation vượt biên.
        xmin = max(0.0, min(float(bbox.findtext("xmin", "0")), width - 1))
        ymin = max(0.0, min(float(bbox.findtext("ymin", "0")), height - 1))
        xmax = max(0.0, min(float(bbox.findtext("xmax", "0")), width))
        ymax = max(0.0, min(float(bbox.findtext("ymax", "0")), height))

        # Bỏ bbox lỗi nếu chiều rộng hoặc chiều cao không dương.
        if xmax <= xmin or ymax <= ymin:
            continue

        # Chuyển class name sang class_id theo vị trí trong CLASSES.
        boxes.append((CLASSES.index(name), xmin, ymin, xmax, ymax))
    return boxes, unknown


def letterbox(image, boxes, size: int):
    """Resize with unchanged aspect ratio and transform absolute xyxy boxes."""

    # Lấy kích thước ảnh gốc theo thứ tự OpenCV: height, width.
    height, width = image.shape[:2]

    # scale là tỉ lệ resize nhỏ nhất để ảnh vừa trong khung vuông size x size.
    # Dùng min để không làm méo ảnh.
    scale = min(size / width, size / height)

    # Kích thước mới sau khi resize giữ nguyên aspect ratio.
    new_width, new_height = round(width * scale), round(height * scale)

    # Resize ảnh về kích thước mới.
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Tính padding trái và trên để đặt ảnh vào giữa canvas vuông.
    left = (size - new_width) // 2
    top = (size - new_height) // 2

    # Tạo ảnh vuông bằng cách thêm viền màu xám 114, giống convention của YOLO.
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        size - new_height - top,
        left,
        size - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # Cập nhật bbox theo cùng phép biến đổi với ảnh:
    # tọa độ mới = tọa độ cũ * scale + padding.
    transformed = []
    for class_id, xmin, ymin, xmax, ymax in boxes:
        transformed.append(
            (
                class_id,
                xmin * scale + left,
                ymin * scale + top,
                xmax * scale + left,
                ymax * scale + top,
            )
        )
    return canvas, transformed


def write_yolo_label(path: Path, boxes, size: int) -> None:
    """Ghi file .txt theo định dạng YOLO normalized."""

    # Mỗi dòng YOLO: class_id x_center y_center width height.
    # Bốn giá trị tọa độ được chia cho size để nằm trong khoảng [0, 1].
    lines = []
    for class_id, xmin, ymin, xmax, ymax in boxes:
        x_center = ((xmin + xmax) / 2) / size
        y_center = ((ymin + ymax) / 2) / size
        width = (xmax - xmin) / size
        height = (ymax - ymin) / size
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Nếu ảnh không có object hợp lệ, file label vẫn được tạo rỗng.
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    """Điều phối toàn bộ pipeline chuẩn bị dữ liệu."""

    # Đọc tham số dòng lệnh.
    args = parse_args()

    # Chuyển raw_dir và output_dir thành đường dẫn tuyệt đối để ghi dataset.yaml rõ ràng.
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Nếu người dùng yêu cầu --clean thì xóa output cũ trước khi tạo lại.
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Nếu không dùng --clean mà output đã có ảnh, dừng lại để tránh ảnh cũ còn sót
    # trong split cũ, gây rò rỉ train/val/test.
    elif any(output_dir.glob("images/*/*")):
        raise RuntimeError(
            f"{output_dir} already contains processed images. Re-run with --clean "
            "to prevent stale files and train/val/test leakage."
        )

    # Tạo sáu thư mục chuẩn YOLO: images/train, images/val, images/test,
    # labels/train, labels/val, labels/test.
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Tìm toàn bộ cặp ảnh/XML và chia split theo weather.
    samples = find_samples(raw_dir)
    assignments = stratified_split(samples, args.seed)

    # unknown_classes gom các class bị bỏ qua để người dùng biết annotation có gì.
    unknown_classes: Counter = Counter()

    # manifest_rows lưu thông tin từng ảnh để kiểm tra và audit split sau này.
    manifest_rows = []

    # Xử lý từng ảnh: đọc ảnh, đọc XML, letterbox, ghi ảnh và label mới.
    for sample in samples:
        # cv2.imread trả về ảnh dạng BGR hoặc None nếu file lỗi.
        image = cv2.imread(str(sample.image))
        if image is None:
            print(f"SKIP: cannot read {sample.image}")
            continue

        # Dùng kích thước ảnh gốc để clamp bbox Pascal VOC.
        height, width = image.shape[:2]

        # Đọc bbox từ XML, đồng thời nhận danh sách class ngoài phạm vi nghiên cứu.
        boxes, unknown = parse_voc(sample.annotation, width, height)
        unknown_classes.update(unknown)

        # Resize ảnh bằng letterbox và biến đổi bbox tương ứng.
        image, boxes = letterbox(image, boxes, args.imgsz)

        # Lấy split đã gán trước đó để ghi vào đúng thư mục.
        split = assignments[sample.image]

        # Tạo hash ngắn từ đường dẫn tương đối để tránh hai ảnh khác thư mục nhưng
        # cùng tên file ghi đè nhau trong output.
        digest = hashlib.sha1(str(sample.image.relative_to(raw_dir)).encode()).hexdigest()[:10]

        # Tên file output có weather + stem gốc + hash để dễ truy vết.
        stem = f"{sample.weather}_{sample.image.stem}_{digest}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"

        # Ghi ảnh chuẩn hóa thành JPG chất lượng 95.
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Ghi nhãn YOLO tương ứng.
        write_yolo_label(label_path, boxes, args.imgsz)

        # Thêm một dòng manifest để biết ảnh gốc nào tạo ra ảnh output nào.
        manifest_rows.append([split, sample.weather, str(sample.image), str(image_path), len(boxes)])

    # Ghi manifest.csv để phục vụ kiểm tra dữ liệu và mô tả khóa luận.
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "weather", "source_image", "output_image", "objects"])
        writer.writerows(manifest_rows)

    # dataset.yaml là file Ultralytics YOLO cần để biết đường dẫn train/val/test
    # và tên lớp tương ứng với class_id.
    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASSES)},
    }

    # sort_keys=False giữ thứ tự key dễ đọc: path, train, val, test, names.
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dataset_yaml, stream, sort_keys=False)

    # In thống kê cuối cùng cho người dùng kiểm tra nhanh.
    counts = Counter(row[0] for row in manifest_rows)
    print(f"Created {dict(counts)} at {output_dir}")

    # Nếu có lớp bị bỏ qua, in ra để người dùng cân nhắc bổ sung ALIASES/CLASSES.
    if unknown_classes:
        print(f"Ignored unknown classes: {dict(unknown_classes)}")


# Chỉ chạy main() khi file được gọi trực tiếp:
# python scripts/prepare_dawn.py ...
# Nếu import file này từ test/script khác thì main() không tự chạy.
if __name__ == "__main__":
    main()
