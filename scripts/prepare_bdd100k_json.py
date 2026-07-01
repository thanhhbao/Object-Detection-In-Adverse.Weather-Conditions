#!/usr/bin/env python3
"""Convert BDD100K official JSON labels to YOLO 6-class format.

BDD100K cung cấp label dưới dạng JSON (bdd100k_labels_images_train.json).
Script này:
  1. Đọc JSON label train + val
  2. Remap category → 6 class (person/bicycle/car/motorcycle/bus/truck)
  3. Chọn subset train theo --subset-mode
  4. Symlink ảnh (mặc định) hoặc copy (--copy-images)
  5. Ghi YOLO label (normalized bằng kích thước gốc — không resize)
  6. Ghi manifest.csv, stats.json, dataset.yaml

BDD100K image size: 1280×720 (tất cả ảnh đều là kích thước này)

Ví dụ:
  python scripts/prepare_bdd100k_json.py \\
    --zip-path /content/drive/MyDrive/adverse_weather_project/datasets/BDD100K.zip \\
    --output-dir /content/bdd100k_6cls_30k_yolo \\
    --subset-size 30000 \\
    --subset-mode condition_aware \\
    --seed 42 --clean

Hoặc nếu đã giải nén:
  python scripts/prepare_bdd100k_json.py \\
    --raw-dir /content/bdd100k_raw \\
    --output-dir /content/bdd100k_6cls_30k_yolo \\
    --subset-size 30000 --subset-mode condition_aware --seed 42 --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

BDD_TO_PROJECT: dict[str, str] = {
    "person":    "person",
    "rider":     "person",
    "bike":      "bicycle",
    "bicycle":   "bicycle",
    "car":       "car",
    "motor":     "motorcycle",
    "motorcycle": "motorcycle",
    "bus":       "bus",
    "truck":     "truck",
    # ignored: traffic light, traffic sign, train, other person, ...
}

# ---------------------------------------------------------------------------
# Condition-aware subset strategy
# Priority: foggy > snowy > rainy > dawn_dusk > night > other
# Each image is assigned to exactly one (highest-priority) condition pool.
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["foggy", "snowy", "rainy", "dawn_dusk", "night", "other"]

# None = take all available
CONDITION_CAPS: dict[str, int | None] = {
    "foggy":     None,   # ~130 images — take all
    "snowy":     5000,
    "rainy":     5000,
    "dawn_dusk": 5000,
    "night":     6000,
    "other":     None,   # filled to reach subset_size
}


def get_condition(weather: str, timeofday: str) -> str:
    w = weather.lower().strip()
    t = timeofday.lower().strip()
    if w == "foggy":
        return "foggy"
    if w == "snowy":
        return "snowy"
    if w == "rainy":
        return "rainy"
    if t in ("dawn/dusk",):
        return "dawn_dusk"
    if t == "night":
        return "night"
    return "other"


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path.name} → {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Find BDD paths in raw dir
# ---------------------------------------------------------------------------


def find_bdd_paths(raw_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return (img_train_dir, img_val_dir, lbl_train_json, lbl_val_json)."""
    img_train = raw_dir / "bdd100k" / "bdd100k" / "images" / "100k" / "train"
    img_val   = raw_dir / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    lbl_train = (
        raw_dir / "bdd100k_labels_release" / "bdd100k" / "labels"
        / "bdd100k_labels_images_train.json"
    )
    lbl_val = (
        raw_dir / "bdd100k_labels_release" / "bdd100k" / "labels"
        / "bdd100k_labels_images_val.json"
    )
    for p in [img_train, img_val, lbl_train, lbl_val]:
        if not p.exists():
            raise FileNotFoundError(
                f"Path not found: {p}\n"
                "Kiểm tra cấu trúc thư mục BDD100K sau khi giải nén."
            )
    return img_train, img_val, lbl_train, lbl_val


# ---------------------------------------------------------------------------
# Parse BDD frame → (name, weather, timeofday, scene, boxes)
# ---------------------------------------------------------------------------


def parse_frame(frame: dict) -> tuple[str, str, str, str, list[tuple]]:
    name = frame["name"]
    attrs = frame.get("attributes", {})
    weather   = str(attrs.get("weather", "undefined")).strip()
    timeofday = str(attrs.get("timeofday", "undefined")).strip()
    scene     = str(attrs.get("scene", "undefined")).strip()

    boxes: list[tuple] = []
    ignored_cats: list[str] = []
    for label in frame.get("labels", []):
        cat = str(label.get("category", "")).lower().strip()
        proj = BDD_TO_PROJECT.get(cat)
        if proj is None:
            ignored_cats.append(cat)
            continue
        box2d = label.get("box2d")
        if not box2d:
            continue
        x1, y1 = float(box2d["x1"]), float(box2d["y1"])
        x2, y2 = float(box2d["x2"]), float(box2d["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((CLASS_TO_ID[proj], x1, y1, x2, y2))

    return name, weather, timeofday, scene, boxes


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def condition_aware_subset(frames: list[dict], subset_size: int, seed: int) -> list[dict]:
    """Priority-based sampling: rare conditions first, fill with 'other'."""
    rng = random.Random(seed)

    pools: dict[str, list[dict]] = {c: [] for c in CONDITION_ORDER}
    for frame in frames:
        attrs = frame.get("attributes", {})
        cond = get_condition(
            attrs.get("weather", "undefined"),
            attrs.get("timeofday", "undefined"),
        )
        pools[cond].append(frame)

    print(f"\n{'Condition':<12} {'Available':>10} {'Cap':>8} {'Selected':>10}")
    print("-" * 44)

    selected: list[dict] = []
    for cond in CONDITION_ORDER[:-1]:  # all except "other"
        pool = pools[cond]
        rng.shuffle(pool)
        cap = CONDITION_CAPS[cond]
        take = pool if cap is None else pool[:cap]
        selected.extend(take)
        cap_str = str(cap) if cap is not None else "all"
        print(f"{cond:<12} {len(pool):>10} {cap_str:>8} {len(take):>10}")

    remaining = subset_size - len(selected)
    other_pool = pools["other"]
    rng.shuffle(other_pool)
    take_other = other_pool[:max(0, remaining)]
    selected.extend(take_other)
    print(f"{'other':<12} {len(other_pool):>10} {'fill':>8} {len(take_other):>10}")
    print(f"\nTotal selected: {len(selected)}/{subset_size} requested\n")

    rng.shuffle(selected)
    return selected


def random_subset(frames: list[dict], subset_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(frames)
    rng.shuffle(pool)
    return pool[:subset_size]


# ---------------------------------------------------------------------------
# Write YOLO label (normalized by original image size — no resize)
# ---------------------------------------------------------------------------

BDD_IMG_W = 1280
BDD_IMG_H = 720


def write_yolo_label(
    label_path: Path,
    boxes: list[tuple],
    img_w: int = BDD_IMG_W,
    img_h: int = BDD_IMG_H,
) -> None:
    lines: list[str] = []
    for cls_id, x1, y1, x2, y2 in boxes:
        x1c = max(0.0, min(x1, img_w))
        y1c = max(0.0, min(y1, img_h))
        x2c = max(0.0, min(x2, img_w))
        y2c = max(0.0, min(y2, img_h))
        if x2c <= x1c or y2c <= y1c:
            continue
        cx = ((x1c + x2c) / 2) / img_w
        cy = ((y1c + y2c) / 2) / img_h
        w  = (x2c - x1c) / img_w
        h  = (y2c - y1c) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------------


def process_split(
    frames: list[dict],
    img_dir: Path,
    output_dir: Path,
    split: str,
    copy_images: bool,
) -> tuple[list[dict], Counter, Counter, Counter, int]:
    """Write YOLO labels + link/copy images. Returns (manifest_rows, class_counts, weather_counts, time_counts, skipped)."""
    out_img = output_dir / "images" / split
    out_lbl = output_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    class_counts:   Counter = Counter()
    weather_counts: Counter = Counter()
    time_counts:    Counter = Counter()
    skipped = 0

    for frame in frames:
        name, weather, timeofday, scene, boxes = parse_frame(frame)

        src_img = img_dir / name
        if not src_img.exists():
            skipped += 1
            continue
        if not boxes:
            skipped += 1
            continue

        stem = Path(name).stem
        dst_img = out_img / (stem + ".jpg")
        dst_lbl = out_lbl / (stem + ".txt")

        write_yolo_label(dst_lbl, boxes)

        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            os.symlink(src_img.resolve(), dst_img)

        for cls_id, *_ in boxes:
            class_counts[TARGET_CLASSES[cls_id]] += 1
        weather_counts[weather] += 1
        time_counts[timeofday] += 1

        manifest_rows.append({
            "split":        split,
            "source_image": str(src_img),
            "output_image": str(dst_img),
            "output_label": str(dst_lbl),
            "weather":      weather,
            "timeofday":    timeofday,
            "scene":        scene,
            "num_objects":  len(boxes),
        })

    print(f"  {split}: {len(manifest_rows)} images written, {skipped} skipped")
    return manifest_rows, class_counts, weather_counts, time_counts, skipped


# ---------------------------------------------------------------------------
# Write dataset.yaml and stats.json
# ---------------------------------------------------------------------------


def write_dataset_yaml(output_dir: Path) -> None:
    content = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
    )
    for i, name in enumerate(TARGET_CLASSES):
        content += f"  {i}: {name}\n"
    (output_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def write_stats(
    output_dir: Path,
    splits: dict[str, int],
    class_counts: Counter,
    weather_counts: Counter,
    time_counts: Counter,
    skipped: dict[str, int],
) -> None:
    stats = {
        "images_per_split": dict(splits),
        "boxes_per_class":  dict(class_counts),
        "weather_distribution": dict(weather_counts),
        "timeofday_distribution": dict(time_counts),
        "skipped": dict(skipped),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_manifest(output_dir: Path, rows: list[dict]) -> None:
    fieldnames = ["split", "source_image", "output_image", "output_label",
                  "weather", "timeofday", "scene", "num_objects"]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BDD100K JSON labels → YOLO 6-class format"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip-path", type=Path,
                     help="Path to BDD100K.zip (sẽ tự giải nén)")
    src.add_argument("--raw-dir", type=Path,
                     help="Thư mục đã giải nén BDD100K")

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=30000,
                        help="Số ảnh train tối đa (default: 30000)")
    parser.add_argument("--subset-mode", choices=["random", "condition_aware"],
                        default="condition_aware",
                        help="Chiến lược chọn subset train (default: condition_aware)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy ảnh thay vì symlink (tốn disk hơn)")
    parser.add_argument("--full", action="store_true",
                        help="Dùng toàn bộ train set (bỏ qua --subset-size)")
    parser.add_argument("--clean", action="store_true",
                        help="Xóa --output-dir nếu đã tồn tại")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.clean and output_dir.exists():
        print(f"Removing existing {output_dir} ...")
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.glob("images/*/*")):
        raise RuntimeError(
            f"{output_dir} đã chứa dữ liệu. Dùng --clean để xóa và tạo lại."
        )

    # Locate raw dir
    if args.zip_path:
        raw_dir = output_dir.parent / "bdd100k_raw"
        if not raw_dir.exists():
            extract_zip(args.zip_path.resolve(), raw_dir)
        else:
            print(f"Raw dir {raw_dir} already exists, skipping extraction.")
    else:
        raw_dir = args.raw_dir.resolve()

    img_train_dir, img_val_dir, lbl_train_json, lbl_val_json = find_bdd_paths(raw_dir)

    # Load train frames
    print("Loading train JSON ...")
    train_frames_all = json.loads(lbl_train_json.read_text(encoding="utf-8"))
    print(f"  Total train records: {len(train_frames_all)}")

    # Filter: keep only frames that have at least 1 target-class object
    # (quick pre-filter by checking labels exist — full parse happens in process_split)
    def has_target(frame: dict) -> bool:
        for lbl in frame.get("labels", []):
            if BDD_TO_PROJECT.get(str(lbl.get("category", "")).lower()):
                if lbl.get("box2d"):
                    return True
        return False

    train_frames_valid = [f for f in train_frames_all if has_target(f)]
    print(f"  Frames with target objects: {len(train_frames_valid)}")

    # Subset train
    if args.full:
        train_frames = train_frames_valid
        print(f"  Using full train set: {len(train_frames)}")
    elif args.subset_mode == "condition_aware":
        train_frames = condition_aware_subset(train_frames_valid, args.subset_size, args.seed)
    else:
        train_frames = random_subset(train_frames_valid, args.subset_size, args.seed)
        print(f"  Random subset: {len(train_frames)}")

    # Load val frames (always use all BDD val)
    print("Loading val JSON ...")
    val_frames_all = json.loads(lbl_val_json.read_text(encoding="utf-8"))
    val_frames = [f for f in val_frames_all if has_target(f)]
    print(f"  Val records with target objects: {len(val_frames)}")

    # Process splits
    all_manifest: list[dict] = []
    all_class: Counter = Counter()
    all_weather: Counter = Counter()
    all_time: Counter = Counter()
    split_counts: dict[str, int] = {}
    split_skipped: dict[str, int] = {}

    print("\nProcessing train split ...")
    rows, cls, wea, tim, skip = process_split(
        train_frames, img_train_dir, output_dir, "train", args.copy_images
    )
    all_manifest.extend(rows)
    all_class.update(cls)
    all_weather.update(wea)
    all_time.update(tim)
    split_counts["train"] = len(rows)
    split_skipped["train"] = skip

    print("Processing val split ...")
    rows, cls, wea, tim, skip = process_split(
        val_frames, img_val_dir, output_dir, "val", args.copy_images
    )
    all_manifest.extend(rows)
    split_counts["val"] = len(rows)
    split_skipped["val"] = skip

    # Write outputs
    write_dataset_yaml(output_dir)
    write_manifest(output_dir, all_manifest)
    write_stats(output_dir, split_counts, all_class, all_weather, all_time, split_skipped)

    print(f"\nDone. Dataset at: {output_dir}")
    print(f"  train: {split_counts.get('train', 0)} images")
    print(f"  val:   {split_counts.get('val', 0)} images")
    print(f"\nClass distribution (boxes):")
    for cls_name in TARGET_CLASSES:
        print(f"  {cls_name}: {all_class.get(cls_name, 0)}")


if __name__ == "__main__":
    main()
