#!/usr/bin/env python3
"""Prepare BDD100K balanced subset for Stage 1 fine-tuning.

Lấy tối đa --per-condition ảnh từ mỗi nhóm điều kiện, ghép thành tập
cân bằng ~30K để Stage 1 không bị bias về clear-daytime.

5 nhóm điều kiện (ánh xạ từ BDD100K weather + timeofday):
  clear_overcast  — weather ∈ {clear, overcast, partly cloudy}, timeofday = daytime
  rainy           — weather = rainy, timeofday = any
  foggy           — weather = foggy,  timeofday = any
  night           — timeofday = night, weather = any
  dawn_dusk       — timeofday = dawn/dusk, weather = any

Ví dụ (30K tổng, ~6K mỗi nhóm):
  python scripts/prepare_bdd100k.py \\
    --images-dir /content/bdd100k/images/100k/train \\
    --labels-json /content/bdd100k/labels/det_20/det_train.json \\
    --output-dir /content/bdd100k_6cls_yolo \\
    --per-condition 6000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import (
    PreparedSample,
    clean_output_dir,
    clamp_box,
    export_yolo_dataset,
    find_image,
    make_yolo_dirs,
    read_image_size,
    split_train_val,
)

# ---------------------------------------------------------------------------
# Định nghĩa 5 nhóm điều kiện cần cân bằng
# ---------------------------------------------------------------------------

BDD_WEATHER_NORMALIZE = {
    "clear": "clear",
    "overcast": "overcast",
    "partly cloudy": "partly cloudy",
    "rainy": "rainy",
    "foggy": "foggy",
    "snowy": "snowy",
    "undefined": "undefined",
}

BDD_TIME_NORMALIZE = {
    "daytime": "daytime",
    "night": "night",
    "dawn/dusk": "dawn/dusk",
    "undefined": "undefined",
}


def classify_condition(weather: str, timeofday: str) -> str | None:
    """Trả về tên nhóm điều kiện hoặc None nếu không thuộc nhóm nào."""
    w = BDD_WEATHER_NORMALIZE.get(weather.lower(), "undefined")
    t = BDD_TIME_NORMALIZE.get(timeofday.lower(), "undefined")

    if t == "night":
        return "night"
    if t == "dawn/dusk":
        return "dawn_dusk"
    if w == "foggy":
        return "foggy"
    if w == "rainy":
        return "rainy"
    if w in {"clear", "overcast", "partly cloudy"} and t == "daytime":
        return "clear_overcast"
    return None


CONDITIONS = ["clear_overcast", "rainy", "foggy", "night", "dawn_dusk"]


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-condition", type=int, default=6000,
                        help="Số ảnh tối đa mỗi nhóm điều kiện (default: 6000 → ~30K tổng)")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Đọc và phân nhóm BDD100K
# ---------------------------------------------------------------------------


def load_bdd_samples(
    images_dir: Path,
    labels_json: Path,
) -> dict[str, list[PreparedSample]]:
    """Đọc toàn bộ BDD100K train JSON và phân nhóm theo 5 điều kiện."""
    frames = json.loads(labels_json.read_text(encoding="utf-8"))
    groups: dict[str, list[PreparedSample]] = {c: [] for c in CONDITIONS}
    skipped = {"no_image": 0, "no_condition": 0, "no_boxes": 0}

    for frame in frames:
        attrs = frame.get("attributes", {})
        weather = str(attrs.get("weather", "")).strip()
        timeofday = str(attrs.get("timeofday", "")).strip()
        condition = classify_condition(weather, timeofday)
        if condition is None:
            skipped["no_condition"] += 1
            continue

        image_path = find_image(images_dir, frame["name"])
        if image_path is None:
            skipped["no_image"] += 1
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        img_w, img_h = image_size

        boxes = []
        for label in frame.get("labels", []):
            box2d = label.get("box2d")
            if not box2d:
                continue
            box = clamp_box(
                label.get("category"),
                box2d["x1"], box2d["y1"],
                box2d["x2"], box2d["y2"],
                img_w, img_h,
            )
            if box is not None:
                boxes.append(box)

        if not boxes:
            skipped["no_boxes"] += 1
            continue

        groups[condition].append(
            PreparedSample(
                image=image_path,
                boxes=tuple(boxes),
                split_key=condition,
                source="bdd100k",
                weather=condition,
            )
        )

    print(f"Skipped: {skipped}")
    return groups


# ---------------------------------------------------------------------------
# Cân bằng và chọn subset
# ---------------------------------------------------------------------------


def balanced_subset(
    groups: dict[str, list[PreparedSample]],
    per_condition: int,
    seed: int,
) -> list[PreparedSample]:
    """Lấy tối đa per_condition ảnh từ mỗi nhóm, in thống kê."""
    rng = random.Random(seed)
    selected: list[PreparedSample] = []
    print(f"\n{'Condition':<18} {'Available':>10} {'Selected':>10}")
    print("-" * 42)
    for cond in CONDITIONS:
        pool = groups[cond]
        rng.shuffle(pool)
        take = min(len(pool), per_condition)
        selected.extend(pool[:take])
        print(f"{cond:<18} {len(pool):>10} {take:>10}")
    print(f"{'TOTAL':<18} {sum(len(g) for g in groups.values()):>10} {len(selected):>10}\n")
    return sorted(selected, key=lambda s: str(s.image))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=("train", "val"))

    print("Đang đọc và phân nhóm BDD100K...")
    groups = load_bdd_samples(images_dir, args.labels_json.resolve())
    samples = balanced_subset(groups, args.per_condition, args.seed)

    if not samples:
        raise RuntimeError("Không tìm thấy ảnh nào sau khi lọc.")

    assignments = split_train_val(samples, args.seed, args.train_ratio)
    export_yolo_dataset(
        samples=samples,
        assignments=assignments,
        raw_root=images_dir,
        output_dir=output_dir,
        imgsz=args.imgsz,
        splits=("train", "val"),
    )


if __name__ == "__main__":
    main()
