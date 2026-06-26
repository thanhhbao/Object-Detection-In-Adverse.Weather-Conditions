#!/usr/bin/env python3
"""Vẽ lưới ảnh mẫu (kèm bounding box ground-truth) cho một dataset YOLO.

Phục vụ phần "Bộ dữ liệu" trong luận văn. Hỗ trợ lấy mẫu ngẫu nhiên, hoặc với
DAWN lấy đại diện mỗi điều kiện thời tiết (đọc manifest.csv).

Ví dụ:
  # DAWN: mỗi thời tiết một ảnh
  python scripts/visualize_dataset_samples.py --data /content/dawn_6cls_yolo/dataset.yaml \\
      --split train --by-weather --out dawn_samples.png

  # BDD100K: 6 ảnh ngẫu nhiên
  python scripts/visualize_dataset_samples.py --data /content/bdd100k_6cls_yolo/dataset.yaml \\
      --split train --num 6 --out bdd_samples.png
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "thesis" / "figures"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Đường dẫn dataset.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num", type=int, default=6, help="Số ảnh khi lấy ngẫu nhiên")
    parser.add_argument("--by-weather", action="store_true", help="Mỗi thời tiết một ảnh (DAWN)")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="dataset_samples.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def label_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_boxes(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = (float(v) for v in parts)
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        boxes.append((int(cls), x1, y1, bw * width, bh * height))
    return boxes


def split_dir(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    sub = Path(data[split])
    return sub if sub.is_absolute() else (root / sub).resolve()


def pick_images(args, data_yaml: Path):
    """Trả về list (image_path, caption)."""
    if args.by_weather:
        manifest = Path(args.manifest) if args.manifest else data_yaml.parent / "manifest.csv"
        groups = defaultdict(list)
        with manifest.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            img_col = "image" if "image" in (reader.fieldnames or []) else "output_image"
            for row in reader:
                if row["split"] == args.split:
                    groups[row["weather"]].append(row[img_col])
        random.seed(args.seed)
        return [(Path(random.choice(paths)), weather)
                for weather, paths in sorted(groups.items())]

    images = sorted(p for p in split_dir(data_yaml, args.split).rglob("*")
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    random.seed(args.seed)
    chosen = random.sample(images, min(args.num, len(images)))
    return [(p, p.stem) for p in chosen]


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]

    samples = pick_images(args, data_yaml)
    if not samples:
        raise SystemExit("Không tìm thấy ảnh phù hợp.")

    cols = min(len(samples), 2 if args.by_weather else 3)
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    seen_classes = set()
    for ax, (image_path, caption) in zip(axes, samples):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ax.imshow(image)
        for cls, x1, y1, bw, bh in read_boxes(label_for(image_path), width, height):
            color = COLORS[cls % len(COLORS)]
            ax.add_patch(Rectangle((x1, y1), bw, bh, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1 - 3, names[cls], color="white", fontsize=8,
                    bbox=dict(facecolor=color, edgecolor="none", pad=1))
            seen_classes.add(cls)
        ax.set_title(caption, fontsize=11)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / args.out
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output} ({len(samples)} ảnh, các lớp xuất hiện: "
          f"{sorted(names[c] for c in seen_classes)})")


if __name__ == "__main__":
    main()
