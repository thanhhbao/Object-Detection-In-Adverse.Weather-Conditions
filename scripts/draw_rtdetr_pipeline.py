#!/usr/bin/env python3
"""Vẽ sơ đồ pipeline RT-DETR cho luận văn -> docs/thesis/figures/rtdetr_pipeline.png

Chạy: /tmp/thesis_venv/bin/python scripts/draw_rtdetr_pipeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "thesis" / "figures" / "rtdetr_pipeline.png"

# (tiêu đề, mô tả, màu nền)
STAGES = [
    ("Ảnh đầu vào", "", "#C8E6C9"),
    ("Backbone", "Trích xuất bản đồ đặc trưng", "#E3F2FD"),
    ("Hybrid Encoder", "Kết hợp đặc trưng đa tỉ lệ và ngữ cảnh toàn cục", "#E3F2FD"),
    ("Query Selection", "Chọn các đặc trưng đại diện làm truy vấn đối tượng", "#E3F2FD"),
    ("Transformer Decoder", "Cập nhật truy vấn và dự đoán đối tượng", "#E3F2FD"),
    ("Detection Head", "Dự đoán hộp bao, nhãn lớp, độ tin cậy", "#E3F2FD"),
    ("Kết quả phát hiện", "Người, xe đạp, ô tô, xe máy, xe buýt, xe tải", "#FFE0B2"),
]

BOX_W, BOX_H, GAP = 7.2, 1.25, 0.7


def main() -> None:
    n = len(STAGES)
    fig_h = n * (BOX_H + GAP)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_center = 5.0
    top = fig_h - 0.3
    centers = []
    for i, (title, desc, color) in enumerate(STAGES):
        y_top = top - i * (BOX_H + GAP)
        y_bottom = y_top - BOX_H
        y_mid = (y_top + y_bottom) / 2
        centers.append((y_top, y_bottom))

        box = FancyBboxPatch(
            (x_center - BOX_W / 2, y_bottom), BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.6, edgecolor="#1565C0", facecolor=color,
        )
        ax.add_patch(box)

        if desc:
            ax.text(x_center, y_mid + 0.22, title, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#0D47A1")
            ax.text(x_center, y_mid - 0.28, desc, ha="center", va="center",
                    fontsize=9.5, color="#333333")
        else:
            ax.text(x_center, y_mid, title, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#1B5E20")

    # Mũi tên nối các khối
    for i in range(n - 1):
        y_from = centers[i][1]
        y_to = centers[i + 1][0]
        ax.add_patch(FancyArrowPatch(
            (x_center, y_from), (x_center, y_to),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#555555",
        ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
