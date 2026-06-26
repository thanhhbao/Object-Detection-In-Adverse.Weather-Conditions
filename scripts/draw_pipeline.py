#!/usr/bin/env python3
"""Vẽ sơ đồ pipeline mô hình (chuyên nghiệp) cho luận văn.

Render kiểu nhiều lớp: mỗi khối có tiêu đề + mô tả + dòng kỹ thuật (màu nhấn),
nhãn bước bên phải, các tag phụ nhỏ phía dưới, và pill kết quả ở cuối.

Chạy: /tmp/thesis_venv/bin/python scripts/draw_pipeline.py rtdetr
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "thesis" / "figures"

# Bảng màu pastel: (nền, viền/nhấn, chữ tiêu đề)
PALETTE = {
    "input": ("#ECEFF1", "#90A4AE", "#37474F"),
    "p1": ("#EDE7F6", "#7E57C2", "#4527A0"),
    "p2": ("#E3F2FD", "#42A5F5", "#1565C0"),
    "p3": ("#E0F2F1", "#26A69A", "#00695C"),
    "p4": ("#FFF3E0", "#FFA726", "#E65100"),
    "p5": ("#FCE4EC", "#EC407A", "#AD1457"),
    "output": ("#E8F5E9", "#66BB6A", "#2E7D32"),
}

CX = 4.6          # tâm cột chính
BOX_W = 6.2
BOX_H = 1.5
GAP = 1.35        # khoảng cách giữa các khối (chừa chỗ cho sub-tag)


def pill(ax, x, y, text, *, fill="#FFFFFF", edge="#B0BEC5", tcolor="#455A64", fontsize=8.5):
    pad_w = 0.07 * len(text) + 0.5
    box = FancyBboxPatch(
        (x - pad_w / 2, y - 0.22), pad_w, 0.44,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=tcolor)


def draw(stages, output_path, title=None):
    n = len(stages)
    fig_h = n * (BOX_H + GAP) + 0.5
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.4
    spans = []
    for i, s in enumerate(stages):
        fill, edge, tcolor = PALETTE[s["color"]]
        is_output = s.get("kind") == "output"
        h = 0.7 if is_output else BOX_H
        w = BOX_W if not is_output else BOX_W * 0.95
        y_top = top - i * (BOX_H + GAP)
        y_bottom = y_top - h
        y_mid = (y_top + y_bottom) / 2
        spans.append((y_top, y_bottom))

        ax.add_patch(FancyBboxPatch(
            (CX - w / 2, y_bottom), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.8, edgecolor=edge, facecolor=fill,
        ))

        if is_output:
            ax.text(CX, y_mid, s["title"], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tcolor)
        else:
            ax.text(CX, y_mid + 0.42, s["title"], ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color=tcolor)
            if s.get("desc"):
                ax.text(CX, y_mid + 0.05, s["desc"], ha="center", va="center",
                        fontsize=9.5, color="#37474F")
            if s.get("tech"):
                ax.text(CX, y_mid - 0.35, s["tech"], ha="center", va="center",
                        fontsize=8.5, color=edge, style="italic")

        # Nhãn bước bên phải
        if s.get("step"):
            ax.text(9.5, y_mid, s["step"], ha="right", va="center",
                    fontsize=9, color="#B0BEC5")
        # Tag chi tiết sát mép phải khối
        if s.get("right_tag"):
            pill(ax, CX + w / 2 + 0.0, y_top - 0.0, s["right_tag"],
                 edge=edge, tcolor=tcolor)

        # Sub-tag phía dưới khối (trong khoảng gap)
        subs = s.get("subs", [])
        if subs and i < n - 1:
            y_sub = y_bottom - GAP / 2
            total = len(subs)
            step_x = min(2.4, BOX_W / total)
            start = CX - step_x * (total - 1) / 2
            for k, tag in enumerate(subs):
                pill(ax, start + k * step_x, y_sub, tag, edge=edge, tcolor=tcolor)

    # Mũi tên nối
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch(
            (CX, spans[i][1]), (CX, spans[i + 1][0]),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#607D8B",
            shrinkA=0, shrinkB=0,
        ))

    if title:
        ax.text(CX, fig_h - 0.1, title, ha="center", va="top",
                fontsize=13, fontweight="bold", color="#263238")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


RTDETR = [
    {"title": "Ảnh đầu vào", "desc": "", "tech": "", "color": "input",
     "step": "Đầu vào", "right_tag": "RGB · 640×640"},
    {"title": "Backbone", "desc": "Trích xuất bản đồ đặc trưng",
     "tech": "ResNet-50 / HGNetv2", "color": "p1", "step": "Bước 1",
     "right_tag": "Feature map", "subs": ["S3", "S4", "S5"]},
    {"title": "Hybrid Encoder", "desc": "Kết hợp đặc trưng đa tỉ lệ và ngữ cảnh toàn cục",
     "tech": "AIFI · CCFM", "color": "p2", "step": "Bước 2"},
    {"title": "Query Selection", "desc": "Chọn các đặc trưng đại diện làm truy vấn đối tượng",
     "tech": "IoU-aware query selection", "color": "p3", "step": "Bước 3",
     "right_tag": "~300 queries"},
    {"title": "Transformer Decoder", "desc": "Cập nhật truy vấn và dự đoán đối tượng",
     "tech": "Self / Cross deformable attention", "color": "p4", "step": "Bước 4"},
    {"title": "Detection Head", "desc": "Dự đoán hộp bao, nhãn lớp, độ tin cậy",
     "tech": "FFN · KHÔNG cần NMS", "color": "p5", "step": "Bước 5",
     "subs": ["Nhánh phân loại", "Nhánh hộp bao"]},
    {"title": "Kết quả phát hiện — Người · xe đạp · ô tô · xe máy · xe buýt · xe tải",
     "color": "output", "kind": "output", "step": "Đầu ra"},
]

DIAGRAMS = {
    "rtdetr": (RTDETR, "Kiến trúc pipeline mô hình RT-DETR", "rtdetr_pipeline.png"),
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "rtdetr"
    if name not in DIAGRAMS:
        raise SystemExit(f"Chưa có sơ đồ '{name}'. Có: {list(DIAGRAMS)}")
    stages, title, filename = DIAGRAMS[name]
    draw(stages, OUT_DIR / filename, title=title)


if __name__ == "__main__":
    main()
