"""Visualization utilities: draw bounding boxes on images/frames."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# Colors per class — Tailwind palette (BGR for OpenCV)
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (68,  68,  239),   # person     → red     #ef4444
    1: (94,  197, 34),    # bicycle    → green   #22c55e
    2: (246, 130, 59),    # car        → blue    #3b82f6
    3: (247, 85,  168),   # motorcycle → purple  #a855f7
    4: (8,   179, 234),   # bus        → yellow  #eab308
    5: (153, 72,  236),   # truck      → pink    #ec4899
}

# Hex versions for display (match app.py home page)
CLASS_COLORS_HEX: dict[int, str] = {
    0: "#ef4444",
    1: "#22c55e",
    2: "#3b82f6",
    3: "#a855f7",
    4: "#eab308",
    5: "#ec4899",
}


def draw_detections(
    image: np.ndarray,
    boxes: list[tuple],
    conf_threshold: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image.

    boxes: list of (cls_id, conf, x1, y1, x2, y2) in pixel coords
    """
    img = image.copy()
    h, w = img.shape[:2]
    font_scale = max(0.4, min(w, h) / 1000)
    thickness = max(1, line_thickness)

    for cls_id, conf, x1, y1, x2, y2 in boxes:
        if conf < conf_threshold:
            continue
        cls_id = int(cls_id)
        color = CLASS_COLORS_BGR.get(cls_id, (200, 200, 200))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"{CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1, th + 4)
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, -1)

        # Label text (dark for readability)
        cv2.putText(
            img, label, (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), thickness,
        )

    return img


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def resize_for_display(image: Image.Image, max_width: int = 900) -> Image.Image:
    w, h = image.size
    if w > max_width:
        image = image.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    return image
