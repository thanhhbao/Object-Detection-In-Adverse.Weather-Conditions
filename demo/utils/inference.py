"""Model loading and inference utilities."""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image

from utils.visualize import draw_detections, pil_to_bgr, bgr_to_pil, CLASSES

MODEL_PATH_ENV = "MODEL_PATH"
GDRIVE_ID_SECRET = "model_gdrive_id"
LOCAL_WEIGHTS = Path(__file__).parent.parent / "weights" / "best.pt"


def _download_weights() -> Path:
    """Download model weights from Google Drive if not cached."""
    cache_dir = Path(tempfile.gettempdir()) / "od_demo_weights"
    cache_dir.mkdir(exist_ok=True)
    cached = cache_dir / "best.pt"
    if cached.exists():
        return cached

    gdrive_id = st.secrets.get(GDRIVE_ID_SECRET, "")
    if not gdrive_id:
        st.error("Model weights not found. Set `model_gdrive_id` in Streamlit secrets.")
        st.stop()

    with st.spinner("Downloading model weights..."):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(cached), quiet=False)

    return cached


@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLO model once and cache across sessions."""
    from ultralytics import YOLO

    # Priority: env var → local weights → Google Drive
    if os.environ.get(MODEL_PATH_ENV):
        path = Path(os.environ[MODEL_PATH_ENV])
    elif LOCAL_WEIGHTS.exists():
        path = LOCAL_WEIGHTS
    else:
        path = _download_weights()

    with st.spinner("Loading model..."):
        model = YOLO(str(path))

    return model


def predict_image(
    model,
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[int] | None = None,
) -> tuple[Image.Image, list[dict], float]:
    """Run inference on a PIL image.

    Returns:
        annotated image, list of detection dicts, inference time (ms)
    """
    img_bgr = pil_to_bgr(image)

    t0 = time.perf_counter()
    results = model.predict(
        img_bgr,
        conf=conf,
        iou=iou,
        classes=classes,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    boxes_raw = []
    detections = []

    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes_raw.append((cls_id, conf_val, x1, y1, x2, y2))
            detections.append({
                "class": CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id),
                "confidence": round(conf_val, 3),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "width": int(x2 - x1), "height": int(y2 - y1),
            })

    annotated_bgr = draw_detections(img_bgr, boxes_raw, conf_threshold=conf)
    annotated_pil = bgr_to_pil(annotated_bgr)

    return annotated_pil, detections, elapsed_ms


def predict_video(
    model,
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[int] | None = None,
    max_frames: int = 300,
) -> Generator[tuple[np.ndarray, list[dict]], None, None]:
    """Yield (annotated_frame_bgr, detections) for each frame."""
    cap = cv2.VideoCapture(video_path)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, iou=iou, classes=classes, verbose=False)
        result = results[0]
        boxes_raw = []
        detections = []

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_raw.append((cls_id, conf_val, x1, y1, x2, y2))
                detections.append({
                    "class": CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id),
                    "confidence": round(conf_val, 3),
                })

        annotated = draw_detections(frame, boxes_raw, conf_threshold=conf)
        yield annotated, detections, frame_idx, total
        frame_idx += 1

    cap.release()


def process_video_to_file(
    model,
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[int] | None = None,
    max_frames: int = 300,
    progress_callback=None,
) -> tuple[str, dict]:
    """Process full video, write annotated output to temp file.

    Returns (output_path, stats_dict)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    cap.release()

    out_path = tempfile.mktemp(suffix="_detected.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    all_detections: list[dict] = []
    for frame, dets, idx, total_f in predict_video(model, video_path, conf, iou, classes, max_frames):
        writer.write(frame)
        all_detections.extend(dets)
        if progress_callback:
            progress_callback(idx, total_f)

    writer.release()

    from collections import Counter
    class_counts = Counter(d["class"] for d in all_detections)
    stats = {
        "total_detections": len(all_detections),
        "class_counts": dict(class_counts),
        "frames_processed": total,
    }

    return out_path, stats
