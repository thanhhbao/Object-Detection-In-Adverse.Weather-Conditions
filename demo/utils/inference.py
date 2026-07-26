"""Model loading and inference utilities."""

from __future__ import annotations

import os
import time
import tempfile
import urllib.request
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse

import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image

from utils.visualize import draw_detections, pil_to_bgr, bgr_to_pil, CLASSES

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

MODEL_PATH_ENV = "MODEL_PATH"
GDRIVE_ID_SECRET = "model_gdrive_id"
_DEMO_ROOT = Path(__file__).parent.parent
# Candidate local locations, checked in order
LOCAL_CANDIDATES = [
    _DEMO_ROOT / "weights" / "best.pt",
    _DEMO_ROOT / "runs" / "best.pt",
]


def _find_local_weights() -> Path | None:
    for p in LOCAL_CANDIDATES:
        if p.exists():
            return p
    return None


def list_local_models() -> dict[str, str]:
    """Return {display_name: path} for every .pt found in demo/runs and demo/weights.

    Lets the demo offer several checkpoints (e.g. v3_960, v8s, v5) to switch between
    for testing, without replacing a single best.pt.
    """
    found: dict[str, str] = {}
    for folder in (_DEMO_ROOT / "runs", _DEMO_ROOT / "weights"):
        if folder.exists():
            for p in sorted(folder.rglob("*.pt")):
                found[p.stem] = str(p)
    return found


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
def load_model(weights_path: str | None = None):
    """Load a YOLO model, cached per path so multiple checkpoints can coexist.

    If weights_path is given, load it. Otherwise: env var → local weights → Google Drive.
    """
    from ultralytics import YOLO

    if weights_path:
        path = Path(weights_path)
    else:
        local = _find_local_weights()
        if os.environ.get(MODEL_PATH_ENV):
            path = Path(os.environ[MODEL_PATH_ENV])
        elif local is not None:
            path = local
        else:
            path = _download_weights()

    with st.spinner("Loading model..."):
        model = YOLO(str(path))

    return model


def detect_image_boxes(
    model,
    image: Image.Image,
    iou: float = 0.45,
    classes: list[int] | None = None,
    conf_floor: float = 0.05,
) -> tuple[float, list[tuple]]:
    """Run inference once at a low confidence floor and return raw boxes.

    The UI can then filter these boxes by confidence live (no re-inference).
    Returns (elapsed_ms, boxes) where each box is (cls_id, conf, x1, y1, x2, y2).
    """
    img_bgr = pil_to_bgr(image)
    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=conf_floor, iou=iou, classes=classes, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = results[0]
    boxes: list[tuple] = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            boxes.append((
                int(box.cls[0]), float(box.conf[0]),
                *[float(v) for v in box.xyxy[0].tolist()],
            ))
    return elapsed_ms, boxes


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


def _is_direct_video_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(VIDEO_EXTENSIONS)


def download_video_from_url(url: str, max_mb: int = 200) -> str:
    """Download a video from a URL to a temp file. Returns local path.

    - Direct video links (.mp4/.avi/...) are fetched via HTTP.
    - Other links (YouTube, streaming sites) fall back to yt-dlp.
    Raises RuntimeError with a readable message on failure.
    """
    url = url.strip()
    if not url:
        raise RuntimeError("Empty URL.")

    if _is_direct_video_url(url):
        out_path = tempfile.mktemp(suffix=Path(urlparse(url).path).suffix or ".mp4")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                size = int(resp.headers.get("Content-Length", 0))
                if size and size > max_mb * 1024 * 1024:
                    raise RuntimeError(f"Video too large ({size // (1024*1024)} MB > {max_mb} MB).")
                with open(out_path, "wb") as f:
                    downloaded = 0
                    while chunk := resp.read(1 << 20):
                        downloaded += len(chunk)
                        if downloaded > max_mb * 1024 * 1024:
                            raise RuntimeError(f"Video exceeds {max_mb} MB limit.")
                        f.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"Could not download video: {exc}") from exc
        return out_path

    # Streaming site → yt-dlp
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "This looks like a streaming link (e.g. YouTube). "
            "Install yt-dlp to support it, or paste a direct .mp4 link."
        ) from exc

    out_path = tempfile.mktemp(suffix=".mp4")
    ydl_opts = {
        "format": "mp4[height<=720]/best[height<=720]/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    # Enable browser impersonation to bypass Cloudflare anti-bot (needs curl_cffi).
    # Pick the first available target rather than a hardcoded one.
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as probe:
            targets = probe._get_available_impersonate_targets()
        if targets:
            ydl_opts["impersonate"] = targets[0][0]
    except Exception:
        pass
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(
            f"yt-dlp failed: {exc}. Streaming sites may block cloud servers — "
            "try a direct video link or upload the file."
        ) from exc

    if not Path(out_path).exists():
        raise RuntimeError("Download finished but no file was produced.")
    return out_path


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
    max_frames: int | None = None,
    max_seconds: float | None = None,
    progress_callback=None,
) -> tuple[str, dict]:
    """Process a video, write annotated output to temp file.

    Frame budget (first match wins):
      1. max_seconds → first N seconds (N * source_fps frames)
      2. max_frames  → first N frames
      3. neither     → the entire video

    Returns (output_path, stats_dict)
    """
    import imageio.v2 as imageio
    from collections import Counter, defaultdict
    from statistics import mean, median

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Resolve frame budget.
    if max_seconds is not None:
        eff_max = int(max_seconds * src_fps)
    elif max_frames is not None:
        eff_max = max_frames
    else:
        eff_max = src_total if src_total > 0 else 10_000_000
    if src_total > 0:
        eff_max = min(eff_max, src_total)
    total = eff_max

    out_path = tempfile.mktemp(suffix="_detected.mp4")
    # H.264 via ffmpeg — produces browser-playable MP4 (mp4v/OpenCV does not).
    writer = imageio.get_writer(
        out_path,
        fps=src_fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=16,   # pads dims to multiple of 16, avoids green artifacts
        output_params=["-crf", "23", "-preset", "veryfast"],
    )

    all_detections: list[dict] = []
    per_frame_counts: list[int] = []          # detections in each frame (temporal)
    confidences: list[float] = []             # every detection's confidence
    per_class_conf: dict[str, list] = defaultdict(list)

    t_start = time.perf_counter()
    for frame, dets, idx, total_f in predict_video(model, video_path, conf, iou, classes, eff_max):
        # OpenCV frame is BGR → imageio expects RGB
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        all_detections.extend(dets)
        per_frame_counts.append(len(dets))
        for d in dets:
            confidences.append(d["confidence"])
            per_class_conf[d["class"]].append(d["confidence"])
        if progress_callback:
            progress_callback(idx, total_f)
    elapsed_s = time.perf_counter() - t_start

    writer.close()

    class_counts = Counter(d["class"] for d in all_detections)
    frames_done = len(per_frame_counts)
    n_det = len(all_detections)

    # Per-class summary: count, share %, avg confidence
    per_class = {
        cls: {
            "count": cnt,
            "share": round(100 * cnt / n_det, 1) if n_det else 0.0,
            "avg_conf": round(mean(per_class_conf[cls]), 3) if per_class_conf[cls] else 0.0,
        }
        for cls, cnt in class_counts.items()
    }

    stats = {
        # counts
        "total_detections": n_det,
        "frames_processed": frames_done,
        "class_counts": dict(class_counts),
        "per_class": per_class,
        # performance
        "elapsed_s": round(elapsed_s, 2),
        "throughput_fps": round(frames_done / elapsed_s, 1) if elapsed_s > 0 else 0.0,
        "ms_per_frame": round(1000 * elapsed_s / frames_done, 1) if frames_done else 0.0,
        # video metadata
        "resolution": f"{src_w}×{src_h}",
        "source_fps": round(src_fps, 1),
        "duration_s": round(src_total / src_fps, 1) if src_fps else 0.0,
        # confidence distribution
        "conf_mean": round(mean(confidences), 3) if confidences else 0.0,
        "conf_median": round(median(confidences), 3) if confidences else 0.0,
        "conf_min": round(min(confidences), 3) if confidences else 0.0,
        "conf_max": round(max(confidences), 3) if confidences else 0.0,
        # temporal / distributions (for charts)
        "per_frame_counts": per_frame_counts,
        "confidences": confidences,
        "peak_per_frame": max(per_frame_counts) if per_frame_counts else 0,
        "avg_per_frame": round(n_det / frames_done, 2) if frames_done else 0.0,
    }

    return out_path, stats
