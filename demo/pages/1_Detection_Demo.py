"""Detection Demo — image and video inference."""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.inference import load_model, predict_image, process_video_to_file
from utils.visualize import CLASS_COLORS_HEX, CLASSES, resize_for_display

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detection Demo · Adverse Weather OD",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = (Path(__file__).parent.parent / "assets" / "style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")

    conf_threshold = st.slider(
        "Confidence threshold", 0.10, 0.95, 0.30, 0.05,
        help="Minimum confidence score to show a detection."
    )
    iou_threshold = st.slider(
        "IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05,
        help="Higher = fewer overlapping boxes."
    )

    st.markdown("**Filter classes**")
    selected_classes = []
    class_cols = st.columns(2)
    for i, cls in enumerate(CLASSES):
        color = CLASS_COLORS_HEX[i]
        checked = class_cols[i % 2].checkbox(
            cls.capitalize(), value=True,
            key=f"cls_{i}"
        )
        if checked:
            selected_classes.append(i)

    classes_arg = selected_classes if len(selected_classes) < len(CLASSES) else None

    st.divider()
    st.markdown("""
    <div class="info-card">
        <h4>Model Info</h4>
        <div style="color:#8892b0; font-size:0.82rem; line-height:1.8;">
            <b style="color:#e8eaf6;">Architecture</b><br>Best Phase-1 model<br><br>
            <b style="color:#e8eaf6;">Input</b><br>640 × 640 px<br><br>
            <b style="color:#e8eaf6;">Classes</b><br>6 (person → truck)<br><br>
            <b style="color:#e8eaf6;">Training</b><br>BDD100K → XWOD
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="font-size:1.8rem; font-weight:800; color:#e8eaf6; margin-bottom:0.2rem;">
        🎯 Detection Demo
    </h1>
    <p style="color:#6b7394; font-size:0.9rem;">
        Upload an image or video — the model detects objects under any weather condition.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📷  Image", "🎬  Video"])


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader(
        "Drop an image here, or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        w, h = image.size

        detect_btn = st.button("Detect Objects", use_container_width=True, key="detect_img")

        if detect_btn:
            with st.spinner("Running inference..."):
                annotated, detections, elapsed_ms = predict_image(
                    model, image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=classes_arg,
                )

            # ── Result images side by side ──
            col_orig, col_det = st.columns(2, gap="medium")
            with col_orig:
                st.markdown('<div class="result-header">Original</div>', unsafe_allow_html=True)
                st.image(resize_for_display(image), use_container_width=True)
            with col_det:
                st.markdown('<div class="result-header">Detected</div>', unsafe_allow_html=True)
                st.image(resize_for_display(annotated), use_container_width=True)

            # ── Stats row ──
            n_det = len(detections)
            class_counts = Counter(d["class"] for d in detections)
            avg_conf = (
                sum(d["confidence"] for d in detections) / n_det if n_det else 0
            )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detections", n_det)
            m2.metric("Inference", f"{elapsed_ms:.0f} ms")
            m3.metric("Avg Confidence", f"{avg_conf:.2f}")
            m4.metric("Image Size", f"{w}×{h}")

            # ── Class breakdown ──
            if detections:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Detections by class**")
                badges = "".join(
                    f'<span class="class-badge" style="background:{CLASS_COLORS_HEX.get(CLASSES.index(cls), "#333")}22; '
                    f'color:{CLASS_COLORS_HEX.get(CLASSES.index(cls), "#aaa")}; '
                    f'border:1px solid {CLASS_COLORS_HEX.get(CLASSES.index(cls), "#333")}44;">'
                    f'{cls} × {cnt}</span>'
                    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
                    if cls in CLASSES
                )
                st.markdown(f'<div style="margin-bottom:1rem;">{badges}</div>', unsafe_allow_html=True)

                # Detection table
                with st.expander("View detection table"):
                    df = pd.DataFrame(detections)[["class", "confidence", "width", "height"]]
                    df.index += 1
                    st.dataframe(df, use_container_width=True)

                # Download annotated image
                import io
                buf = io.BytesIO()
                annotated.save(buf, format="JPEG", quality=95)
                st.download_button(
                    "⬇️ Download annotated image",
                    buf.getvalue(),
                    file_name=f"detected_{uploaded.name}",
                    mime="image/jpeg",
                )
            else:
                st.info("No objects detected above the confidence threshold. Try lowering it in the sidebar.")

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#4a5270;">
            <div style="font-size:3rem;">📷</div>
            <div style="margin-top:1rem; font-size:1rem;">Upload an image to get started</div>
            <div style="font-size:0.8rem; margin-top:0.4rem;">JPG, PNG, WebP supported</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Drop a video here, or click to browse",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="video_upload",
    )

    max_frames = st.slider(
        "Max frames to process", 30, 500, 150, 10,
        help="Limit processing time. 150 frames ≈ 5–6 seconds at 25fps."
    )

    if uploaded_vid:
        # Write to temp file
        tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_in.write(uploaded_vid.read())
        tmp_in.flush()

        # Preview original
        st.markdown("**Preview (original)**")
        st.video(tmp_in.name)

        detect_vid_btn = st.button("Process Video", use_container_width=True, key="detect_vid")

        if detect_vid_btn:
            progress_bar = st.progress(0, text="Processing frames...")
            status_text = st.empty()

            def update_progress(idx: int, total: int) -> None:
                pct = min(idx / max(total, 1), 1.0)
                progress_bar.progress(pct, text=f"Frame {idx+1} / {total}")
                status_text.markdown(f"<small style='color:#6b7394;'>Processing...</small>", unsafe_allow_html=True)

            with st.spinner("Running detection on all frames..."):
                out_path, stats = process_video_to_file(
                    model, tmp_in.name,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=classes_arg,
                    max_frames=max_frames,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0, text="Done!")
            status_text.empty()

            # ── Result video ──
            st.markdown("<br>**Detected output**", unsafe_allow_html=True)
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            # ── Stats ──
            st.markdown("<br>", unsafe_allow_html=True)
            sv1, sv2, sv3 = st.columns(3)
            sv1.metric("Frames Processed", stats["frames_processed"])
            sv2.metric("Total Detections", stats["total_detections"])
            sv3.metric("Avg per Frame",
                       f"{stats['total_detections'] / max(stats['frames_processed'],1):.1f}")

            # Class breakdown
            if stats["class_counts"]:
                st.markdown("**Detections by class**")
                badges = "".join(
                    f'<span class="class-badge" style="background:{CLASS_COLORS_HEX.get(CLASSES.index(cls), "#333")}22; '
                    f'color:{CLASS_COLORS_HEX.get(CLASSES.index(cls), "#aaa")}; '
                    f'border:1px solid {CLASS_COLORS_HEX.get(CLASSES.index(cls), "#333")}44;">'
                    f'{cls} × {cnt}</span>'
                    for cls, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1])
                    if cls in CLASSES
                )
                st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

            # Download
            st.download_button(
                "⬇️ Download annotated video",
                video_bytes,
                file_name=f"detected_{uploaded_vid.name}",
                mime="video/mp4",
            )

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#4a5270;">
            <div style="font-size:3rem;">🎬</div>
            <div style="margin-top:1rem; font-size:1rem;">Upload a video to get started</div>
            <div style="font-size:0.8rem; margin-top:0.4rem;">MP4, AVI, MOV supported · Up to 200MB</div>
        </div>
        """, unsafe_allow_html=True)
