"""Aperture — object detection dashboard (image + video)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_paste_button import paste_image_button

sys.path.insert(0, str(Path(__file__).parent))
from utils.inference import (
    load_model,
    detect_image_boxes,
    process_video_to_file,
    download_video_from_url,
)
from utils.visualize import (
    CLASS_COLORS_HEX, CLASSES, resize_for_display,
    draw_detections, pil_to_bgr, bgr_to_pil,
)

st.set_page_config(page_title="Aperture · Object Detection",
                   page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

css = (Path(__file__).parent / "assets" / "style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

CLASS_LABELS = [c.capitalize() for c in CLASSES]
model = load_model()

# ── Session init ──────────────────────────────────────────────────────────────
st.session_state.setdefault("recents", [])       # list of {name, kind, bytes, meta}
st.session_state.setdefault("active_name", None)
st.session_state.setdefault("last_upload", None)


def upsert_recent(name: str, kind: str, data: bytes, meta: str) -> None:
    recents = [r for r in st.session_state["recents"] if r["name"] != name]
    recents.insert(0, {"name": name, "kind": kind, "bytes": data, "meta": meta})
    st.session_state["recents"] = recents[:5]


def get_active() -> dict | None:
    for r in st.session_state["recents"]:
        if r["name"] == st.session_state["active_name"]:
            return r
    return None


def donut(pct: int) -> str:
    return (f'<div class="donut" style="background:conic-gradient(var(--accent) {pct}%, '
            f'var(--line) 0);"><span>{pct}%</span></div>')


def object_rows(items: list[dict]) -> str:
    html = ""
    for it in items:
        html += (
            f'<div class="obj-row"><div class="obj-top">'
            f'<span class="obj-name"><span class="obj-dot" style="background:{it["color"]}"></span>'
            f'{it["label"]}</span><span class="obj-conf">{it["right"]}</span></div>'
            f'<div class="obj-bar"><span style="width:{it["pct"]}%;background:{it["color"]}"></span></div></div>'
        )
    return html


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOLBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    tb_brand, tb_mode, tb_actions = st.columns([3, 3, 4], vertical_alignment="center")
    with tb_brand:
        st.markdown("""
        <div class="brand">
            <div class="brand-logo">◉</div>
            <div>
                <div class="brand-title">Aperture</div>
                <div class="brand-sub">Object detection platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tb_mode:
        mode = st.segmented_control("mode", ["Image", "Video"], default="Image",
                                    label_visibility="collapsed")
        mode = mode or "Image"
    with tb_actions:
        a1, a2 = st.columns([3, 2], vertical_alignment="center")
        with a1:
            st.selectbox("Model", ["Best model · fast"], label_visibility="collapsed")
        with a2:
            run = st.button("Run detection", key="run_btn")

KIND = "image" if mode == "Image" else "video"

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — 3 columns
# ═══════════════════════════════════════════════════════════════════════════════
left, center, right = st.columns([1.15, 2.3, 1.35])

# ── LEFT: input source ────────────────────────────────────────────────────────
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-label">Input source</div>', unsafe_allow_html=True)

        if KIND == "image":
            up = st.file_uploader("Drag & drop an image, or click to browse",
                                  type=["jpg", "jpeg", "png", "webp", "bmp"],
                                  label_visibility="collapsed", key="img_up")
        else:
            up = st.file_uploader("Drag & drop a video, or click to browse",
                                  type=["mp4", "avi", "mov", "mkv"],
                                  label_visibility="collapsed", key="vid_up")

        if up is not None and st.session_state["last_upload"] != (KIND, up.name, up.size):
            data = up.getvalue()
            if KIND == "image":
                im = Image.open(io.BytesIO(data))
                meta = f"{im.width}×{im.height} · {up.size/1e6:.1f}MB"
            else:
                meta = f"{up.size/1e6:.1f}MB"
            upsert_recent(up.name, KIND, data, meta)
            st.session_state["active_name"] = up.name
            st.session_state["last_upload"] = (KIND, up.name, up.size)
            st.session_state.pop("img_result", None)
            st.session_state.pop("vid_result", None)

        # Paste image from clipboard (screenshots)
        if KIND == "image":
            pasted = paste_image_button("Paste from clipboard", key="paste_btn",
                                        background_color="#111827",
                                        hover_background_color="#000000",
                                        text_color="#ffffff", errors="ignore")
            if pasted is not None and pasted.image_data is not None:
                pimg = pasted.image_data.convert("RGB")
                buf = io.BytesIO(); pimg.save(buf, format="PNG")
                pbytes = buf.getvalue()
                digest = hashlib.md5(pbytes).hexdigest()[:8]
                if st.session_state.get("last_paste") != digest:
                    name = f"pasted-{digest}.png"
                    upsert_recent(name, "image", pbytes,
                                  f"{pimg.width}×{pimg.height} · pasted")
                    st.session_state["active_name"] = name
                    st.session_state["last_paste"] = digest
                    st.session_state.pop("img_result", None)
                    st.rerun()

        # Import video from URL
        if KIND == "video":
            with st.form("url_form", clear_on_submit=False):
                url = st.text_input("Video URL",
                                    placeholder="https://…/clip.mp4 or a YouTube link",
                                    label_visibility="collapsed")
                fetch = st.form_submit_button("Fetch", use_container_width=True)
            if fetch and url.strip():
                try:
                    with st.spinner("Downloading…"):
                        fp = download_video_from_url(url, max_mb=200)
                    with open(fp, "rb") as f:
                        vdata = f.read()
                    os.remove(fp)
                    vname = Path(url).name or "video.mp4"
                    upsert_recent(vname, "video", vdata, f"{len(vdata)/1e6:.1f}MB · url")
                    st.session_state["active_name"] = vname
                    st.session_state.pop("vid_result", None)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

        # Recent files (matching current mode)
        recents = [r for r in st.session_state["recents"] if r["kind"] == KIND]
        if recents:
            st.markdown('<div class="panel-label" style="margin-top:0.9rem;">Recent files</div>',
                        unsafe_allow_html=True)
            for i, r in enumerate(recents):
                is_active = r["name"] == st.session_state["active_name"]
                label = f"{'Active — ' if is_active else ''}{r['name']}"
                if st.button(label, key=f"rec_{i}", use_container_width=True):
                    st.session_state["active_name"] = r["name"]
                    st.session_state.pop("img_result", None)
                    st.session_state.pop("vid_result", None)
                    st.rerun()
                st.caption(r["meta"])

        # Confidence threshold
        st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)
        conf_pct = st.slider("Confidence threshold", 10, 95, 30, 1, format="%d%%")
        conf = conf_pct / 100.0

        with st.expander("Advanced"):
            iou = st.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
            sel_labels = st.multiselect("Classes", CLASS_LABELS, default=CLASS_LABELS)
            sel_ids = [CLASS_LABELS.index(l) for l in sel_labels]
            sel_set = set(sel_ids) if 0 < len(sel_ids) < len(CLASSES) else set(range(len(CLASSES)))

    if KIND == "video":
        with st.container(border=True):
            st.markdown('<div class="panel-label">Duration</div>', unsafe_allow_html=True)
            DUR = {"10s": 10.0, "30s": 30.0, "60s": 60.0, "Entire": None}
            dur_label = st.radio("Duration", list(DUR.keys()), index=1,
                                 horizontal=True, label_visibility="collapsed")
            max_seconds = DUR[dur_label]

active = get_active()

# ── Run detection (image is fast — done here; video is processed in the center) ──
if run and active and active["kind"] == "image":
    img = Image.open(io.BytesIO(active["bytes"])).convert("RGB")
    elapsed, boxes = detect_image_boxes(model, img, iou=iou, classes=None)
    st.session_state["img_result"] = {
        "name": active["name"], "boxes": boxes, "elapsed": elapsed,
        "w": img.width, "h": img.height,
    }
    st.session_state["scroll_to_results"] = True

# Video processing runs inside the center column (progress shows under the video)
do_video_run = bool(run and active and active["kind"] == "video")

# ── Build current results ─────────────────────────────────────────────────────
summary = None          # {avg, total, n_classes, time_ms, counts, confidences}
obj_items: list[dict] = []
image_dets: list[dict] = []

img_res = st.session_state.get("img_result")
vid_res = st.session_state.get("vid_result")

if KIND == "image" and active and img_res and img_res["name"] == active["name"]:
    boxes = [b for b in img_res["boxes"] if b[1] >= conf and b[0] in sel_set]
    img = Image.open(io.BytesIO(active["bytes"])).convert("RGB")
    annotated = bgr_to_pil(draw_detections(pil_to_bgr(img), boxes, conf_threshold=0))
    dets = [{"class": CLASSES[b[0]], "confidence": round(b[1], 3),
             "width": int(b[4] - b[2]), "height": int(b[5] - b[3])} for b in boxes]
    counts = Counter(d["class"] for d in dets)
    confs = [d["confidence"] for d in dets]
    summary = {
        "avg": (sum(confs) / len(confs)) if confs else 0.0,
        "total": len(dets), "n_classes": len(counts),
        "time_ms": img_res["elapsed"], "counts": dict(counts), "confidences": confs,
    }
    image_dets = dets   # sorted + turned into obj_items inside the right column

elif KIND == "video" and active and vid_res and vid_res["name"] == active["name"]:
    stats = vid_res["stats"]
    summary = {
        "avg": stats["conf_mean"], "total": stats["total_detections"],
        "n_classes": len(stats["class_counts"]), "time_ms": stats["elapsed_s"] * 1000,
        "counts": stats["class_counts"], "confidences": stats.get("confidences", []),
    }
    total = stats["total_detections"] or 1
    for cls, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        if cls not in CLASSES:
            continue
        obj_items.append({"label": cls.capitalize(), "right": str(cnt),
                          "pct": int(100 * cnt / total),
                          "color": CLASS_COLORS_HEX[CLASSES.index(cls)]})

# ── CENTER: preview ───────────────────────────────────────────────────────────
with center:
    with st.container(border=True):
        st.markdown(f"""
        <div class="canvas-head">
            <div class="canvas-status">{"Static image mode" if KIND == "image" else "Video mode"}</div>
            <div class="canvas-zoom">{active["name"] if active else "no source"}</div>
        </div>
        """, unsafe_allow_html=True)

        if not active:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">◇</div>
                <div class="empty-state-title">No source selected</div>
                <div class="empty-state-hint">Upload a file on the left to begin</div>
            </div>
            """, unsafe_allow_html=True)
        elif KIND == "image":
            if img_res and img_res["name"] == active["name"]:
                st.image(resize_for_display(annotated), use_container_width=True)
            else:
                st.image(resize_for_display(Image.open(io.BytesIO(active["bytes"])).convert("RGB")),
                         use_container_width=True)
        else:  # video
            if do_video_run:
                # Show the source while processing; progress bar sits right under it.
                st.video(active["bytes"])
                prog = st.progress(0, text="Processing frames...")

                def _cb(idx, total):
                    prog.progress(min(idx / max(total, 1), 1.0), text=f"Frame {idx+1} / {total}")

                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.write(active["bytes"]); tmp.flush()
                out_path, stats = process_video_to_file(
                    model, tmp.name, conf=conf, iou=iou,
                    classes=(list(sel_set) if len(sel_set) < len(CLASSES) else None),
                    max_seconds=max_seconds, progress_callback=_cb,
                )
                os.remove(tmp.name)
                with open(out_path, "rb") as f:
                    vbytes = f.read()
                os.remove(out_path)
                st.session_state["vid_result"] = {
                    "name": active["name"], "bytes": vbytes, "stats": stats,
                }
                st.session_state["scroll_to_results"] = True
                st.rerun()
            elif vid_res and vid_res["name"] == active["name"]:
                st.video(vid_res["bytes"])
            else:
                st.video(active["bytes"])

# ── RIGHT: detected objects ───────────────────────────────────────────────────
with right:
    with st.container(border=True):
        n = summary["total"] if summary else 0
        st.markdown(f'<div class="panel-label">Detected objects '
                    f'<span class="count">{n}</span></div>', unsafe_allow_html=True)

        if KIND == "image":
            order = st.segmented_control(
                "sort", ["Confidence", "Type"],
                default="Confidence", label_visibility="collapsed") or "Confidence"
            dets_sorted = sorted(
                image_dets,
                key=(lambda d: -d["confidence"]) if order == "Confidence" else (lambda d: d["class"]),
            )
            obj_items = [{"label": d["class"].capitalize(),
                          "right": f"{int(d['confidence']*100)}%",
                          "pct": int(d["confidence"] * 100),
                          "color": CLASS_COLORS_HEX[CLASSES.index(d["class"])]}
                         for d in dets_sorted]

        if obj_items:
            st.markdown(object_rows(obj_items), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">◎</div>
                <div class="empty-state-title">Nothing yet</div>
                <div class="empty-state-hint">Click "Run detection" to analyze this file</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  BOTTOM — post-detection analysis (only shown once there are results)
# ═══════════════════════════════════════════════════════════════════════════════
if summary:
    avg = summary["avg"]
    total = summary["total"]
    n_classes = summary["n_classes"]
    time_ms = summary["time_ms"]

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span id="results-anchor" class="section-title">Post-detection analysis</span>',
                unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    mc = st.columns(4)
    with mc[0]:
        with st.container(border=True):
            st.markdown(f"""
            <div class="donut-card">
                {donut(int(avg*100))}
                <div>
                    <div class="metric-label" style="font-size:0.8rem;color:var(--muted);">Avg confidence</div>
                    <div style="font-size:1.6rem;font-weight:800;color:var(--ink);">{avg*100:.0f}<span style="font-size:0.9rem;color:var(--muted);">%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    mc[1].metric("Total objects", total)
    mc[2].metric("Object classes", n_classes)
    mc[3].metric("Processing time", f"{time_ms:.0f} ms")

    st.markdown("<div style='margin-bottom:0.25rem;'></div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown('<div class="panel-label">Distribution by class</div>', unsafe_allow_html=True)
            if summary["counts"]:
                df = pd.DataFrame(
                    {"count": list(summary["counts"].values())},
                    index=[c.capitalize() for c in summary["counts"].keys()],
                )
                st.bar_chart(df, height=200, color="#374151")
            else:
                st.caption("No detections.")

    with p2:
        with st.container(border=True):
            st.markdown('<div class="panel-label">Confidence chart</div>', unsafe_allow_html=True)
            confs = summary["confidences"]
            if confs:
                counts, edges = np.histogram(confs, bins=10, range=(0.0, 1.0))
                hist = pd.DataFrame({"count": counts},
                                    index=[f"{edges[i]:.1f}" for i in range(len(counts))])
                st.bar_chart(hist, height=200, color="#9ca3af")
            else:
                st.caption("No detections.")

    with p3:
        with st.container(border=True):
            st.markdown('<div class="panel-label">Notes</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="notes">
                Drag the confidence threshold on the left to filter out low-certainty
                detections — useful for checking false positives before exporting a report.
                The detector is trained for adverse weather (fog, rain, snow, night).
            </div>
            """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    e1, e2, e3, _ = st.columns([1, 1, 1, 5])
    if KIND == "image":
        export_df = pd.DataFrame(
            [{"class": it["label"], "confidence": it["pct"] / 100} for it in obj_items]
        )
    else:
        export_df = pd.DataFrame(
            [{"class": c, "count": v} for c, v in summary["counts"].items()]
        )
    e1.download_button("Export CSV", export_df.to_csv(index=False).encode(),
                       file_name="detections.csv", mime="text/csv", use_container_width=True)
    e2.download_button("Export JSON",
                       json.dumps({"summary": {k: summary[k] for k in
                                   ("avg", "total", "n_classes", "time_ms", "counts")}},
                                  indent=2).encode(),
                       file_name="detections.json", mime="application/json", use_container_width=True)
    report = (f"Aperture detection report\n"
              f"Source: {active['name'] if active else '-'}\n"
              f"Total objects: {total}\nObject classes: {n_classes}\n"
              f"Avg confidence: {avg*100:.1f}%\nProcessing time: {time_ms:.0f} ms\n\n"
              f"Per class:\n" + "\n".join(f"  {c}: {v}" for c, v in summary["counts"].items()))
    e3.download_button("Export report", report.encode(),
                       file_name="report.txt", mime="text/plain", use_container_width=True)

    # Auto-scroll to results after a fresh detection
    if st.session_state.pop("scroll_to_results", False):
        components.html(
            """
            <script>
                const doc = window.parent.document;
                const el = doc.querySelector('#results-anchor');
                if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            </script>
            """,
            height=0,
        )
