"""Home page — project overview and navigation."""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Adverse Weather Object Detection",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
css = (Path(__file__).parent / "assets" / "style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Adverse Weather Object Detection</div>
    <div class="hero-subtitle">
        Deep learning model trained to detect objects under fog, rain, snow, and night conditions.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics overview ──────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">6</div>
        <div class="metric-label">Object Classes</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">4</div>
        <div class="metric-label">Weather Conditions</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">~40K</div>
        <div class="metric-label">Training Images</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">640px</div>
        <div class="metric-label">Input Resolution</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two columns: pipeline + classes ──────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### Training Pipeline")
    st.markdown("""
    <div class="info-card">
        <h4>Phase 1 — Domain Adaptation</h4>
        <p style="color:#8892b0; font-size:0.9rem; line-height:1.7;">
            Models are first pre-trained on <b>COCO</b> (general objects), then adapted to
            the driving domain using <b>BDD100K</b> (30K condition-aware subset), and finally
            fine-tuned on <b>XWOD</b> (adverse weather driving dataset).
        </p>
        <div style="display:flex; gap:0.5rem; align-items:center; margin-top:1rem; font-size:0.85rem; color:#6b7394;">
            <span style="background:#1e3a5f; padding:0.3rem 0.8rem; border-radius:20px;">COCO</span>
            <span>→</span>
            <span style="background:#1e3a5f; padding:0.3rem 0.8rem; border-radius:20px;">BDD100K</span>
            <span>→</span>
            <span style="background:#1e3a5f; padding:0.3rem 0.8rem; border-radius:20px;">XWOD</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Evaluation Datasets</h4>
        <p style="color:#8892b0; font-size:0.9rem; line-height:1.7;">
            Final models are evaluated on <b>XWOD test</b>, <b>DAWN</b> (zero-shot),
            <b>ACDC</b>, and <b>BDD100K val</b> to measure generalization across
            unseen adverse weather conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Detected Classes")
    classes = [
        ("🚶", "Person",     "#FF6B4B"),
        ("🚲", "Bicycle",    "#4ECD4E"),
        ("🚗", "Car",        "#37AFEE"),
        ("🏍️", "Motorcycle", "#E664E6"),
        ("🚌", "Bus",        "#FFD737"),
        ("🚚", "Truck",      "#DC64DC"),
    ]
    for icon, name, color in classes:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.8rem; padding:0.6rem 0;
                    border-bottom:1px solid #1e2a4a;">
            <span style="font-size:1.3rem;">{icon}</span>
            <span style="color:#e8eaf6; font-weight:500;">{name}</span>
            <span style="margin-left:auto; width:12px; height:12px; border-radius:50%;
                         background:{color}; display:inline-block;"></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Supported Conditions</h4>
        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
            <span class="weather-badge">🌫️ Fog</span>
            <span class="weather-badge">🌧️ Rain</span>
            <span class="weather-badge">❄️ Snow</span>
            <span class="weather-badge">🌙 Night</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── CTA ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Navigate to **Detection Demo** in the sidebar to try the model.", icon="🎯")
