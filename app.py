"""
Traffic Congestion Analyzer — Streamlit App
Uses YOLO (ultralytics) for vehicle detection, PCU conversion, VCR calculation,
and Google Gemini AI for a human-readable traffic report.
"""

import os
import math
import base64
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from io import BytesIO
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Page Config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Russo+One&family=Share+Tech+Mono&family=Barlow+Condensed:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    :root {
        --bg:       #0A0A08;
        --bg2:      #111110;
        --amber:    #F0A500;
        --amber-d:  rgba(240,165,0,0.14);
        --amber-h:  rgba(240,165,0,0.55);
        --border:   rgba(240,165,0,0.16);
        --border-h: rgba(240,165,0,0.48);
        --glow:     0 0 24px rgba(240,165,0,0.12), 0 0 0 1px rgba(240,165,0,0.22);
        --text:     #E8E4D9;
        --muted:    #4E4C46;
    }

    @keyframes fadeUp   { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
    @keyframes blink    { 0%,100% { opacity:1; } 50% { opacity:0; } }
    @keyframes scanline { 0% { top:-4px; } 100% { top:100%; } }

    html, body, [class*="css"] { font-family: 'Barlow Condensed', sans-serif !important; }

    /* ── Grid-texture background ── */
    .stApp {
        background-color: var(--bg) !important;
        background-image:
            linear-gradient(rgba(240,165,0,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(240,165,0,0.03) 1px, transparent 1px);
        background-size: 44px 44px;
        color: var(--text) !important;
    }
    /* Scanline sweep */
    .stApp::before {
        content: '';
        position: fixed; left: 0; right: 0; top: -4px; height: 3px; z-index: 9999;
        background: linear-gradient(transparent, rgba(240,165,0,0.18), transparent);
        animation: scanline 8s linear infinite;
        pointer-events: none;
    }

    /* ── Main header ── */
    .main-header {
        font-family: 'Russo One', sans-serif !important;
        font-size: 2.6rem; font-weight: 400; letter-spacing: 2px;
        color: var(--amber) !important;
        margin-bottom: 0.1rem; line-height: 1.05;
        animation: fadeUp 0.5s ease both;
        text-transform: uppercase;
    }
    .sub-header {
        font-family: 'Share Tech Mono', monospace !important;
        color: var(--muted); font-size: 0.78rem;
        letter-spacing: 1.5px; margin-bottom: 2rem;
        animation: fadeUp 0.5s 0.07s ease both;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        border-radius: 0; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color .25s ease, box-shadow .25s ease, background .25s ease;
        cursor: default; animation: fadeUp 0.45s ease both;
    }
    .metric-card:hover {
        border-color: var(--border-h);
        border-left-color: var(--amber);
        box-shadow: var(--glow);
        background: rgba(240,165,0,0.04);
    }
    .metric-card h4 {
        font-family: 'Share Tech Mono', monospace !important;
        color: var(--muted); font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.4rem;
    }

    /* ── Gauge wrapper ── */
    .gauge-wrapper {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-top: 3px solid var(--amber);
        border-radius: 0; padding: 1.4rem;
        text-align: center; margin-bottom: 1rem;
        transition: border-color .25s ease, box-shadow .25s ease;
    }
    .gauge-wrapper:hover {
        border-color: var(--border-h);
        border-top-color: var(--amber);
        box-shadow: var(--glow);
    }

    /* ── Report box ── */
    .report-box {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        border-radius: 0; padding: 1.4rem 1.8rem;
        line-height: 1.8; font-size: 0.92rem; color: #B8B4A9;
        transition: border-color .25s ease, box-shadow .25s ease;
    }
    .report-box:hover { border-color: var(--border-h); box-shadow: var(--glow); }

    /* ── Status badges ── */
    .status-badge {
        font-family: 'Share Tech Mono', monospace !important;
        display: inline-block; padding: 4px 14px; border-radius: 0;
        font-size: 0.68rem; font-weight: 400; letter-spacing: 1.5px; text-transform: uppercase;
    }
    .badge-green  { background: rgba(240,165,0,0.12); color: #F0A500; border: 1px solid rgba(240,165,0,0.35); }
    .badge-yellow { background: rgba(240,165,0,0.08); color: #C88800; border: 1px solid rgba(240,165,0,0.25); }
    .badge-orange { background: rgba(200,100,0,0.1);  color: #C86400; border: 1px solid rgba(200,100,0,0.3); }
    .badge-red    { background: rgba(180,40,40,0.12); color: #C04040; border: 1px solid rgba(180,40,40,0.3); }

    /* ── Upload zone ── */
    section[data-testid="stFileUploadDropzone"] {
        background: var(--bg2) !important;
        border: 2px dashed rgba(240,165,0,0.22) !important;
        border-radius: 0 !important;
        transition: all .25s ease !important;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(240,165,0,0.55) !important;
        box-shadow: var(--glow) !important;
        background: rgba(240,165,0,0.03) !important;
    }

    /* ── Metric containers ── */
    [data-testid="metric-container"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--amber) !important;
        border-radius: 0 !important; padding: 0.9rem 1.1rem !important;
        transition: all .25s ease !important;
    }
    [data-testid="metric-container"]:hover {
        border-color: var(--border-h) !important;
        box-shadow: var(--glow) !important;
        background: rgba(240,165,0,0.03) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace !important;
        font-weight: 400 !important; color: var(--amber) !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] { font-family: 'Share Tech Mono', monospace !important; color: var(--muted) !important; }

    /* ── Primary button ── */
    .stButton > button {
        background: var(--amber) !important; color: #0A0A08 !important;
        border: none !important; border-radius: 0 !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Russo One', sans-serif !important; font-weight: 400 !important;
        font-size: 0.9rem !important; letter-spacing: 2px !important;
        text-transform: uppercase !important;
        transition: all .2s cubic-bezier(.4,0,.2,1) !important;
        box-shadow: 0 0 18px rgba(240,165,0,0.25) !important;
    }
    .stButton > button:hover {
        background: #FFB800 !important; transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(240,165,0,0.4), 0 0 0 1px rgba(240,165,0,0.5) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important; color: var(--amber) !important;
        border: 1px solid var(--border) !important; border-left: 3px solid var(--amber) !important;
        border-radius: 0 !important;
        font-family: 'Share Tech Mono', monospace !important; font-weight: 400 !important;
        letter-spacing: 1px !important;
        transition: all .22s ease !important;
    }
    .stDownloadButton > button:hover {
        background: var(--amber-d) !important;
        border-color: var(--amber-h) !important;
        box-shadow: var(--glow) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: var(--amber) !important; border-radius: 0 !important;
        box-shadow: 0 0 8px rgba(240,165,0,0.4);
    }
    .stProgress > div { background: rgba(240,165,0,0.1) !important; border-radius: 0 !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #050504 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div { padding: 1.5rem 1.1rem !important; }
    .sidebar-logo {
        font-family: 'Russo One', sans-serif !important;
        font-size: 1.05rem; color: var(--amber); letter-spacing: 3px;
        text-transform: uppercase; margin-bottom: 0.1rem;
    }
    .sidebar-version {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.62rem; color: var(--muted); margin-bottom: 1.5rem; letter-spacing: 1px;
    }
    .sidebar-label {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.58rem; font-weight: 400; text-transform: uppercase;
        letter-spacing: 2.5px; color: var(--muted);
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); display: block;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div > div { background: var(--amber) !important; }
    [data-testid="stSlider"] > div > div > div > div { background: #0A0A08 !important; border: 2px solid var(--amber) !important; }

    /* ── Text inputs ── */
    .stTextInput > div > div > input {
        background: var(--bg2) !important; border: 1px solid var(--border) !important;
        border-left: 2px solid var(--amber) !important;
        border-radius: 0 !important; color: var(--text) !important;
        font-family: 'Share Tech Mono', monospace !important;
        transition: all .2s ease !important; letter-spacing: 0.5px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--amber-h) !important;
        box-shadow: var(--glow) !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--bg2) !important; border: 1px solid var(--border) !important;
        border-left: 3px solid var(--amber) !important; border-radius: 0 !important;
        transition: all .25s ease !important;
    }
    [data-testid="stExpander"]:hover { border-color: var(--border-h) !important; box-shadow: var(--glow) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: rgba(240,165,0,0.3); border-radius: 0; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(240,165,0,0.6); }

    hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
    .stSuccess, .stAlert { border-radius: 0 !important; border-left: 3px solid var(--amber) !important; }
    div[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — lazy imports (avoid crashing before pip install)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config() -> dict:
    """Load persisted settings from config.json next to app.py."""
    try:
        import json
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(data: dict):
    """Merge and persist settings to config.json."""
    try:
        import json
        existing = load_config()
        existing.update(data)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(existing, f)
    except Exception:
        pass



@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except ImportError:
        st.error("ultralytics is not installed. Run: pip install ultralytics")
        return None


def generate_ai_report(
    api_key: str,
    provider: str,
    vcr: float,
    status_label: str,
    vehicle_counts: dict,
    total_pcu: float,
    duration_sec: float,
    timeout_sec: int = 30,
) -> str:
    """Generate a traffic report using Gemini, OpenAI, or Claude.
    Runs in a thread with timeout_sec timeout to prevent hangs.
    """
    import concurrent.futures

    def _call():
        vc_lines = "\n".join(
            f"  - {cls}: {int(count)} unique vehicles (PCU factor: {class_to_pcu(cls):.3f})"
            for cls, count in sorted(vehicle_counts.items(), key=lambda x: -x[1])
            if count > 0
        )
        prompt = (
            f"You are a professional traffic engineer writing an executive summary.\n\n"
            f"## Traffic Analysis Data\n"
            f"- VCR: {vcr:.4f} | Status: {status_label} | Road used: {vcr*100:.1f}%\n"
            f"- Road Capacity: {ROAD_CAPACITY_PCU} PCU/hr | Duration: {duration_sec:.1f}s\n"
            f"- Total PCU: {total_pcu:.1f}\n"
            f"- Vehicle breakdown:\n{vc_lines}\n\n"
            f"PCU factors: Cars 1.0, Motorcycles 0.5, Tricycles 0.535, Buses/Trucks 3.0, Jeepneys 1.5\n\n"
            f"Write a concise professional traffic engineering report (3 paragraphs): "
            f"(1) situation & VCR meaning, (2) dominant vehicles & congestion contribution, "
            f"(3) practical recommendations. Use language accessible to city planners."
        )
        if provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model.generate_content(prompt).text
        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return resp.choices[0].message.content

        elif provider == "Claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        else:
            return f"⚠️ Unknown provider: {provider}"


# PCU equivalency factors per vehicle class
# Keys should match your YOLO model's class names (case-insensitive matching below).
PCU_FACTORS: dict[str, float] = {
    "car":          1.0,
    "motorcycle":   0.5,
    "tricycle":     0.535,  # ← your research factor
    "bus":          3.0,
    "truck":        3.0,
    "van":          1.5,
    "jeepney":      1.5,
    "bicycle":      0.3,
    "person":       0.2,
}
DEFAULT_PCU = 1.0          # fallback for unlisted classes
ROAD_CAPACITY_PCU = 3000   # PCU/hour — multi-lane urban road (was 1600, adjusted for real-world observation)

# BGR colors for bounding boxes (OpenCV uses BGR)
CLASS_COLORS_BGR: dict[str, tuple] = {
    "jeepney":    (180, 105, 255),   # hot pink
    "jeepneys":   (180, 105, 255),
    "sedan":      (255, 160,  40),   # dodger blue
    "car":        (255, 160,  40),
    "suv":        (50,  205,  50),   # lime green
    "bus":        (0,   165, 255),   # orange
    "truck":      (60,   20, 220),   # crimson red
    "van":        (0,   215, 255),   # gold yellow
    "motorcycle": (211,   0, 148),   # purple
    "tricycle":   (0,   210, 210),   # cyan
    "bicycle":    (255, 255,  80),   # light yellow
    "person":     (150, 150, 150),   # gray
}
_DEFAULT_COLOR_BGR = (0, 165, 240)   # amber fallback

def _cls_color(cls_name: str) -> tuple:
    return CLASS_COLORS_BGR.get(cls_name.lower(), _DEFAULT_COLOR_BGR)



def class_to_pcu(class_name: str) -> float:
    """Return the PCU factor for a detected class."""
    return PCU_FACTORS.get(class_name.lower(), DEFAULT_PCU)


def compute_vcr(total_pcu: float, fps: float, total_frames: int) -> float:
    """
    Estimate VCR from accumulated PCU counts.
    We approximate 'volume' by summing average PCU counts across frames
    and scaling to vehicles-per-hour.
    """
    if fps <= 0 or total_frames <= 0:
        return 0.0
    duration_seconds = total_frames / fps
    # PCU/hour = average_pcu_per_frame * fps * 3600
    # But since we counted *detections per frame* (not unique vehicles),
    # we compute average instantaneous density, then use it as a proxy for flow.
    hourly_pcu = (total_pcu / total_frames) * fps * 3600
    return min(hourly_pcu / ROAD_CAPACITY_PCU, 2.0)  # cap at 2.0 (extreme congestion)


def vcr_status(vcr: float) -> tuple[str, str]:
    """Return (label, badge_class) for a VCR value."""
    if vcr < 0.4:
        return "Free Flow", "badge-green"
    elif vcr < 0.6:
        return "Stable Flow", "badge-yellow"
    elif vcr < 0.8:
        return "Moderate Congestion", "badge-orange"
    elif vcr < 1.0:
        return "Approaching Capacity", "badge-orange"
    else:
        return "Over Capacity / Congested", "badge-red"


# ─────────────────────────────────────────────────────────────────────────────
# Gauge SVG generator
# ─────────────────────────────────────────────────────────────────────────────

def make_live_pie_html(vehicle_counts: dict, vcr: float, slabel: str, sbadge: str) -> str:
    """
    SVG donut pie chart showing live vehicle class distribution + VCR badge.
    Renders in ~1ms (pure Python math, no Plotly overhead).
    """
    filtered = {cls: cnt for cls, cnt in vehicle_counts.items() if cnt > 0}
    total = sum(filtered.values())

    if not filtered or total == 0:
        return (
            '<div style="text-align:center;padding:1.5rem 0;font-family:\'Share Tech Mono\',monospace;'
            'color:#4E4C46;font-size:0.72rem;letter-spacing:1px">AWAITING DETECTIONS...</div>'
        )

    # Build SVG donut segments
    cx, cy, R, r = 85, 85, 68, 38
    segments = []
    angle = -90.0  # start from top
    legend_rows = []

    for cls, cnt in sorted(filtered.items(), key=lambda x: -x[1]):
        sweep = (cnt / total) * 360
        bgr = _cls_color(cls)           # BGR tuple
        rgb = (bgr[2], bgr[1], bgr[0])  # convert to RGB
        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        # Arc path
        a1, a2 = math.radians(angle), math.radians(angle + sweep)
        laf = 1 if sweep > 180 else 0
        ox1, oy1 = cx + R * math.cos(a1), cy + R * math.sin(a1)
        ox2, oy2 = cx + R * math.cos(a2), cy + R * math.sin(a2)
        ix1, iy1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        ix2, iy2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        path = (f"M {ox1:.1f} {oy1:.1f} A {R} {R} 0 {laf} 1 {ox2:.1f} {oy2:.1f} "
                f"L {ix2:.1f} {iy2:.1f} A {r} {r} 0 {laf} 0 {ix1:.1f} {iy1:.1f} Z")
        segments.append(f'<path d="{path}" fill="{color}" stroke="#0A0A08" stroke-width="1.5"/>')
        angle += sweep

        pct = cnt / total * 100
        legend_rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:1px;'
            f'flex-shrink:0;background:{color}"></span>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
            f'color:#9E9A94;white-space:nowrap">{cls.title()} '
            f'<span style="color:#F0A500">{int(cnt)}</span> '
            f'<span style="color:#4E4C46">({pct:.0f}%)</span></span></div>'
        )

    svg = (
        f'<svg viewBox="0 0 170 170" width="170" height="170" xmlns="http://www.w3.org/2000/svg">'
        + "".join(segments)
        + f'<circle cx="{cx}" cy="{cy}" r="{r - 2}" fill="#111110"/>'
        + f'<text x="{cx}" y="{cy - 7}" text-anchor="middle" fill="#F0A500" '
          f'font-family="Share Tech Mono,monospace" font-size="16">{vcr:.3f}</text>'
        + f'<text x="{cx}" y="{cy + 9}" text-anchor="middle" fill="#4E4C46" '
          f'font-family="Share Tech Mono,monospace" font-size="7">VCR</text>'
        + f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" fill="#4E4C46" '
          f'font-family="Share Tech Mono,monospace" font-size="7">{total} VEHICLES</text>'
        + '</svg>'
    )

    legend_html = "".join(legend_rows)
    return (
        f'<div style="display:flex;align-items:center;gap:14px;justify-content:center">'
        f'{svg}'
        f'<div style="text-align:left">{legend_html}</div>'
        f'</div>'
        f'<p style="text-align:center;margin-top:6px">'
        f'<span class="status-badge {sbadge}">{slabel}</span></p>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI Report

# ─────────────────────────────────────────────────────────────────────────────
# PDF Chart Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_bar_png(vehicle_counts: dict) -> bytes:
    """Horizontal bar chart of vehicle detections for embedding in PDF."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(vehicle_counts.items(), key=lambda x: x[1])
    labels = [c.title() for c, _ in items]
    values = [int(n) for _, n in items]
    fig, ax = plt.subplots(figsize=(6, max(2.2, len(labels) * 0.45)), facecolor="white")
    bars = ax.barh(labels, values, color="#0f172a", edgecolor="white", linewidth=0.4, height=0.55)
    ax.set_xlabel("Detections", fontsize=8, color="#64748b")
    ax.set_facecolor("#f8fafc")
    ax.tick_params(colors="#64748b", labelsize=7.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=7.5, color="#475569")
    ax.set_xlim(0, max(values) * 1.18)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _pdf_pie_png(vehicle_counts: dict) -> bytes:
    """Pie chart of PCU contributions for embedding in PDF."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(c, cnt * class_to_pcu(c)) for c, cnt in vehicle_counts.items() if cnt > 0]
    items.sort(key=lambda x: -x[1])
    labels = [c.title() for c, _ in items]
    values = [v for _, v in items]
    grays = [str(round(0.85 - i * (0.55 / max(len(items) - 1, 1)), 2)) for i in range(len(items))]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=grays, startangle=140,
        wedgeprops=dict(linewidth=0.6, edgecolor="white"),
        textprops=dict(fontsize=7.5, color="#1e293b"),
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
    ax.set_title("PCU Share", fontsize=9, color="#1e293b", pad=6)
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_bytes(
    vcr: float,
    status_label: str,
    vehicle_counts: dict,
    total_pcu: float,
    duration_sec: float,
    fps: float,
    report_text: str,
    video_name: str,
    frames_processed: int = 0,
) -> bytes:
    """Build a single-page PDF report using fpdf2. Returns raw bytes."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    import datetime

    AMBER  = (240, 165, 0)
    BLUE   = (40, 100, 220)
    DARK   = (18, 18, 16)
    MUTED  = (90, 90, 85)
    WHITE  = (255, 255, 255)
    ROW_A  = (245, 243, 235)
    GREEN  = (56, 142, 60)
    RED    = (198, 40, 40)

    # Status color
    _sc = {"Free Flow": GREEN, "Stable Flow": (245, 180, 0),
           "Approaching Capacity": (200, 100, 0), "Over Capacity": RED}
    status_color = _sc.get(status_label, MUTED)

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(*DARK)
            self.rect(0, 0, 210, 12, "F")
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*AMBER)
            self.set_y(2.5)
            self.cell(0, 7, "TRAFFIC CONGESTION ANALYSIS REPORT", align="C")
            self.set_y(12)

        def footer(self):
            self.set_y(-10)
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*MUTED)
            self.cell(0, 6,
                f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                f"Source: {video_name}  |  Traffic Congestion Analyzer",
                align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=False)   # single page — we manage layout manually
    pdf.add_page()
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    W = pdf.w - 24   # usable width
    LW = W * 0.48    # left column
    RW = W * 0.48    # right column
    GAP = W * 0.04
    LEFT_X  = 12
    RIGHT_X = 12 + LW + GAP * W / W  # ≈ LEFT_X + LW + small gap
    RIGHT_X = 12 + LW + 4

    def mini_header(text, x, w, color):
        pdf.set_x(x)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*WHITE)
        pdf.set_fill_color(*color)
        pdf.cell(w, 6, f"  {text}", fill=True, ln=True)
        pdf.ln(1)

    # ── Title row ──
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*DARK)
    pdf.cell(LW + 4, 9, "Traffic Analysis Report", ln=False)
    # VCR badge — right-aligned on same row
    badge_x = RIGHT_X
    pdf.set_x(badge_x)
    pdf.set_fill_color(*status_color)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(RW, 9, f"  {status_label}  |  VCR {vcr:.4f}", fill=True, align="C", ln=True)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(W, 4.5, f"Source: {video_name}", ln=True)
    pdf.ln(3)

    # ── Save Y — start two columns ──
    col_start_y = pdf.get_y()

    # ════ LEFT COLUMN — Key Metrics ════
    pdf.set_xy(LEFT_X, col_start_y)
    mini_header("KEY METRICS", LEFT_X, LW, DARK)
    metrics = [
        ("VCR",                  f"{vcr:.4f}"),
        ("Capacity Utilised",    f"{vcr * 100:.1f}%"),
        ("Capacity Baseline",    "3,000 PCU/hr"),
        ("Total PCU",            f"{total_pcu:.1f}"),
        ("Duration",             f"{duration_sec:.1f}s"),
        ("Frame Rate",           f"{fps:.1f} fps"),
        ("Frames Analyzed",      f"{frames_processed:,}"),
    ]
    pdf.set_font("Helvetica", "", 7)
    for idx, (lbl, val) in enumerate(metrics):
        pdf.set_x(LEFT_X)
        pdf.set_fill_color(*(ROW_A if idx % 2 == 0 else WHITE))
        pdf.set_text_color(*MUTED)
        pdf.cell(LW * 0.62, 5.5, f"  {lbl}", fill=True, border="B")
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(LW * 0.38, 5.5, val, fill=True, border="B", align="R", ln=True)
        pdf.set_font("Helvetica", "", 7)
    pdf.ln(3)

    # VCR guide (compact, inline)
    pdf.set_x(LEFT_X)
    mini_header("VCR GUIDE", LEFT_X, LW, (80, 60, 150))
    guide = [
        ("< 0.50", "Free Flow",    GREEN),
        ("0.50-0.75", "Stable",    (180, 140, 0)),
        ("0.75-1.00", "Congested", (200, 100, 0)),
        ("> 1.00",   "Over Cap.",  RED),
    ]
    pdf.set_font("Helvetica", "", 6.5)
    for rng, status, col in guide:
        pdf.set_x(LEFT_X)
        pdf.set_fill_color(*WHITE)
        pdf.set_text_color(*DARK)
        pdf.cell(LW * 0.30, 5, f"  {rng}",   border="B", fill=True)
        pdf.set_text_color(*col)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.cell(LW * 0.70, 5, f"  {status}", border="B", fill=True, ln=True)
        pdf.set_font("Helvetica", "", 6.5)

    left_col_end_y = pdf.get_y()

    # ════ RIGHT COLUMN — Vehicle Table ════
    pdf.set_xy(RIGHT_X, col_start_y)
    mini_header("VEHICLE BREAKDOWN", RIGHT_X, RW, BLUE)
    sorted_vc = sorted(vehicle_counts.items(), key=lambda x: -x[1])
    t_cols = [RW * 0.40, RW * 0.20, RW * 0.20, RW * 0.20]
    # Header
    pdf.set_x(RIGHT_X)
    pdf.set_font("Helvetica", "B", 6.5)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(*WHITE)
    for h, w in zip(["Class", "Count", "PCU Fac.", "Total"], t_cols):
        pdf.cell(w, 5.5, f" {h}", border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 6.5)
    for idx2, (cls, cnt) in enumerate(sorted_vc):
        pf = class_to_pcu(cls)
        pdf.set_x(RIGHT_X)
        pdf.set_fill_color(*(ROW_A if idx2 % 2 == 0 else WHITE))
        pdf.set_text_color(*DARK)
        cls_d = cls.title()[:18] + ("..." if len(cls) > 18 else "")
        pdf.cell(t_cols[0], 5.5, f" {cls_d}",        border=1, fill=True)
        pdf.cell(t_cols[1], 5.5, f" {int(cnt)}",      border=1, fill=True, align="C")
        pdf.cell(t_cols[2], 5.5, f" {pf:.2f}",        border=1, fill=True, align="C")
        pdf.cell(t_cols[3], 5.5, f" {cnt*pf:.1f}",    border=1, fill=True, align="C")
        pdf.ln()
        if pdf.get_y() > 240:   # safety: don't overflow page
            break

    right_col_end_y = pdf.get_y()

    # ── AI Report — below both columns ──
    below_y = max(left_col_end_y, right_col_end_y) + 4
    if below_y < 255 and report_text and not report_text.startswith("\u26a0"):
        pdf.set_xy(LEFT_X, below_y)
        mini_header("AI-GENERATED SUMMARY", LEFT_X, W, DARK)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*DARK)
        clean = (report_text
                 .replace("\u2014", "-").replace("\u2013", "-")
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("\u201c", '"').replace("\u201d", '"')
                 .replace("**", "").replace("*", "")
                 .replace("##", "").replace("#", ""))
        clean = clean.encode("latin-1", errors="replace").decode("latin-1")
        # Truncate to fit remaining space
        max_chars = 900
        if len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0] + "..."
        pdf.set_x(LEFT_X)
        pdf.multi_cell(W, 4.8, clean)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Core video processing
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_b64jpeg(rgb_array, quality: int = 70, max_width: int = 320) -> str:
    """Convert RGB numpy array to a small base64 JPEG thumbnail.
    Resizes to max_width before encoding to keep payloads small (~5-8 KB each).
    """
    from PIL import Image as _PILImg
    img = _PILImg.fromarray(rgb_array)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), _PILImg.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def process_video(
    video_path: str,
    model,
    conf_threshold: float,
    frame_skip: int,
    video_placeholder,
    gauge_placeholder,
    metrics_placeholder,
    status_placeholder,
    tracker: str = "ByteTrack (Fast)",
    imgsz: int = 640,
) -> tuple:
    """
    Stream through the video with YOLO + tracker.
    Tracker: 'ByteTrack (Fast)' uses model.track() (no embedder overhead, ~2x faster).
             'DeepSORT (Accurate)' uses deep-sort-realtime with mobilenet embedder.
    Returns (vehicle_counts, total_pcu, fps, duration_sec, frames_processed,
             final_vcr, vcr_timeline, frame_store_b64).
    frame_store_b64: list of (label_str, b64_jpeg_str).
    """
    use_bytetrack = tracker.startswith("ByteTrack")

    # ── Init DeepSORT only if needed ──
    ds_tracker = None
    if not use_bytetrack:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            ds_tracker = DeepSort(
                max_age=5, n_init=3, nms_max_overlap=0.7,
                max_cosine_distance=0.4, embedder="mobilenet",
                half=True, bgr=True, embedder_gpu=False,
            )
        except Exception:
            use_bytetrack = True   # fall back to ByteTrack if DeepSORT unavailable

    cap = cv2.VideoCapture(video_path)
    fps              = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec     = total_vid_frames / fps

    unique_track_ids: dict[str, set]   = defaultdict(set)
    vehicle_counts:   dict[str, float] = defaultdict(float)

    cumulative_pcu   = 0.0
    frames_processed = 0
    frame_idx        = 0
    vcr_timeline:    list[tuple[float, float]] = []

    # ── Frame store: save EVERY analyzed frame as a small thumbnail ──
    # Thumbnails are resized to 320px wide (~5 KB each) inside _rgb_to_b64jpeg,
    # so 500 analyzed frames ≈ 2.5 MB — well within Streamlit session state limits.
    # The gallery paginates (12/page) so the display stays fast regardless of count.
    frame_store_b64: list[tuple[str, str]] = []

    progress_bar = st.progress(0, text="Analysing video…")

    LABEL_SCALE = 0.70     # bigger, readable labels
    LABEL_THICK = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frames_processed += 1
        frame_pcu = 0.0
        annotated = frame.copy()

        if use_bytetrack:
            # ── ByteTrack via ultralytics model.track() ──
            results = model.track(
                frame, conf=conf_threshold, persist=True,
                verbose=False, imgsz=imgsz, tracker="bytetrack.yaml",
            )
            result = results[0]
            if result.boxes.id is not None:
                for box_xyxy, tid, cls_id_t in zip(
                    result.boxes.xyxy,
                    result.boxes.id.int(),
                    result.boxes.cls.int(),
                ):
                    x1, y1, x2, y2 = map(int, box_xyxy.tolist())
                    track_id = int(tid)
                    cls_name = model.names.get(int(cls_id_t), f"class_{int(cls_id_t)}")
                    color    = _cls_color(cls_name)
                    pcu      = class_to_pcu(cls_name)
                    frame_pcu += pcu
                    unique_track_ids[cls_name].add(track_id)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"#{track_id} {cls_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (10, 10, 8), LABEL_THICK, cv2.LINE_AA)
            live_vc = {cls: len(ids) for cls, ids in unique_track_ids.items()}

        else:
            # ── DeepSORT ──
            results = model(frame, conf=conf_threshold, verbose=False, imgsz=imgsz)
            result  = results[0]
            raw_dets = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_dets.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0])))

            tracks = ds_tracker.update_tracks(raw_dets, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                cls_id = track.get_det_class()
                if cls_id is None:
                    continue
                cls_name = model.names.get(int(cls_id), f"class_{cls_id}")
                color    = _cls_color(cls_name)
                pcu      = class_to_pcu(cls_name)
                frame_pcu += pcu
                track_id  = track.track_id
                unique_track_ids[cls_name].add(track_id)

                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {cls_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (10, 10, 8), LABEL_THICK, cv2.LINE_AA)
            live_vc = {cls: len(ids) for cls, ids in unique_track_ids.items()}

        cumulative_pcu += frame_pcu
        live_vcr = compute_vcr(cumulative_pcu, fps / frame_skip, frames_processed)
        slabel, sbadge = vcr_status(live_vcr)
        vcr_timeline.append((frame_idx / fps, live_vcr))

        # ── Convert + store frame (every analyzed frame) ──
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ts_label = f"{frame_idx / fps:.1f}s  (frame {frame_idx})"
        frame_store_b64.append((ts_label, _rgb_to_b64jpeg(rgb, quality=70)))

        # ── Live UI updates (throttled) ──
        # Streamlit Cloud throttles WebSocket messages: if we push every processed
        # frame the browser only receives the last one. Throttle to every 5 frames
        # so the live feed actually animates instead of showing a single frozen frame.
        _UI_EVERY = 5
        if frames_processed % _UI_EVERY == 1 or frames_processed == 1:
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            gauge_placeholder.markdown(
                f'<div class="gauge-wrapper">'
                f'<p style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;'
                f'font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem">'
                f'VEHICLE DISTRIBUTION · VCR</p>'
                + make_live_pie_html(live_vc, live_vcr, slabel, sbadge)
                + '</div>',
                unsafe_allow_html=True,
            )

            top_vehicles = sorted(live_vc.items(), key=lambda x: -x[1])[:5]
            track_label  = f"Unique Vehicles ({tracker.split()[0]})"
            vc_html = "".join(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid rgba(240,165,0,0.1);">'
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;color:#9E9A94;font-size:0.88rem">{cls.title()}</span>'
                f'<span style="font-family:\'Share Tech Mono\',monospace;color:#F0A500">{int(cnt)}</span></div>'
                for cls, cnt in top_vehicles
            )
            metrics_placeholder.markdown(
                f'<div class="metric-card">'
                f'<h4>LIVE METRICS</h4>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.8rem">'
                f'<div><div style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;font-size:0.62rem;letter-spacing:1.5px">VCR</div>'
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.5rem;color:#F0A500">{live_vcr:.3f}</div></div>'
                f'<div><div style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;font-size:0.62rem;letter-spacing:1.5px">PCU TOTAL</div>'
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.5rem;color:#F0A500">{cumulative_pcu:.0f}</div></div>'
                f'</div>'
                f'<h4 style="margin-top:0.5rem">{track_label}</h4>'
                f"{vc_html}</div>",
                unsafe_allow_html=True,
            )

        # Progress bar + status always update (lightweight text only)
        progress_bar.progress(
            min(frame_idx / max(total_vid_frames, 1), 1.0),
            text=f"Frame {frame_idx}/{total_vid_frames}",
        )
        status_placeholder.markdown(
            f'<p style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;font-size:0.75rem">'
            f'&#9201; {frame_idx/fps:.1f}s / {duration_sec:.1f}s &nbsp;|&nbsp; '
            f'Frames analysed: {frames_processed}</p>',
            unsafe_allow_html=True,
        )

    cap.release()
    progress_bar.empty()

    final_counts = {cls: len(ids) for cls, ids in unique_track_ids.items()} if unique_track_ids else dict(vehicle_counts)
    final_vcr    = compute_vcr(cumulative_pcu, fps / frame_skip, max(frames_processed, 1))
    return final_counts, cumulative_pcu, fps, duration_sec, frames_processed, final_vcr, vcr_timeline, frame_store_b64


def main():
    # ── Session Persistence Init ──
    if "sessions" not in st.session_state:
        st.session_state.sessions = []   # list of dicts, newest first

    # ── Header ──
    st.markdown('<h1 class="main-header">🚦 Traffic Congestion Analyzer</h1>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sub-header">YOLO-powered vehicle detection · PCU conversion · VCR analysis · AI-generated traffic report</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar — configuration ──
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🚦 TrafficAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-version">v2.0 · Gemini + YOLO</div>', unsafe_allow_html=True)

        st.markdown('<span class="sidebar-label">Model</span>', unsafe_allow_html=True)

        # ── Auto-detect .pt files in the app directory ──
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        _pt_files = sorted(f for f in os.listdir(_app_dir) if f.endswith(".pt"))

        if _pt_files:
            _mcol1, _mcol2 = st.columns([5, 1])
            model_path = _mcol1.selectbox(
                "Select model", options=_pt_files, index=0,
                label_visibility="collapsed",
            )
            # Align trash button with selectbox using CSS
            _mcol2.markdown('<div style="margin-top:0.45rem"></div>', unsafe_allow_html=True)
            if _mcol2.button("🗑", key="del_model", help="Delete selected model", use_container_width=True):
                _del_path = os.path.join(_app_dir, model_path)
                try:
                    os.unlink(_del_path)
                    st.success(f"Deleted {model_path}")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not delete: {_e}")
        else:
            model_path = st.text_input(
                "YOLO model path", value="exp-2.pt",
                label_visibility="collapsed",
                help="No .pt files found. Place your model in the same folder as app.py.",
            )

        # ── Upload a new model ──
        uploaded_model = st.file_uploader(
            "Upload a new .pt model", type=["pt"],
            help="Drop any YOLO .pt file here.",
            label_visibility="visible",
        )
        if uploaded_model is not None:
            _save_path = os.path.join(_app_dir, uploaded_model.name)
            with open(_save_path, "wb") as _f:
                _f.write(uploaded_model.getbuffer())
            st.success(f"✅ Saved **{uploaded_model.name}**")
            model_path = _save_path

        st.markdown('<span class="sidebar-label">AI Report Provider</span>', unsafe_allow_html=True)
        ai_provider = st.selectbox(
            "AI Provider", ["Gemini", "OpenAI", "Claude"], index=0,
            label_visibility="collapsed",
        )
        _placeholders = {"Gemini": "AIza...", "OpenAI": "sk-...", "Claude": "sk-ant-..."}
        _help_urls = {
            "Gemini": "aistudio.google.com (free)",
            "OpenAI": "platform.openai.com",
            "Claude": "console.anthropic.com",
        }

        # ── Load saved key for current provider ──
        _cfg = load_config()
        _saved_key = _cfg.get(f"api_key_{ai_provider}", "")

        api_key = st.text_input(
            f"{ai_provider} API Key",
            value=_saved_key,
            type="password",
            label_visibility="collapsed",
            placeholder=_placeholders[ai_provider],
            help=f"Get your key at {_help_urls[ai_provider]}",
        )

        # Save / forget key
        _sk_col1, _sk_col2 = st.columns(2)
        if _sk_col1.button("💾 Save Key", key="save_key", use_container_width=True,
                           help="Save this key locally so you don't need to re-enter it",
                           disabled=not api_key):
            save_config({f"api_key_{ai_provider}": api_key})
            st.success("Key saved!")
        if _sk_col2.button("🗑 Forget Key", key="del_key", use_container_width=True,
                           help="Remove saved key"):
            save_config({f"api_key_{ai_provider}": ""})
            st.rerun()

        # ── API Key Status Indicator ──
        if api_key:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:7px;padding:6px 10px;'
                f'background:rgba(76,175,80,0.10);border:1px solid rgba(76,175,80,0.30);'
                f'border-radius:6px;margin-bottom:0.4rem">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:#4caf50;'
                f'box-shadow:0 0 6px #4caf50;flex-shrink:0"></span>'
                f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;'
                f'color:#81c784">{ai_provider} key loaded{" (saved)" if _saved_key else ""} ✓</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:7px;padding:6px 10px;'
                'background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.25);'
                'border-radius:6px;margin-bottom:0.4rem">'
                '<span style="width:8px;height:8px;border-radius:50%;background:#f44336;flex-shrink:0"></span>'
                '<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;'
                'color:#ef9a9a">No key — AI report disabled</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<span class="sidebar-label">Detection & Tracker</span>', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.95, 0.40, 0.05)
        frame_skip = st.slider(
            "Process every N frames", 1, 15, 4,
            help="Higher = faster. 4 processes every 4th frame for a good speed/accuracy balance.",
        )
        tracker_choice = st.selectbox(
            "Tracker",
            ["ByteTrack (Fast)", "DeepSORT (Accurate)"],
            index=0,
            help="ByteTrack: 2× faster, built-in. DeepSORT: uses appearance embedder, more accurate but slower.",
            label_visibility="collapsed",
        )
        analysis_imgsz = st.select_slider(
            "Analysis resolution",
            options=[320, 416, 480, 640],
            value=320,
            help="Lower = faster analysis, less detail. 320 is best for speed on Cloud.",
        )


        st.markdown('<span class="sidebar-label">PCU Factors</span>', unsafe_allow_html=True)
        for k, v in PCU_FACTORS.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">'
                f'<span style="color:#64748b">{k.title()}</span>'
                f'<span style="color:#94a3b8;font-weight:600">{v}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:rgba(59,130,246,0.1);'
            f'border-radius:8px;border:1px solid rgba(59,130,246,0.2);font-size:0.78rem;color:#60a5fa;">'
            f'🛣️ Capacity: <strong>{ROAD_CAPACITY_PCU} PCU/hr</strong></div>',
            unsafe_allow_html=True,
        )

    # ── File Uploader ──
    st.markdown("### 📁 Upload Traffic Video")
    uploaded = st.file_uploader(
        "Drag & drop your MP4 video here",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported: MP4, AVI, MOV, MKV",
    )

    if uploaded is None:
        st.markdown(
            """
            <div style="background:rgba(255,255,255,0.03);border:2px dashed rgba(100,181,246,0.2);
            border-radius:16px;padding:3rem;text-align:center;margin-top:1rem">
              <p style="font-size:2rem">🎬</p>
              <p style="color:#546e7a">Upload an MP4 traffic video to begin analysis</p>
              <p style="color:#37474f;font-size:0.8rem">Add your Gemini API key and model path in the sidebar ←</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Load Model ──
    if not Path(model_path).exists():
        st.error(
            f"Model file **{model_path}** not found. "
            "Place `exp-2.pt` in the same folder as `app.py` and update the path in the sidebar."
        )
        return

    model = load_model(model_path)
    if model is None:
        return

    # ── Model Verification Panel ──
    num_classes = len(model.names)
    class_names = [model.names[i] for i in sorted(model.names)]
    # Build color swatches HTML
    swatch_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin:2px 4px 2px 0;'
        f'font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:#9E9A94;">'  
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:1px;'
        f'background:rgb{tuple(reversed(_cls_color(c)))};flex-shrink:0"></span>'
        f'{c}</span>'
        for c in class_names
    )
    st.markdown(
        f'<div class="metric-card" style="margin-bottom:1rem">'
        f'<h4>ACTIVE MODEL</h4>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;color:#F0A500;'
        f'font-size:0.85rem;margin-bottom:0.4rem">📁 {os.path.basename(str(model_path))}</div>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;'
        f'font-size:0.65rem;margin-bottom:0.5rem">{num_classes} CLASSES DETECTED</div>'
        f'<div style="line-height:1.8">{swatch_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Save uploaded video to temp file ──
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp_path = tmp.name

    st.success(f"✅ **{uploaded.name}** uploaded — {uploaded.size / 1_048_576:.1f} MB")

    # ── Layout: left = video | right = gauge + report ──
    col_video, col_right = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown("#### 🎥 Live Detection Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_right:
        st.markdown("#### 📊 Real-Time Analysis")
        gauge_placeholder = st.empty()
        metrics_placeholder = st.empty()
        st.markdown("#### 🤖 AI Traffic Report")
        report_placeholder = st.empty()
        report_placeholder.markdown(
            '<div class="report-box" style="color:#546e7a;font-style:italic">'
            "AI report will appear here after analysis completes…"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Run Analysis ──
    if st.button("▶ Start Analysis", use_container_width=True):
        # NOTE: No st.spinner here — the spinner context creates a blocking
        # overlay that prevents video_placeholder from streaming frames live.
        # Status is shown via status_placeholder inside the feed column instead.
        status_placeholder.markdown(
            '<p style="font-family:\'Share Tech Mono\',monospace;color:#F0A500;font-size:0.75rem">'
            '⚙️ Analysis running…</p>',
            unsafe_allow_html=True,
        )
        (
            vehicle_counts, total_pcu, fps, duration_sec,
            frames_processed, final_vcr, vcr_timeline, frame_store_b64
        ) = process_video(
            tmp_path, model, conf_thresh, frame_skip,
            video_placeholder, gauge_placeholder,
            metrics_placeholder, status_placeholder,
            tracker=tracker_choice, imgsz=analysis_imgsz,
        )
        import datetime as _dt
        slabel, sbadge = vcr_status(final_vcr)

        # Generate AI report immediately (while still in button block)
        report_text = ""
        if api_key:
            with st.spinner(f"Generating report with {ai_provider}…"):
                report_text = generate_ai_report(
                    api_key, ai_provider, final_vcr, slabel,
                    vehicle_counts, total_pcu, duration_sec
                )

        # Save everything to session_state so re-runs don't lose it
        st.session_state["last_result"] = {
            "vehicle_counts": vehicle_counts,
            "total_pcu": total_pcu,
            "fps": fps,
            "duration_sec": duration_sec,
            "frames_processed": frames_processed,
            "final_vcr": final_vcr,
            "vcr_timeline": vcr_timeline,
            "frame_store_b64": frame_store_b64,
            "slabel": slabel,
            "sbadge": sbadge,
            "report_text": report_text,
            "video_name": uploaded.name,
            "api_key": api_key,
            "ai_provider": ai_provider,
        }
        # Also save to session history (only once per unique analysis)
        import datetime as _dt2
        _session_key = f"{uploaded.name}_{final_vcr:.4f}"
        if st.session_state.get("_last_session_key") != _session_key:
            st.session_state["_last_session_key"] = _session_key
            st.session_state.sessions.insert(0, {
                "id": len(st.session_state.sessions) + 1,
                "ts": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "video": uploaded.name,
                "vcr": final_vcr, "status": slabel, "badge": sbadge,
                "total_pcu": total_pcu, "duration": duration_sec,
                "vehicle_counts": vehicle_counts, "vcr_timeline": vcr_timeline,
                "frame_store_b64": frame_store_b64,
                "tracker": tracker_choice.split()[0],
            })
        # NO st.rerun() — session_state is readable in the same run below

    # ══════════════════════════════════════════════════════
    # RESULTS RENDERING  (reads session_state — survives any button click)
    # ══════════════════════════════════════════════════════
    r = st.session_state.get("last_result")
    if r:
        vehicle_counts  = r["vehicle_counts"]
        total_pcu       = r["total_pcu"]
        fps             = r["fps"]
        duration_sec    = r["duration_sec"]
        frames_processed= r["frames_processed"]
        final_vcr       = r["final_vcr"]
        vcr_timeline    = r["vcr_timeline"]
        frame_store_b64 = r["frame_store_b64"]
        slabel          = r["slabel"]
        sbadge          = r["sbadge"]
        report_text     = r["report_text"]
        video_name      = r["video_name"]

        st.success(f"✅ Analysis complete — VCR: **{final_vcr:.4f}** · Status: **{slabel}**")

        # Final pie chart into gauge placeholder
        gauge_placeholder.markdown(
            '<div class="gauge-wrapper">'
            '<p style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;'
            'font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem">'
            'FINAL VEHICLE DISTRIBUTION · VCR</p>'
            + make_live_pie_html(vehicle_counts, final_vcr, slabel, sbadge)
            + '</div>',
            unsafe_allow_html=True,
        )

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final VCR", f"{final_vcr:.4f}")
        m2.metric("Total PCU", f"{total_pcu:.0f}")
        m3.metric("Duration", f"{duration_sec:.1f}s")
        m4.metric("Frames Analyzed", f"{frames_processed:,}")

        # AI Report
        st.markdown("#### 🤖 AI Traffic Report")
        if not report_text:
            if not api_key:
                st.markdown(
                    '<div class="report-box">⚠️ Enter your <b>API Key</b> in the sidebar to generate the AI report.</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Key present but report is empty (failed or not run yet)
                _rpt_col1, _rpt_col2 = st.columns([2, 1])
                _rpt_col1.warning("Report generation failed or not yet run. Click to try again.")
                if _rpt_col2.button("🔄 Generate Report", key="regen_report", use_container_width=True):
                    _rpt_placeholder = st.empty()
                    with _rpt_placeholder:
                        with st.spinner(f"Generating with {r.get('ai_provider', 'Gemini')}…"):
                            new_report = generate_ai_report(
                                r["api_key"], r["ai_provider"], final_vcr, slabel,
                                vehicle_counts, total_pcu, duration_sec
                            )
                    _rpt_placeholder.empty()
                    st.session_state["last_result"]["report_text"] = new_report
                    st.markdown(
                        f'<div class="report-box">{new_report}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<div class="report-box">{report_text}</div>',
                unsafe_allow_html=True,
            )

        # Interactive Charts
        st.markdown("---")
        st.markdown("### 📈 Interactive Analysis Charts")
        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            if vcr_timeline:
                times = [t for t, _ in vcr_timeline]
                vcrs  = [v for _, v in vcr_timeline]
                fig_vcr = go.Figure()
                fig_vcr.add_hrect(y0=0,    y1=0.5,  fillcolor="rgba(76,175,80,0.08)",  line_width=0)
                fig_vcr.add_hrect(y0=0.5,  y1=0.75, fillcolor="rgba(255,193,7,0.08)",  line_width=0)
                fig_vcr.add_hrect(y0=0.75, y1=1.0,  fillcolor="rgba(255,152,0,0.08)",  line_width=0)
                fig_vcr.add_hrect(y0=1.0,  y1=2.0,  fillcolor="rgba(244,67,54,0.08)",  line_width=0)
                fig_vcr.add_hline(y=1.0, line_dash="dash", line_color="#ef5350",
                                  annotation_text="Capacity Limit", annotation_position="top right")
                fig_vcr.add_trace(go.Scatter(
                    x=times, y=vcrs, mode="lines",
                    line=dict(color="#F0A500", width=2.5, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(240,165,0,0.08)", name="VCR",
                    hovertemplate="Time: %{x:.1f}s<br>VCR: %{y:.3f}<extra></extra>",
                ))
                fig_vcr.update_layout(
                    title=dict(text="VCR Over Time", font=dict(color="#E8E4D9", size=14), x=0.01),
                    xaxis=dict(title="Time (s)", color="#4E4C46", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    yaxis=dict(title="VCR", color="#4E4C46", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    paper_bgcolor="rgba(17,17,16,0.8)", plot_bgcolor="rgba(17,17,16,0.6)",
                    font=dict(color="#9E9A94"), margin=dict(l=10,r=10,t=45,b=30), height=300,
                )
                st.plotly_chart(fig_vcr, use_container_width=True)
        with chart_col2:
            if vehicle_counts:
                sorted_vc = sorted(vehicle_counts.items(), key=lambda x: -x[1])
                fig_bar = go.Figure(go.Bar(
                    x=[c for c,_ in sorted_vc], y=[int(n) for _,n in sorted_vc],
                    marker=dict(color=[f"rgb{tuple(reversed(_cls_color(c)))}" for c,_ in sorted_vc]),
                    text=[int(n) for _,n in sorted_vc], textposition="outside",
                    hovertemplate="%{x}: %{y}<extra></extra>",
                ))
                fig_bar.update_layout(
                    title=dict(text="Vehicle Count by Class", font=dict(color="#E8E4D9", size=14), x=0.01),
                    xaxis=dict(color="#4E4C46"), yaxis=dict(color="#4E4C46", gridcolor="rgba(255,255,255,0.04)"),
                    paper_bgcolor="rgba(17,17,16,0.8)", plot_bgcolor="rgba(17,17,16,0.6)",
                    font=dict(color="#9E9A94"), margin=dict(l=10,r=10,t=45,b=10), height=300, bargap=0.35,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2, gap="large")
        with chart_col3:
            if vehicle_counts:
                pcu_rows = sorted(
                    [(c, round(n * class_to_pcu(c), 2)) for c, n in vehicle_counts.items() if n > 0],
                    key=lambda x: -x[1]
                )
                fig_donut = go.Figure(go.Pie(
                    labels=[p[0] for p in pcu_rows], values=[p[1] for p in pcu_rows],
                    hole=0.55, textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    hovertemplate="%{label}<br>PCU: %{value:.1f}<extra></extra>",
                ))
                fig_donut.update_layout(
                    title=dict(text="PCU Contribution", font=dict(color="#E8E4D9", size=14), x=0.01),
                    paper_bgcolor="rgba(17,17,16,0.8)", font=dict(color="#9E9A94"),
                    margin=dict(l=10, r=10, t=45, b=10), height=320,
                    annotations=[dict(text=f"{total_pcu:.0f}<br>PCU", x=0.5, y=0.5,
                                      font=dict(size=16, color="white"), showarrow=False)],
                )
                st.plotly_chart(fig_donut, use_container_width=True)
        with chart_col4:
            fig_ind = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=final_vcr,
                delta={"reference": 1.0, "increasing": {"color": "#ef5350"}, "decreasing": {"color": "#66bb6a"}},
                number={"valueformat": ".4f", "font": {"color": "white", "size": 28}},
                gauge={"axis": {"range": [0, 2.0]}, "bar": {"color": "#F0A500", "thickness": 0.25},
                       "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                       "steps": [{"range": [0, .5], "color": "rgba(76,175,80,0.2)"},
                                  {"range": [.5, .75], "color": "rgba(255,193,7,0.2)"},
                                  {"range": [.75, 1.0], "color": "rgba(255,152,0,0.2)"},
                                  {"range": [1.0, 2.0], "color": "rgba(244,67,54,0.2)"}],
                       "threshold": {"line": {"color": "#ef5350", "width": 3}, "thickness": 0.8, "value": 1.0}},
                title={"text": "Final VCR", "font": {"color": "#F0A500", "size": 14}},
            ))
            fig_ind.update_layout(paper_bgcolor="rgba(17,17,16,0.8)", font=dict(color="#9E9A94"),
                                  margin=dict(l=20, r=20, t=60, b=20), height=320)
            st.plotly_chart(fig_ind, use_container_width=True)

        # Raw data
        with st.expander("📋 Raw Detection Data"):
            raw_rows = [{"Vehicle Class": c, "Count": int(n), "PCU Factor": class_to_pcu(c),
                         "PCU Total": round(n * class_to_pcu(c), 2)}
                        for c, n in sorted(vehicle_counts.items(), key=lambda x: -x[1])]
            if raw_rows:
                st.dataframe(pd.DataFrame(raw_rows), use_container_width=True)

        # PDF Download
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        try:
            pdf_bytes = generate_pdf_bytes(
                vcr=final_vcr, status_label=slabel, vehicle_counts=vehicle_counts,
                total_pcu=total_pcu, duration_sec=duration_sec, fps=fps,
                report_text=report_text, video_name=video_name,
                frames_processed=frames_processed,
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"traffic_report_{video_name.rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as _pdf_err:
            st.warning(f"PDF generation error: {_pdf_err}")

        # ── Frame Gallery ──
        st.markdown("---")
        _n_frames = len(frame_store_b64)
        with st.expander(f"\U0001f39e Frame Gallery  ({_n_frames} frames captured)", expanded=False):
            if not frame_store_b64:
                st.info("No frames captured yet. Run analysis first.")
            else:
                view = st.radio(
                    "View", ["Grid", "Compact", "List"],
                    index=["Grid", "Compact", "List"].index(
                        st.session_state.get("gallery_view", "Grid")
                    ),
                    horizontal=True, label_visibility="collapsed", key="gallery_view",
                )

                _drag_tip = 'title="Drag to desktop to save, or click Save"'

                if view == "List":
                    # Scrollable list — all rows in one fixed-height div
                    list_rows = []
                    for i, (ts_label, b64) in enumerate(frame_store_b64):
                        data_uri = f'data:image/jpeg;base64,{b64}'
                        list_rows.append(
                            f'<div class="lrow" draggable="true">'
                            f'<img src="{data_uri}" style="width:56px;height:40px;object-fit:cover;'
                            f'border-radius:3px;border:1px solid rgba(240,165,0,0.15)" {_drag_tip}/>'
                            f'<div>'
                            f'<div style="font-family:\'Share Tech Mono\',monospace;color:#E8E4D9;font-size:0.78rem">frame_{i+1:05d}</div>'
                            f'<div style="font-family:\'Share Tech Mono\',monospace;color:#4E4C46;font-size:0.62rem;margin-top:2px">{ts_label}</div>'
                            f'</div>'
                            f'<a href="{data_uri}" download="frame_{i+1:05d}.jpg" class="lrow-dl">&#8595; Save</a>'
                            f'</div>'
                        )
                    st.markdown(
                        '<style>'
                        '.lrow{display:grid;grid-template-columns:56px 1fr auto;align-items:center;'
                        'gap:12px;padding:8px 12px;border-bottom:1px solid rgba(240,165,0,0.08);'
                        'background:rgba(17,17,16,0.6);transition:background 0.15s;cursor:grab;}'
                        '.lrow:hover{background:rgba(240,165,0,0.05);}'
                        '.lrow-dl{font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                        'color:#F0A500;text-decoration:none;padding:4px 10px;'
                        'border:1px solid rgba(240,165,0,0.3);border-radius:4px;white-space:nowrap;}'
                        '.lrow-dl:hover{background:rgba(240,165,0,0.12);}'
                        '</style>'
                        '<div style="border:1px solid rgba(240,165,0,0.12);border-radius:6px;overflow:hidden;margin-top:0.5rem">'
                        '<div style="display:grid;grid-template-columns:56px 1fr auto;gap:12px;padding:6px 12px;'
                        'background:rgba(240,165,0,0.05);border-bottom:1px solid rgba(240,165,0,0.12)">'
                        '<span style="font-size:0.62rem;color:#4E4C46;font-family:\'Share Tech Mono\',monospace;text-transform:uppercase">Preview</span>'
                        '<span style="font-size:0.62rem;color:#4E4C46;font-family:\'Share Tech Mono\',monospace;text-transform:uppercase">Name / Timestamp</span>'
                        '<span style="font-size:0.62rem;color:#4E4C46;font-family:\'Share Tech Mono\',monospace;text-transform:uppercase">Download</span>'
                        '</div>'
                        f'<div style="height:520px;overflow-y:auto;">'
                        + "".join(list_rows)
                        + "</div></div>",
                        unsafe_allow_html=True,
                    )

                else:
                    # Grid / Compact — all cells in one scrollable container
                    if view == "Grid":
                        cols_css, thumb_h, show_ts, scroll_h = "repeat(4,1fr)", 160, True, "560px"
                    else:
                        cols_css, thumb_h, show_ts, scroll_h = "repeat(6,1fr)", 100, False, "460px"

                    cells = []
                    for i, (ts_label, b64) in enumerate(frame_store_b64):
                        data_uri = f'data:image/jpeg;base64,{b64}'
                        ts_div = (
                            f'<div style="font-size:0.52rem;color:#4E4C46;padding:2px 4px;'
                            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{ts_label}</div>'
                            if show_ts else ""
                        )
                        cells.append(
                            f'<div class="gframe" style="position:relative;background:#111110;'
                            f'border:1px solid rgba(240,165,0,0.12);overflow:hidden;cursor:grab">'
                            f'<img src="{data_uri}" draggable="true" {_drag_tip} '
                            f'style="width:100%;height:{thumb_h}px;object-fit:cover;display:block"/>'
                            + ts_div
                            + f'<a href="{data_uri}" download="frame_{i+1:05d}.jpg" class="gdl">&#8595; Save</a>'
                            f'</div>'
                        )
                    st.markdown(
                        '<style>'
                        '.gframe .gdl{display:none;position:absolute;bottom:0;left:0;width:100%;'
                        'text-align:center;background:rgba(10,10,8,0.88);color:#F0A500;'
                        "font-family:'Share Tech Mono',monospace;font-size:0.58rem;"
                        'text-decoration:none;padding:5px 0;}'
                        '.gframe:hover .gdl{display:block;}'
                        '.gframe:hover{border-color:rgba(240,165,0,0.4)!important;}'
                        '</style>'
                        f'<div style="height:{scroll_h};overflow-y:auto;padding-right:4px">'
                        f'<div style="display:grid;grid-template-columns:{cols_css};gap:4px;">'
                        + "".join(cells)
                        + "</div></div>",
                        unsafe_allow_html=True,
                    )


    # ── Session History ──
    if st.session_state.get("sessions"):
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'Russo One\',sans-serif;color:#F0A500;'
            'letter-spacing:2px;font-size:1.1rem;text-transform:uppercase;margin-bottom:0.8rem">'
            '📂 Session History</p>',
            unsafe_allow_html=True,
        )
        for sess in st.session_state.sessions:
            badge_color = {"badge-green": "#F0A500", "badge-yellow": "#C88800",
                           "badge-orange": "#C86400", "badge-red": "#C04040"}.get(sess["badge"], "#4E4C46")
            with st.expander(
                f"#{sess['id']} · {sess['ts']} · {sess['video']} · VCR {sess['vcr']:.4f} · {sess['status']}",
                expanded=False,
            ):
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("VCR",      f"{sess['vcr']:.4f}")
                sc2.metric("PCU",      f"{sess['total_pcu']:.0f}")
                sc3.metric("Duration", f"{sess['duration']:.0f}s")
                sc4.metric("Tracker",  sess["tracker"])

                # Mini vehicle breakdown
                if sess["vehicle_counts"]:
                    mini_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:3px 0;border-bottom:1px solid rgba(240,165,0,0.08);">'
                        f'<span style="font-family:\'Barlow Condensed\',sans-serif;color:#9E9A94;font-size:0.8rem">{c.title()}</span>'
                        f'<span style="font-family:\'Share Tech Mono\',monospace;color:#F0A500;font-size:0.8rem">{int(n)}</span></div>'
                        for c, n in sorted(sess["vehicle_counts"].items(), key=lambda x: -x[1])
                    )
                    st.markdown(f'<div style="margin:0.5rem 0">{mini_html}</div>', unsafe_allow_html=True)

                # Mini frame gallery
                if sess.get("frame_store_b64"):
                    sess_frames = sess["frame_store_b64"][::max(1, len(sess["frame_store_b64"]) // 12)][:12]
                    # Pure HTML grid — no st.columns to avoid None-button artifacts
                    mini_cells = "".join(
                        f'<div style="position:relative;overflow:hidden">'
                        f'<img src="data:image/jpeg;base64,{fb64}" '
                        f'style="width:100%;height:70px;object-fit:cover;display:block;'
                        f'border-radius:3px;border:1px solid rgba(240,165,0,0.1)"/>'
                        f'</div>'
                        for fts, fb64 in sess_frames
                    )
                    st.markdown(
                        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-top:0.5rem">'
                        + mini_cells + "</div>",
                        unsafe_allow_html=True,
                    )

    # Cleanup temp file (best-effort)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
