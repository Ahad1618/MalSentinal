"""
app/main.py
-----------
MalSentinel — Streamlit web interface (Redesigned).

Run with:
    streamlit run app/main.py
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MalSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-void:      #070A0F;
    --bg-deep:      #0C1018;
    --bg-surface:   #111827;
    --bg-raised:    #1A2234;
    --bg-glass:     rgba(17, 24, 39, 0.7);
    --border-dim:   rgba(99, 179, 237, 0.12);
    --border-mid:   rgba(99, 179, 237, 0.25);
    --border-bright:rgba(99, 179, 237, 0.55);
    --cyan:         #38BDF8;
    --cyan-dim:     #0EA5E9;
    --cyan-glow:    rgba(56, 189, 248, 0.15);
    --red:          #F87171;
    --red-dim:      #EF4444;
    --red-glow:     rgba(248, 113, 113, 0.15);
    --green:        #34D399;
    --green-dim:    #10B981;
    --green-glow:   rgba(52, 211, 153, 0.15);
    --amber:        #FBBF24;
    --text-primary: #E2E8F0;
    --text-mid:     #94A3B8;
    --text-dim:     #475569;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
    --font-ui:      'Space Mono', monospace;
}

/* ── Global resets ── */
* { box-sizing: border-box; word-wrap: break-word; overflow-wrap: break-word; word-break: break-word; }
html, body, .stApp {
    background-color: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
}

/* Scanline overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Grid dot pattern background */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(56, 189, 248, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
    position: relative;
    z-index: 1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border-dim) !important;
    position: relative;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 1px; height: 100%;
    background: linear-gradient(180deg, transparent, var(--cyan) 40%, transparent);
    opacity: 0.3;
}
[data-testid="stSidebar"] * {
    color: var(--text-mid) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-size: 0.95rem !important;
    color: var(--cyan) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Sidebar section dividers ── */
.sidebar-section {
    border-top: 1px solid var(--border-dim);
    padding-top: 1rem;
    margin-top: 0.5rem;
}

/* ── Status pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin: 3px 0;
}
.status-online {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.35);
    color: var(--green) !important;
}
.status-offline {
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
    color: var(--red) !important;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.dot-online { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse-dot 2s ease-in-out infinite; }
.dot-offline { background: var(--red); }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Metric card in sidebar ── */
.sidebar-metric {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric-label { color: var(--text-dim) !important; font-size: 0.75rem !important; }
.sidebar-metric-value { color: var(--cyan) !important; font-family: var(--font-mono) !important; font-weight: 500 !important; font-size: 0.85rem !important; }

/* ── Hero section ── */
.hero-wrapper {
    position: relative;
    padding: 2.5rem 0 1.5rem 0;
    margin-bottom: 0.5rem;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px; height: 1px;
    background: var(--cyan);
}
.hero-title {
    font-family: var(--font-display) !important;
    font-size: 3.4rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    margin: 0 0 0.4rem 0 !important;
    background: linear-gradient(135deg, #E2E8F0 0%, var(--cyan) 60%, #7DD3FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1rem;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    padding: 4px 10px;
    border-radius: 4px;
}

/* ── Divider line ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-bright), transparent);
    margin: 1.5rem 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1px dashed var(--border-mid) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    background: rgba(56, 189, 248, 0.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-family: var(--font-ui) !important;
    color: var(--text-mid) !important;
}
[data-testid="stFileUploader"] button {
    font-family: var(--font-ui) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 8px !important;
    color: var(--cyan) !important;
    background: transparent !important;
    padding: 6px 16px !important;
    white-space: normal !important;
    word-wrap: break-word !important;
}
[data-testid="stFileUploader"] button svg, 
[data-testid="stFileUploader"] button .material-symbols-rounded,
[data-testid="stFileUploader"] span[data-testid="stIcon"] {
    display: none !important;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--cyan-glow) !important;
}
[data-testid="stFileUploader"] small {
    font-family: var(--font-mono) !important;
}

/* ── Feature cards (landing) ── */
.feature-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
    height: 100%;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.feature-card-cyan::before  { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.feature-card-red::before   { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.feature-card-green::before { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.feature-card:hover { border-color: var(--border-bright); transform: translateY(-2px); }

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: 0.75rem;
    display: block;
}
.feature-title {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.5rem 0 !important;
}
.feature-cyan  { color: var(--cyan) !important; }
.feature-red   { color: var(--red) !important; }
.feature-green { color: var(--green) !important; }
.feature-body {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-mid) !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}

/* ── Upload section header ── */
.upload-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.6rem;
}
.upload-header-label {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}
.upload-badge {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    color: var(--text-dim);
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── File info strip ── */
.file-info-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg-raised);
    border: 1px solid var(--border-mid);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 1rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
}
.file-info-name { color: var(--cyan); font-weight: 500; }
.file-info-sep  { color: var(--text-dim); }
.file-info-size { color: var(--text-mid); }

/* ── Progress step ── */
.progress-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 8px;
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--cyan);
    margin-bottom: 4px;
    animation: step-slide 0.3s ease-out;
}
@keyframes step-slide {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Verdict banners ── */
.verdict-wrapper {
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.verdict-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    background: repeating-linear-gradient(
        -45deg, currentColor, currentColor 1px, transparent 1px, transparent 8px
    );
}
.verdict-mal {
    background: linear-gradient(135deg, rgba(127,29,29,0.6), rgba(153,27,27,0.4));
    border: 1px solid rgba(248,113,113,0.4);
    color: var(--red);
}
.verdict-ben {
    background: linear-gradient(135deg, rgba(6,78,59,0.6), rgba(5,150,105,0.3));
    border: 1px solid rgba(52,211,153,0.4);
    color: var(--green);
}
.verdict-label {
    font-family: var(--font-display);
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 4px;
}
.verdict-text {
    font-family: var(--font-display);
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    line-height: 1;
}
.verdict-conf {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    opacity: 0.75;
    margin-top: 6px;
}
.verdict-icon { font-size: 3.5rem; opacity: 0.9; }

/* ── Metric tiles ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.metric-tile {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.tile-cyan::after  { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.tile-red::after   { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.tile-green::after { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.tile-amber::after { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
.metric-value {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
}

/* ── Confidence bar cards ── */
.conf-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.conf-label {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-mid);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
}
.conf-label span { color: var(--text-primary); font-weight: 600; }
.prog-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
    position: relative;
}
.prog-fill {
    height: 100%;
    border-radius: 100px;
    position: relative;
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.prog-red   { background: linear-gradient(90deg, #7F1D1D, var(--red)); box-shadow: 0 0 12px rgba(248,113,113,0.4); }
.prog-green { background: linear-gradient(90deg, #064E3B, var(--green)); box-shadow: 0 0 12px rgba(52,211,153,0.4); }
.prog-blue  { background: linear-gradient(90deg, #1E3A5F, var(--cyan)); box-shadow: 0 0 12px rgba(56,189,248,0.4); }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: var(--border-dim);
}
.section-header-text {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
    letter-spacing: 0.08em;
    white-space: nowrap;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid var(--border-dim) !important;
    border-bottom: none !important;
    padding: 6px 8px 0 8px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
    transition: color 0.2s, background 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; background: var(--bg-raised) !important; }
.stTabs [aria-selected="true"] {
    background: var(--bg-raised) !important;
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1.5rem !important;
}

/* ── Dataframes ── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-dim) !important;
}
.stDataFrame * { font-family: var(--font-mono) !important; font-size: 0.8rem !important; }

/* ── API badges ── */
.api-badge-danger {
    display: inline-block;
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.4);
    color: var(--red);
    padding: 3px 10px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    margin: 3px;
    letter-spacing: 0.04em;
}
.api-badge-ok {
    display: inline-block;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    color: var(--green);
    padding: 3px 10px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    margin: 3px;
    letter-spacing: 0.04em;
}

/* ── Code blocks ── */
.stCode, code, pre {
    font-family: var(--font-mono) !important;
    background: var(--bg-void) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--bg-raised) !important;
    border-radius: 8px !important;
    color: var(--text-mid) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--border-dim) !important;
}
.streamlit-expanderContent {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-top: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.5rem !important;
}
.stButton > button:hover {
    background: var(--cyan-glow) !important;
    box-shadow: 0 0 20px var(--cyan-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cyan-dim), var(--cyan)) !important;
    color: var(--bg-void) !important;
    font-weight: 700 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(56,189,248,0.4) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--cyan-dim), var(--cyan)) !important;
    color: var(--bg-void) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.5rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── Info / Warning / Error boxes ── */
.stAlert { border-radius: 10px !important; font-family: var(--font-mono) !important; font-size: 0.82rem !important; }

/* ── Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── st.metric override ── */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 0.75rem !important; color: var(--text-dim) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cyan) !important; }

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-dim);
}

/* ── Results section title ── */
.results-title {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.25rem;
}
.results-title-accent {
    font-size: 0.75rem;
    font-family: var(--font-mono);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
  <div style="font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800;
              background:linear-gradient(135deg,#E2E8F0,#38BDF8);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              letter-spacing:-0.01em;">
    🛡️ MalSentinel
  </div>
  <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
              letter-spacing:0.15em; text-transform:uppercase; color:#475569;
              margin-top:3px;">
    Static PE Analysis System
  </div>
</div>
""", unsafe_allow_html=True)

    # Model status
    st.markdown('<div class="section-label">Engine Status</div>', unsafe_allow_html=True)

    lgbm_ok, bayes_ok = False, False
    lgbm_metrics = {}
    try:
        from train import load_model
        payload = load_model()
        lgbm_metrics = payload.get("metrics", {})
        lgbm_ok = True
    except Exception:
        pass
    try:
        from bayesian import load_bayes
        load_bayes()
        bayes_ok = True
    except Exception:
        pass

    st.markdown(f"""
<div class="status-pill {'status-online' if lgbm_ok else 'status-offline'}">
  <span class="status-dot {'dot-online' if lgbm_ok else 'dot-offline'}"></span>
  LightGBM — {'READY' if lgbm_ok else 'OFFLINE'}
</div>
<div class="status-pill {'status-online' if bayes_ok else 'status-offline'}">
  <span class="status-dot {'dot-online' if bayes_ok else 'dot-offline'}"></span>
  Naive Bayes — {'READY' if bayes_ok else 'OFFLINE'}
</div>
""", unsafe_allow_html=True)

    if lgbm_metrics:
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Model Performance</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="sidebar-metric">
  <span class="sidebar-metric-label">AUC-ROC</span>
  <span class="sidebar-metric-value">{lgbm_metrics.get('auc', 0):.4f}</span>
</div>
<div class="sidebar-metric">
  <span class="sidebar-metric-label">F1 Score</span>
  <span class="sidebar-metric-value">{lgbm_metrics.get('f1', 0):.4f}</span>
</div>
<div class="sidebar-metric">
  <span class="sidebar-metric-label">Accuracy</span>
  <span class="sidebar-metric-value">{lgbm_metrics.get('accuracy', 0):.4f}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-metric">
  <span class="sidebar-metric-label">Source</span>
  <span class="sidebar-metric-value">EMBER 2018</span>
</div>
<div class="sidebar-metric">
  <span class="sidebar-metric-label">PE Files</span>
  <span class="sidebar-metric-value">600,000</span>
</div>
<div class="sidebar-metric">
  <span class="sidebar-metric-label">Features</span>
  <span class="sidebar-metric-value">2,381</span>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Analysis Pipeline</div>', unsafe_allow_html=True)
    steps_html = ""
    for i, step in enumerate([
        "Upload .exe / .dll",
        "Extract features (no exec)",
        "LightGBM classifies",
        "Bayesian API inference",
        "SHAP explains decision",
        "Download PDF report",
    ], 1):
        steps_html += f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;font-size:0.78rem;color:#94A3B8;"><span style="color:#38BDF8;font-family:\'IBM Plex Mono\',monospace;font-weight:600;">{i:02d}</span> {step}</div>'
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Train Models</div>', unsafe_allow_html=True)
    with st.expander("⚙️ Terminal Commands"):
        st.code("python setup_data.py\npython train.py\npython bayesian.py", language="bash")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-eyebrow">v2.0 · AI-Powered · Static Analysis</div>
  <div class="hero-title">MalSentinel</div>
  <div class="hero-subtitle">AI-Powered Static Malware Detection &amp; Classification System</div>
  <div class="hero-tags">
    <span class="hero-tag">LightGBM</span>
    <span class="hero-tag">Bayesian Inference</span>
    <span class="hero-tag">SHAP Explainability</span>
    <span class="hero-tag">EMBER 2018 Dataset</span>
    <span class="hero-tag">600K PE Samples</span>
    <span class="hero-tag">2381 Features</span>
  </div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-header">
  <span class="upload-header-label">Upload a PE Binary</span>
  <span class="upload-badge">.exe · .dll · .bin</span>
  <span class="upload-badge" style="color:#34D399;border-color:rgba(52,211,153,0.3);">Never Executed</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload a PE file",
    type=["exe", "dll", "bin"],
    help="Upload a Windows PE file. It will never be executed.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="feature-card feature-card-cyan">
  <span class="feature-icon">🤖</span>
  <div class="feature-title feature-cyan">LightGBM Classifier</div>
  <p class="feature-body">
    Gradient boosting on 2,381 PE structural features.<br>
    Trained on 600K labeled binaries from the EMBER 2018 dataset.<br>
    Industry-grade detection accuracy.
  </p>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="feature-card feature-card-red">
  <span class="feature-icon">🔗</span>
  <div class="feature-title feature-red">Bayesian Network</div>
  <p class="feature-body">
    Gaussian Naive Bayes on dangerous API imports.<br>
    Computes P(Malicious | API features) for a transparent, interpretable second opinion.
  </p>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="feature-card feature-card-green">
  <span class="feature-icon">💡</span>
  <div class="feature-title feature-green">SHAP Explainability</div>
  <p class="feature-body">
    Every prediction is explained. Waterfall charts show exactly which features pushed
    the decision toward malicious or benign.
  </p>
</div>""", unsafe_allow_html=True)

else:
    # ── File info strip ───────────────────────────────────────────────────────
    st.markdown(f"""
<div class="file-info-strip">
  <span style="color:#38BDF8;">📄</span>
  <span class="file-info-name">{uploaded_file.name}</span>
  <span class="file-info-sep">·</span>
  <span class="file-info-size">{uploaded_file.size:,} bytes</span>
  <span class="file-info-sep">·</span>
  <span style="color:#34D399;font-size:0.75rem;">Queued for analysis</span>
</div>""", unsafe_allow_html=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Progress animation
    progress_container = st.container()
    with progress_container:
        prog_bar = st.progress(0)
        status_placeholder = st.empty()

    steps = [
        (15, "🔍", "Parsing PE structure…"),
        (35, "⚙️", "Extracting 2,381 EMBER features…"),
        (55, "🤖", "Running LightGBM classifier…"),
        (70, "🔗", "Computing Bayesian probability…"),
        (85, "💡", "Generating SHAP explanation…"),
        (95, "📊", "Building results dashboard…"),
    ]

    for pct, icon, msg in steps:
        prog_bar.progress(pct)
        status_placeholder.markdown(
            f'<div class="progress-step">{icon} <span style="color:#94A3B8;">{msg}</span></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.25)

    # Run actual prediction
    from predict import run_prediction
    with st.spinner(""):
        result = run_prediction(tmp_path)

    prog_bar.progress(100)
    status_placeholder.markdown(
        '<div class="progress-step" style="border-color:rgba(52,211,153,0.4);background:rgba(52,211,153,0.06);">'
        '✅ <span style="color:#34D399;">Analysis complete</span></div>',
        unsafe_allow_html=True
    )
    time.sleep(0.4)
    progress_container.empty()

    # Cleanup temp
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    # ── Error handling ────────────────────────────────────────────────────────
    if result["error"]:
        st.error(f"❌ {result['error']}")
        if "not found" in result["error"].lower():
            st.info("💡 Train the models first: run `python setup_data.py` then `python train.py` then `python bayesian.py`")
        st.stop()

    # ── RESULTS DASHBOARD ─────────────────────────────────────────────────────
    lgbm_prob    = result["lgbm_prob"]
    lgbm_verdict = result["lgbm_verdict"]
    bayes_prob   = result["bayes_prob"]
    pe_info      = result["pe_info"]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="results-title">
  🔬 Analysis Results
  <span class="results-title-accent">· LightGBM + Bayesian + SHAP</span>
</div>
""", unsafe_allow_html=True)

    # ── Verdict banner ────────────────────────────────────────────────────────
    if lgbm_verdict == "MALICIOUS":
        conf_pct = lgbm_prob * 100
        st.markdown(f"""
<div class="verdict-wrapper verdict-mal">
  <div>
    <div class="verdict-label">Threat Classification</div>
    <div class="verdict-text">MALICIOUS</div>
    <div class="verdict-conf">LightGBM Confidence: {conf_pct:.1f}%</div>
  </div>
  <div class="verdict-icon">⚠️</div>
</div>""", unsafe_allow_html=True)
    else:
        conf_pct = (1 - lgbm_prob) * 100
        st.markdown(f"""
<div class="verdict-wrapper verdict-ben">
  <div>
    <div class="verdict-label">Threat Classification</div>
    <div class="verdict-text">BENIGN</div>
    <div class="verdict-conf">LightGBM Confidence: {conf_pct:.1f}%</div>
  </div>
  <div class="verdict-icon">✔️</div>
</div>""", unsafe_allow_html=True)

    # ── Top metric tiles ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        color = "#F87171" if lgbm_prob > 0.5 else "#34D399"
        tile_class = "tile-red" if lgbm_prob > 0.5 else "tile-green"
        st.markdown(f"""
<div class="metric-tile {tile_class}">
  <div class="metric-value" style="color:{color};">{lgbm_prob*100:.1f}%</div>
  <div class="metric-label">Malicious Probability</div>
</div>""", unsafe_allow_html=True)
    with m2:
        if bayes_prob >= 0:
            b_color = "#F87171" if bayes_prob > 0.5 else "#34D399"
            b_class = "tile-red" if bayes_prob > 0.5 else "tile-green"
            st.markdown(f"""
<div class="metric-tile {b_class}">
  <div class="metric-value" style="color:{b_color};">{bayes_prob*100:.1f}%</div>
  <div class="metric-label">Bayesian Probability</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="metric-tile tile-cyan">
  <div class="metric-value" style="color:#475569;">N/A</div>
  <div class="metric-label">Bayesian Probability</div>
</div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
<div class="metric-tile tile-cyan">
  <div class="metric-value" style="color:#38BDF8;">{pe_info.get('num_sections', 'N/A')}</div>
  <div class="metric-label">PE Sections</div>
</div>""", unsafe_allow_html=True)
    with m4:
        n_sus = len(pe_info.get("suspicious_apis", []))
        sus_color = "#F87171" if n_sus > 0 else "#34D399"
        sus_class = "tile-red" if n_sus > 0 else "tile-green"
        st.markdown(f"""
<div class="metric-tile {sus_class}">
  <div class="metric-value" style="color:{sus_color};">{n_sus}</div>
  <div class="metric-label">Suspicious APIs</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence bar cards ──────────────────────────────────────────────────
    cb1, cb2 = st.columns(2)
    with cb1:
        fill_class = "prog-red" if lgbm_prob > 0.5 else "prog-green"
        pct = int(lgbm_prob * 100)
        st.markdown(f"""
<div class="conf-card">
  <div class="conf-label">
    LightGBM Malicious Score
    <span>{pct}%</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill {fill_class}" style="width:{pct}%;"></div>
  </div>
</div>""", unsafe_allow_html=True)
    with cb2:
        if bayes_prob >= 0:
            b_pct = int(bayes_prob * 100)
            b_fill = "prog-red" if bayes_prob > 0.5 else "prog-green"
            st.markdown(f"""
<div class="conf-card">
  <div class="conf-label">
    Bayesian API-based Score
    <span>{b_pct}%</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill {b_fill}" style="width:{b_pct}%;"></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  SHAP Explanation",
        "🔗  Bayesian Network",
        "📋  PE Details",
        "⬇️  Download Report",
    ])

    with tab1:
        st.markdown('<div class="section-label">Feature Contribution Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.8rem; color:#475569;">'
            '🔴 Red bars push toward MALICIOUS &nbsp;·&nbsp; 🔵 Blue bars push toward BENIGN'
            '</p>',
            unsafe_allow_html=True
        )
        if result["shap_figure"]:
            st.pyplot(result["shap_figure"])
        else:
            st.info("SHAP explanation not available. Install shap: `pip install shap`")

    with tab2:
        st.markdown('<div class="section-label">Bayesian Network — API Import Analysis</div>', unsafe_allow_html=True)
        bayes_det = result.get("bayes_details", {})
        if bayes_det:
            feat_vals  = bayes_det.get("feature_values", {})
            node_probs = bayes_det.get("node_probs", {})
            feat_names = bayes_det.get("feature_names", [])

            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#475569;margin-bottom:0.8rem;">Observed node values and conditional probabilities</p>', unsafe_allow_html=True)
            import pandas as pd
            rows = []
            for fn in feat_names:
                v = feat_vals.get(fn, 0)
                np_val = node_probs.get(fn, 0)
                active = "🔴 Active" if v > 0 else "⚪ Not observed"
                rows.append({"API / Feature": fn, "Observed": active, "P(feature|Malicious)": f"{np_val:.4f}"})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-label" style="margin-top:1.2rem;">Network Graph</div>', unsafe_allow_html=True)
            fig_bn, ax = plt.subplots(figsize=(10, 5))
            fig_bn.patch.set_facecolor("#0C1018")
            ax.set_facecolor("#0C1018")

            n_api = len(feat_names)
            ys = np.linspace(0.9, 0.1, n_api)
            for i, (fname, y) in enumerate(zip(feat_names, ys)):
                val = feat_vals.get(fname, 0)
                node_color = "#7F1D1D" if val > 0 else "#1A2234"
                text_color = "#FCA5A5" if val > 0 else "#64748B"
                ax.add_patch(FancyBboxPatch((0.02, y-0.045), 0.36, 0.09,
                    boxstyle="round,pad=0.01", facecolor=node_color,
                    edgecolor="#F87171" if val > 0 else "#334155", linewidth=1.5))
                ax.text(0.20, y, fname, ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold" if val > 0 else "normal",
                        fontfamily="monospace")
                prob = node_probs.get(fname, 0)
                ax.annotate("", xy=(0.72, 0.5), xytext=(0.38, y),
                    arrowprops=dict(arrowstyle="->",
                        color="#F87171" if val > 0 else "#334155", lw=1.5 if val > 0 else 0.8))
                ax.text(0.55, (y + 0.5) / 2 + (i % 2) * 0.02,
                        f"{prob:.3f}", fontsize=7, color="#64748B", ha="center", fontfamily="monospace")

            mal_color = "#7F1D1D" if lgbm_verdict == "MALICIOUS" else "#064E3B"
            mal_edge  = "#F87171" if lgbm_verdict == "MALICIOUS" else "#34D399"
            ax.add_patch(FancyBboxPatch((0.73, 0.38), 0.24, 0.24,
                boxstyle="round,pad=0.02", facecolor=mal_color,
                edgecolor=mal_edge, linewidth=2.5))
            ax.text(0.85, 0.5, f"MALICIOUS\nP={bayes_prob:.3f}" if bayes_prob >= 0 else "MALICIOUS",
                    ha="center", va="center", fontsize=9, color=mal_edge, fontweight="bold",
                    fontfamily="monospace")

            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.axis("off")
            ax.set_title("Bayesian Network: API Features → Malicious Classification",
                         color="#38BDF8", fontsize=11, pad=12, fontfamily="monospace")
            plt.tight_layout()
            st.pyplot(fig_bn)
        else:
            st.info("Bayesian model not available. Run: `python bayesian.py`")

    with tab3:
        st.markdown('<div class="section-label">PE File Metadata</div>', unsafe_allow_html=True)
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#38BDF8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">File Properties</p>', unsafe_allow_html=True)
            import pandas as pd
            props = {
                "File Size":     f"{pe_info.get('file_size_bytes', 0):,} bytes",
                "Sections":      pe_info.get("num_sections", "N/A"),
                "Machine":       pe_info.get("machine_type", "N/A"),
                "Entry Point":   pe_info.get("entry_point", "N/A"),
                "Has TLS":       "Yes" if pe_info.get("has_tls") else "No",
                "Has Debug":     "Yes" if pe_info.get("has_debug") else "No",
                "Has Resources": "Yes" if pe_info.get("has_resources") else "No",
            }
            df_props = pd.DataFrame(props.items(), columns=["Property", "Value"])
            st.dataframe(df_props, use_container_width=True, hide_index=True)

        with c_right:
            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#F87171;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Suspicious API Imports</p>', unsafe_allow_html=True)
            suspicious = pe_info.get("suspicious_apis", [])
            if suspicious:
                badges = "".join([f'<span class="api-badge-danger">{a}</span>' for a in suspicious])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown('<span class="api-badge-ok">✔ None detected</span>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#94A3B8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">DLL Import Table</p>', unsafe_allow_html=True)
            imports = pe_info.get("imports", [])
            if imports:
                for dll_entry in imports[:10]:
                    with st.expander(f"📦 {dll_entry['dll']}  ({len(dll_entry['functions'])} functions)"):
                        st.code("\n".join(dll_entry["functions"][:30]), language="text")
            else:
                st.info("No import table found.")
        
        # ── Malware Family & Extracted Text Features ──────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Classification Metadata</div>', unsafe_allow_html=True)
        mf_left, mf_right = st.columns(2)
        
        with mf_left:
            malware_family = result.get("malware_family", "Not Available")
            family_icon = "🔍" if "Not Available" in malware_family else "📌"
            st.markdown(f"""
<div class="metric-tile" style="border-left: 4px solid #A78BFA;">
  <div style="font-size:0.7rem;color:#C4B5FD;font-weight:bold;letter-spacing:0.05em;">Malware Family</div>
  <div style="font-size:1.3rem;color:#E9D5FF;margin-top:0.5rem;">{family_icon} {malware_family}</div>
  <div style="font-size:0.65rem;color:#D8B4FE;margin-top:0.5rem;">
    {"Extended datasets enable family classification" if "Not Available" in malware_family else "Detected from training data"}
  </div>
</div>""", unsafe_allow_html=True)
        
        with mf_right:
            indicators = pe_info.get("suspicious_indicators", [])
            indicator_count = len(indicators)
            if indicator_count > 0:
                st.markdown(f"""
<div class="metric-tile" style="border-left: 4px solid #FB923C;">
  <div style="font-size:0.7rem;color:#FED7AA;font-weight:bold;letter-spacing:0.05em;">Suspicious Indicators</div>
  <div style="font-size:1.3rem;color:#FEE2E2;margin-top:0.5rem;">⚠️ {indicator_count} found</div>
  <div style="font-size:0.65rem;color:#FECACA;margin-top:0.5rem;">URLs, registry keys, file paths detected</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="metric-tile" style="border-left: 4px solid #34D399;">
  <div style="font-size:0.7rem;color:#A7F3D0;font-weight:bold;letter-spacing:0.05em;">Suspicious Indicators</div>
  <div style="font-size:1.3rem;color:#DBEAFE;margin-top:0.5rem;">✔ None</div>
  <div style="font-size:0.65rem;color:#BFDBFE;margin-top:0.5rem;">No suspicious text artifacts found</div>
</div>""", unsafe_allow_html=True)
        
        # Show extracted text features if available
        if indicators:
            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#FB923C;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Extracted Text Indicators</p>', unsafe_allow_html=True)
            for ind_type, ind_value in indicators:
                display_val = ind_value[:60] + "..." if len(ind_value) > 60 else ind_value
                st.code(f"[{ind_type}] {display_val}", language="text")

        fv = result.get("feature_vector")
        if fv is not None:
            st.markdown('<div class="section-label" style="margin-top:1.2rem;">Feature Vector Preview — First 50 Features</div>', unsafe_allow_html=True)
            feat_names_path = Path("data/feature_names.txt")
            if feat_names_path.exists():
                names = feat_names_path.read_text().splitlines()[:50]
            else:
                names = [f"feature_{i}" for i in range(50)]
            df_fv = pd.DataFrame({
                "Feature": names,
                "Value":   [f"{v:.6f}" for v in fv[:50]],
            })
            st.dataframe(df_fv, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown('<div class="section-label">PDF Report Generator</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:#64748B;margin-bottom:1.2rem;">'
            'Generate a full analysis report including verdict, PE metadata, Bayesian analysis, and SHAP chart.'
            '</p>',
            unsafe_allow_html=True
        )
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Building report…"):
                try:
                    from report.generate import generate_report
                    pdf_bytes = generate_report(result, filename=f"malsentinel_{uploaded_file.name}.pdf")
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"malsentinel_{uploaded_file.name}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ Report ready for download!")
                except ImportError:
                    st.error("fpdf2 not installed. Run: `pip install fpdf2`")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider" style="margin-top:2rem;"></div>
<div style="display:flex; justify-content:space-between; align-items:center;
            padding:0.75rem 0; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#334155;">
  <span>MalSentinel · CS 2005 AI Project</span>
  <span>Static Analysis Only · Files are never executed</span>
  <span style="color:#38BDF8;">🛡️ v2.0</span>
</div>
""", unsafe_allow_html=True)