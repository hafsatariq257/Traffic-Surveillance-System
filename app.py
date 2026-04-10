import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
import requests
from ultralytics import YOLO
from inference import process_frame, VehicleTracker
from utils.excel_logger import get_violations, init_excel
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Traffic Surveillance System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Updated Yellow & Black Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Title - Bright Yellow Gradient */
    h1 { 
        background: linear-gradient(90deg, #FFFF00, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
    }

    /* Sidebar - Yellow Accents */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121212 0%, #1a1a1a 100%);
        border-right: 1px solid #FFFF0033;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFF00 !important;
    }

    /* Metric cards - Yellow Borders */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e1e, #252525);
        border: 1px solid #FFFF0066;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFF00 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] { color: #ffffff !important; }

    /* Buttons - Solid Yellow with Black Text */
    .stButton > button {
        background: #FFFF00 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 2rem !important;
        transition: transform 0.2s ease !important;
    }
    .stButton > button:hover { 
        transform: scale(1.03) !important;
        background: #FFD700 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #FFD700 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }

    /* Progress bar - Yellow */
    .stProgress > div > div {
        background: #FFFF00 !important;
        border-radius: 10px !important;
    }

    /* Dataframe - Yellow Border */
    [data-testid="stDataFrame"] {
        border: 1px solid #FFFF0033 !important;
        border-radius: 10px !important;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed #FFFF0066 !important;
        border-radius: 12px !important;
        background: #1a1a1a !important;
    }

    /* Headings & Subheaders - High Visibility Yellow */
    h2, h3, h4 { 
        color: #FFFF00 !important; 
        font-family: 'Inter', sans-serif;
    }

    /* Markdown text */
    .stMarkdown p { color: #ffffff; }

    /* Signal badges */
    .signal-red {
        display: inline-block;
        background: #ff0000;
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
    }
    .signal-green {
        display: inline-block;
        background: #00ff00;
        color: black;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
    }

    /* Section divider - Yellow */
    hr { border-color: #FFFF0044 !important; }
</style>
""", unsafe_allow_html=True)

# ── Lottie loader ──────────────────────────────────────────────────────────────
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_traffic = load_lottie("https://assets9.lottiefiles.com/packages/lf20_UJNc2t.json")
lottie_car     = load_lottie("https://assets4.lottiefiles.com/packages/lf20_xlmz9xwm.json")
lottie_done    = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jR229r.json")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if lottie_car:
        st_lottie(lottie_car, height=140, key="sidebar_anim")
    else:
        st.markdown("## 🚗")

    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    stop_line_pct = st.slider("📍 Stop line position (%)", 30, 80, 55)
    signal_red    = st.toggle("🔴 Signal is RED", value=True)

    if signal_red:
        st.markdown('<span class="signal-red">● RED SIGNAL ACTIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="signal-green">● GREEN SIGNAL ACTIVE</span>', unsafe_allow_html=True)

    st.markdown("---")
    speed_limit = st.number_input(" Speed limit (km/h)", value=60)

    st.markdown("---")
    st.markdown("**🎯 Classes detected:**")
    for cls in ["🚗 Car", "🏍️ Motorcycle", "🚌 Bus", "🚛 Truck", "🚲 Bicycle"]:
        st.markdown(f"- {cls}")

    st.markdown("---")
    st.caption("Powered by YOLOv8 + Streamlit")


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_anim = st.columns([3, 1])
with col_title:
    st.title("🚦 Traffic Surveillance System")
    st.markdown("**AI-Driven Traffic** Violation Management System.")
with col_anim:
    if lottie_traffic:
        st_lottie(lottie_traffic, height=120, key="header_anim")

st.markdown("---")


# ── Model + Excel init ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
init_excel()


# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("### 📁 Upload Traffic Video")
uploaded = st.file_uploader(
    "Drop your video here",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎥 Live Detection")
        frame_placeholder = st.empty()

    with col2:
        st.markdown("### 📊 Stats")
        count_metric     = st.empty()
        violation_metric = st.empty()
        st.markdown("#### 🚨 Recent Violations")
        violation_log = st.empty()

    st.markdown("")
    run = st.button("▶️  Run Detection", type="primary", use_container_width=True)

    if run:
        cap           = cv2.VideoCapture(tfile.name)
        tracker       = VehicleTracker()
        total_violations = 0
        frame_count   = 0
        all_violations = []

        st.markdown("**Processing video...**")
        stop_bar    = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            h = frame.shape[0]
            stop_line_y = int(h * stop_line_pct / 100)

            annotated, count, violations = process_frame(
                frame, model, tracker, stop_line_y, signal_red, frame_count
            )

            total_violations += len(violations)
            all_violations.extend(violations)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            count_metric.metric("🚗 Vehicles Counted",   count)
            violation_metric.metric("🚨 Violations Detected", total_violations)

            if all_violations:
                df = pd.DataFrame(all_violations[-5:])
                violation_log.dataframe(df, use_container_width=True)

            progress = min(frame_count / max(total_frames, 1), 1.0)
            stop_bar.progress(progress)

        cap.release()

        # ── Done ──────────────────────────────────────────────────────────────
        st.markdown("---")
        done_col, txt_col = st.columns([1, 3])
        with done_col:
            if lottie_done:
                st_lottie(lottie_done, height=120, key="done_anim")
        with txt_col:
            st.success(f"✅ Done! Detected **{count}** vehicles and **{total_violations}** violations.")

        st.markdown("### 📋 All Violations Log")
        violations_data = get_violations()
        if violations_data:
            df_all = pd.DataFrame(
                violations_data,
                columns=["#", "Timestamp", "Type", "Plate", "Class", "Speed"]
            )
            st.dataframe(df_all, use_container_width=True)

            with open("violations.xlsx", "rb") as f:
                st.download_button(
                    "⬇️  Download Violations Excel",
                    f,
                    file_name="violations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ No violations recorded.")

        os.unlink(tfile.name)