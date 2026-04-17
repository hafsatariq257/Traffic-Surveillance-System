"""
Traffic Monitoring System — Streamlit Frontend
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from detector import (
    TrackedVehicle, ViolationRecord,
    SpeedEstimator, SignalViolationDetector,
    NumberPlateReader, HelmetDetector,
    VehicleCounter, ViolationLogger,
    draw_vehicle, draw_ui, COLORS,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Traffic Monitoring System",
    page_icon   = "🚦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F3864;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #d0d8f0;
    }
    .violation-alert {
        background: #fff0f0;
        border-left: 4px solid #e53e3e;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .signal-red   { color: #e53e3e; font-weight: 700; font-size: 1.1rem; }
    .signal-green { color: #38a169; font-weight: 700; font-size: 1.1rem; }
    .stProgress > div > div { background-color: #1F3864; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────

def init_state():
    defaults = dict(
        running        = False,
        signal_red     = True,
        violations     = [],
        counts         = defaultdict(int),
        total_count    = 0,
        current_speed  = {},
        plates_seen    = set(),
        frames_processed = 0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-light.png", width=80)
    st.markdown("## ⚙ Configuration")

    model_path = st.text_input("YOLOv8 model path", value="best.pt",
                               help="Path to your trained best.pt file")

    st.markdown("---")
    st.markdown("### Detection Settings")
    conf_thresh    = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    iou_thresh     = st.slider("NMS IOU threshold",    0.1, 0.9, 0.50, 0.05)
    pixels_per_m   = st.slider("Pixels per metre (calibration)", 2.0, 30.0, 8.0, 0.5,
                               help="Measure a known distance in your scene to calibrate speed")

    st.markdown("---")
    st.markdown("### Line Positions (%)")
    stop_pct  = st.slider("Stop line (% from top)",  10, 90, 40)
    count_pct = st.slider("Count line (% from top)", 10, 90, 65)

    st.markdown("---")
    st.markdown("### Signal Control")
    sig_col1, sig_col2 = st.columns(2)
    with sig_col1:
        if st.button("🔴 RED", use_container_width=True):
            st.session_state.signal_red = True
    with sig_col2:
        if st.button("🟢 GREEN", use_container_width=True):
            st.session_state.signal_red = False

    sig_status = "🔴 RED" if st.session_state.signal_red else "🟢 GREEN"
    st.markdown(f"**Current signal:** {sig_status}")

    st.markdown("---")
    st.markdown("### Optional Models")
    plate_model  = st.text_input("Plate model path (optional)", value="")
    helmet_model = st.text_input("Helmet model path (optional)", value="")

    show_fps    = st.checkbox("Show FPS overlay", value=True)
    save_output = st.checkbox("Save annotated video", value=False)

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🚦 Traffic Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered vehicle detection, speed estimation, and violation alerts</div>',
            unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_live, tab_violations, tab_stats, tab_help = st.tabs(
    ["📹 Live Detection", "⚠ Violations Log", "📊 Statistics", "📖 Help"])

# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 1 — Live Detection                                   ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_live:

    col_upload, col_spacer = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a traffic video file to analyse",
        )

    use_webcam = st.checkbox("Use webcam (source 0)", value=False)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    total_box    = m1.empty()
    car_box      = m2.empty()
    moto_box     = m3.empty()
    violation_box= m4.empty()
    fps_box      = m5.empty()

    # Video display
    video_placeholder = st.empty()
    progress_bar      = st.empty()
    status_text       = st.empty()

    # Plate display
    st.markdown("### 🔍 Recently detected plates")
    plates_placeholder = st.empty()

    # Control buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        start_btn = st.button("▶ Start", type="primary", use_container_width=True)
    with btn_col2:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)

    if stop_btn:
        st.session_state.running = False

    # ── Run detection ─────────────────────────────────────────────────────────
    if start_btn and (uploaded or use_webcam):
        # Validate model
        if not Path(model_path).exists():
            st.error(f"Model not found: `{model_path}`. Please check the path in the sidebar.")
            st.stop()

        st.session_state.running      = True
        st.session_state.violations   = []
        st.session_state.counts       = defaultdict(int)
        st.session_state.total_count  = 0
        st.session_state.frames_processed = 0
        st.session_state.plates_seen  = set()

        # Load model
        with st.spinner("Loading YOLOv8 model …"):
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.fuse()

        # Set up video source
        if use_webcam:
            source = 0
            tmp_path = None
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            source   = tmp.name
            tmp_path = tmp.name

        cap    = cv2.VideoCapture(source)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999999

        stop_line_y  = int(height * stop_pct  / 100)
        count_line_y = int(height * count_pct / 100)

        speed_est   = SpeedEstimator(fps, pixels_per_meter=pixels_per_m)
        sig_det     = SignalViolationDetector(stop_line_y, signal_red=st.session_state.signal_red)
        counter     = VehicleCounter(count_line_y)
        plate_reader= NumberPlateReader(plate_model or None)
        helmet_det  = HelmetDetector(helmet_model or None)

        out_dir  = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger   = ViolationLogger(str(out_dir / f"violations_{ts_str}.xlsx"))

        writer    = None
        if save_output:
            out_vid = str(out_dir / f"annotated_{ts_str}.mp4")
            writer  = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (width, height))

        vehicles   = {}
        frame_num  = 0
        prev_time  = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            sig_det.set_signal(st.session_state.signal_red)

            # Track
            results = model.track(
                frame,
                imgsz   = 640,
                conf    = conf_thresh,
                iou     = iou_thresh,
                persist = True,
                tracker = "bytetrack.yaml",
                verbose = False,
            )

            if (results[0].boxes is not None and
                    results[0].boxes.id is not None):
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                ids     = results[0].boxes.id.cpu().numpy().astype(int)

                for bbox, cls_id, track_id in zip(boxes, cls_ids, ids):
                    cls_name = model.names[cls_id]
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)

                    if track_id not in vehicles:
                        vehicles[track_id] = TrackedVehicle(
                            track_id=track_id, class_name=cls_name,
                            bbox=bbox, centroid=(cx, cy))
                    else:
                        v = vehicles[track_id]
                        v.bbox     = bbox
                        v.centroid = (cx, cy)

                    v = vehicles[track_id]
                    speed_est.update(v)
                    counter.update(v)

                    if sig_det.check(v):
                        v.plate_text = plate_reader.read(frame, v.bbox)
                        rec = ViolationRecord(
                            timestamp   = datetime.now().strftime("%H:%M:%S"),
                            track_id    = track_id,
                            vehicle_cls = cls_name,
                            plate_text  = v.plate_text,
                            speed_kmh   = v.speed_kmh,
                            violation   = "Signal Violation",
                        )
                        logger.log(rec)
                        st.session_state.violations.append(rec)

                    if frame_num % 10 == 0:
                        v.has_helmet = helmet_det.detect(frame, v.bbox, cls_name)
                        if not v.has_helmet:
                            v.plate_text = v.plate_text or plate_reader.read(frame, v.bbox)
                            rec = ViolationRecord(
                                timestamp   = datetime.now().strftime("%H:%M:%S"),
                                track_id    = track_id,
                                vehicle_cls = cls_name,
                                plate_text  = v.plate_text,
                                speed_kmh   = v.speed_kmh,
                                violation   = "No Helmet",
                            )
                            logger.log(rec)
                            st.session_state.violations.append(rec)

                    if v.plate_text:
                        st.session_state.plates_seen.add(v.plate_text)

                    draw_vehicle(frame, v, st.session_state.signal_red)

            draw_ui(frame, counter, st.session_state.signal_red,
                    stop_line_y, count_line_y)

            # FPS overlay
            now = time.time()
            live_fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            if show_fps:
                cv2.putText(frame, f"FPS: {live_fps:.1f}",
                            (width - 100, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer:
                writer.write(frame)

            # Update Streamlit
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, use_container_width=True)

            # Metrics
            total_box.metric("🚗 Total", counter.total)
            car_box.metric("🚘 Cars", counter.counts.get("car", 0))
            moto_box.metric("🏍 Motorbikes", counter.counts.get("motorcycle", 0))
            violation_box.metric("⚠ Violations", len(st.session_state.violations))
            fps_box.metric("⚡ FPS", f"{live_fps:.0f}")

            # Progress
            if total_frames < 9999999:
                progress_bar.progress(min(frame_num / total_frames, 1.0))
            status_text.text(f"Frame {frame_num} | Signal: {'RED' if st.session_state.signal_red else 'GREEN'}")

            # Plates
            if st.session_state.plates_seen:
                plates_placeholder.code(", ".join(sorted(st.session_state.plates_seen)))

        # Done
        cap.release()
        if writer:
            writer.release()
        if tmp_path:
            os.unlink(tmp_path)
        logger.save()
        status_text.success(f"✅ Processing complete! {frame_num} frames analysed.")
        progress_bar.progress(1.0)

    elif start_btn:
        st.warning("Please upload a video or enable webcam.")

# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 2 — Violations Log                                   ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_violations:
    st.markdown("### ⚠ Violation Log")

    if not st.session_state.violations:
        st.info("No violations recorded yet. Start detection to see results here.")
    else:
        rows = []
        for v in st.session_state.violations:
            rows.append({
                "Time":         v.timestamp,
                "Track ID":     v.track_id,
                "Vehicle":      v.vehicle_cls,
                "Number Plate": v.plate_text or "Undetected",
                "Speed (km/h)": round(v.speed_kmh, 1),
                "Violation":    v.violation,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        # Download button
        xlsx_path = Path("outputs") / f"violations_{datetime.now().strftime('%Y%m%d')}.xlsx"
        if xlsx_path.exists():
            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel Report",
                    data        = f.read(),
                    file_name   = xlsx_path.name,
                    mime        = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # Individual violation cards
        st.markdown("---")
        for rec in reversed(st.session_state.violations[-5:]):
            st.markdown(
                f'<div class="violation-alert">'
                f'⏱ <b>{rec.timestamp}</b> | {rec.violation.upper()} | '
                f'{rec.vehicle_cls} #{rec.track_id} | '
                f'Plate: <code>{rec.plate_text or "N/A"}</code> | '
                f'{rec.speed_kmh:.0f} km/h'
                f'</div>',
                unsafe_allow_html=True,
            )

# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 3 — Statistics                                       ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_stats:
    st.markdown("### 📊 Traffic Statistics")

    if not st.session_state.violations and st.session_state.total_count == 0:
        st.info("Statistics will appear here once detection has been run.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Vehicle distribution")
            if st.session_state.counts:
                import plotly.express as px
                df_counts = pd.DataFrame(
                    list(st.session_state.counts.items()),
                    columns=["Vehicle", "Count"]
                )
                fig = px.pie(df_counts, names="Vehicle", values="Count",
                             color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Violation breakdown")
            if st.session_state.violations:
                viol_types = defaultdict(int)
                for v in st.session_state.violations:
                    viol_types[v.violation] += 1
                df_viols = pd.DataFrame(
                    list(viol_types.items()),
                    columns=["Violation", "Count"]
                )
                fig2 = px.bar(df_viols, x="Violation", y="Count",
                              color="Violation",
                              color_discrete_sequence=["#e53e3e", "#dd6b20"])
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total vehicles", st.session_state.total_count)
        col_b.metric("Total violations", len(st.session_state.violations))
        col_c.metric("Unique plates", len(st.session_state.plates_seen))

# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 4 — Help                                             ║
# ╚═══════════════════════════════════════════════════════════╝
with tab_help:
    st.markdown("""
### 📖 How to use this app

#### 1. Prepare your model
Train your YOLOv8 model using the provided `train_colab.py` notebook.
Copy `best.pt` to the same folder as `app.py` (or enter the full path in the sidebar).

#### 2. Upload a video
Click **Upload traffic video** and select an `.mp4` or `.avi` file.
Alternatively, check **Use webcam** to process live footage.

#### 3. Configure settings (sidebar)
| Setting | Description |
|---|---|
| Confidence threshold | Minimum detection confidence (lower = more detections, more false positives) |
| NMS IOU | Overlap threshold for non-max suppression |
| Pixels per metre | Calibrate speed by measuring a known distance in pixels |
| Stop line | Y position of the traffic stop line (% from top) |
| Count line | Y position where vehicles are counted (% from top) |

#### 4. Control the signal
Use the 🔴 RED / 🟢 GREEN buttons to simulate signal state.
Vehicles crossing the stop line while signal is RED trigger a violation.

#### 5. Violations & Excel export
All violations (signal breach + no helmet) are logged automatically.
Download the Excel report from the **Violations Log** tab.

#### 6. Speed calibration
Measure a known real-world distance (e.g. a 3 m lane width = X pixels in your video).
Set **Pixels per metre = X / 3**.

---
### Detected classes
`car` · `motorcycle` · `bicycle` · `auto` (rickshaw) · `truck` · `tanker` · `heavy_bike`

### Requirements
```
ultralytics easyocr openpyxl opencv-python streamlit plotly pandas
```
""")
