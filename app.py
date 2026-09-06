"""
ANPR Trajectory Tracking — Streamlit Dashboard
------------------------------------------------
A simple web page version of the notebook prototype:
  1. Upload one or more "camera" videos
  2. Click "Run Detection" — finds vehicles, reads plates, logs sightings
  3. Search a plate number and see its trajectory across cameras
  4. View basic city-wide stats

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Run from Google Colab (no local install needed):
    See the README.md that comes with this file.
"""

import os
import sqlite3
import tempfile

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import easyocr

# ---------- Page setup ----------
st.set_page_config(page_title="ANPR Trajectory Tracker", layout="wide")
st.title("🚗 City-Wide ANPR Trajectory Tracking — Prototype")
st.caption("Upload camera videos, detect vehicles, read plates, and trace a vehicle's route across cameras.")

with st.expander("ℹ️ About this project — read this first", expanded=True):
    st.markdown("""
**Problem:** Cities have hundreds of CCTV cameras, but each one operates in isolation —
no system connects what one camera sees to what another camera sees. Tracing a specific
vehicle across a city currently means manually scrubbing through hours of disconnected footage.

**What this prototype does:**
1. **Detects** every vehicle in each uploaded video using a pretrained AI model (YOLOv8)
2. **Reads** each vehicle's license plate automatically (EasyOCR)
3. **Logs** every sighting — which plate, which camera, what time
4. **Reconstructs the trajectory** — search any plate and see its full route across all cameras
5. **Surfaces city-wide stats** — busiest camera, total unique vehicles, etc.

**How to try it:** upload 1–2 short videos in the sidebar (each video = one "camera"), click
**Run Detection**, then search a plate number once processing finishes.

*Note: this free hosting runs on CPU, so processing may take a minute or two per video.*
    """)

DB_PATH = "anpr_log.db"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
FRAME_SKIP = 10  # only check every Nth frame — much faster, still enough for a demo


# ---------- Cached model loading (only loads once, not on every click) ----------
@st.cache_resource
def load_models():
    detector = YOLO("yolov8n.pt")
    reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if you have a GPU available
    return detector, reader


def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sightings (
            plate TEXT,
            camera TEXT,
            frame_number INTEGER,
            timestamp_seconds REAL
        )
    """)
    conn.commit()
    return conn


def read_plate_text(reader, crop_image):
    if crop_image.size == 0:
        return None
    results = reader.readtext(crop_image)
    if not results:
        return None
    best = max(results, key=lambda r: len(r[1]))
    text = "".join(ch for ch in best[1].upper() if ch.isalnum())
    return text if len(text) >= 4 else None


def process_video(detector, reader, conn, video_path, camera_name, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_num = 0
    logged = 0
    cur = conn.cursor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % FRAME_SKIP == 0:
            results = detector(frame, verbose=False)[0]
            for box in results.boxes:
                cls_name = detector.names[int(box.cls[0])]
                if cls_name in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    plate = read_plate_text(reader, crop)
                    if plate:
                        timestamp = frame_num / fps
                        cur.execute(
                            "INSERT INTO sightings (plate, camera, frame_number, timestamp_seconds) VALUES (?, ?, ?, ?)",
                            (plate, camera_name, frame_num, round(timestamp, 2)),
                        )
                        logged += 1
        frame_num += 1
        if progress_callback and total_frames:
            progress_callback(min(frame_num / total_frames, 1.0))

    conn.commit()
    cap.release()
    return logged


# ---------- Sidebar: upload + run ----------
st.sidebar.header("1. Upload camera videos")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more videos (each = one camera)",
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True,
)

run_button = st.sidebar.button("▶ Run Detection", type="primary", disabled=not uploaded_files)

if "conn" not in st.session_state:
    st.session_state.conn = None
if "has_data" not in st.session_state:
    st.session_state.has_data = False

if run_button:
    detector, reader = load_models()
    conn = init_db()
    st.session_state.conn = conn

    total_logged = 0
    for uf in uploaded_files:
        camera_name = uf.name
        st.sidebar.write(f"Processing **{camera_name}** ...")
        progress_bar = st.sidebar.progress(0.0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(camera_name)[1]) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        logged = process_video(
            detector, reader, conn, tmp_path, camera_name,
            progress_callback=progress_bar.progress,
        )
        total_logged += logged
        os.remove(tmp_path)
        st.sidebar.success(f"{camera_name}: {logged} plate sightings logged")

    st.session_state.has_data = True
    st.sidebar.info(f"Done — {total_logged} total sightings logged across all videos.")

# ---------- Main area ----------
if not st.session_state.has_data:
    st.info("⬅ Upload videos and click **Run Detection** in the sidebar to get started.")
else:
    conn = st.session_state.conn

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📷 Detected plates")
        all_plates_df = pd.read_sql_query("SELECT DISTINCT plate FROM sightings", conn)
        st.dataframe(all_plates_df, use_container_width=True, height=250)

        st.subheader("🔎 Search a plate")
        query = st.text_input("Enter a full or partial plate number", value="")

    with col2:
        st.subheader("🛣️ Trajectory")
        if query:
            trajectory_df = pd.read_sql_query(
                "SELECT plate, camera, timestamp_seconds FROM sightings WHERE plate LIKE ? ORDER BY timestamp_seconds",
                conn,
                params=(f"%{query}%",),
            )
            if trajectory_df.empty:
                st.warning("No matches found for that plate.")
            else:
                st.dataframe(trajectory_df, use_container_width=True)
                st.caption("Sightings ordered by time — this is the vehicle's reconstructed route across cameras.")
        else:
            st.caption("Type a plate above to see where and when it was spotted.")

    st.divider()
    st.subheader("📊 City-wide summary")
    summary_df = pd.read_sql_query(
        """
        SELECT camera,
               COUNT(*) AS total_sightings,
               COUNT(DISTINCT plate) AS unique_vehicles
        FROM sightings
        GROUP BY camera
        """,
        conn,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(summary_df, use_container_width=True)
    with c2:
        if not summary_df.empty:
            st.bar_chart(summary_df.set_index("camera")["total_sightings"])
