# compatible with python 3.10.13
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# --- App UI ---
st.set_page_config(page_title="Russian Number Plate Detection", layout="centered")
st.title("🚗 Russian Number Plate Detection")
st.write("Upload a video and detect Russian number plates using **OpenCV**.")

# --- Session State Setup ---
if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False
if "resume_processing" not in st.session_state:
    st.session_state.resume_processing = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = 0
if "out_path" not in st.session_state:
    st.session_state.out_path = None

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Haarcascade classifier
    number_plate = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    )

    def detect_numberplate(img):
        """Detect Russian number plates and draw rectangles."""
        russian_plate = img.copy()
        gray = cv2.cvtColor(russian_plate, cv2.COLOR_BGR2GRAY)
        face_rects = number_plate.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(russian_plate, (x, y), (x + w, y + h), (0, 0, 255), 10)
        return russian_plate

    # --- Video Properties ---
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resume from last frame
    if st.session_state.resume_processing and st.session_state.last_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.last_frame)

    # Prepare output video (always generate so Preview can later Download)
    if st.session_state.out_path is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        st.session_state.out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_writer = cv2.VideoWriter(st.session_state.out_path, fourcc, fps, (width, height))

    # --- UI Controls ---
    st.subheader("⚙️ Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏹ Stop"):
            st.session_state.stop_processing = True
            st.session_state.resume_processing = False
    with col2:
        if st.button("▶️ Resume"):
            st.session_state.resume_processing = True
            st.session_state.stop_processing = False
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.stop_processing = False
            st.session_state.resume_processing = False
            st.session_state.last_frame = 0
            if st.session_state.out_path and os.path.exists(st.session_state.out_path):
                os.unlink(st.session_state.out_path)
            st.session_state.out_path = None
            st.success("✅ Reset successful. Upload a new video to start again.")
            st.stop()

    # --- Placeholders ---
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # --- Processing Loop ---
    while cap.isOpened():
        if st.session_state.stop_processing:
            st.session_state.last_frame = frame_count
            status_text.text(f"⏹ Processing stopped at frame {frame_count}/{total_frames}.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect plates
        frame = detect_numberplate(frame)

        # Show live preview
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Save to processed video
        out_writer.write(frame)

        # Progress bar update
        if total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")

    cap.release()
    out_writer.release()
    os.unlink(tfile.name)

    # --- Final UI ---
    if frame_count == total_frames:
        st.success("✅ Processing complete!")

    # Always show video player + download button
    st.subheader("🎬 Processed Video")
    st.video(st.session_state.out_path)

    with open(st.session_state.out_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="processed_video.mp4")
