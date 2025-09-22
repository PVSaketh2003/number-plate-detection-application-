# compatible with python 3.10.13
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# --- App UI ---
st.set_page_config(page_title="Russian Number Plate Detection", layout="wide")
st.title("🚗 Russian Number Plate Detection")

# --- User Guide ---
with st.expander("📖 How to use this app", expanded=True):
    st.markdown("""
    1. **Select whether you want to process a photo or video**.  
    2. **Adjust settings** in the sidebar.  
    3. **Upload the file** using the uploader.  
    4. Click **Start Processing** to detect number plates frame by frame.  
    5. After processing, download the processed result directly or change parameters and re-run detection.  
    """)

# --- Session State ---
for key in ["uploaded_file", "original_img", "processed_img", "temp_video_path", "processed_video", "file_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Detection Settings (Mandatory)")

resize_scale = st.sidebar.number_input(
    "🖼️ Resize Scale (0.1 – 1.0)",
    min_value=0.1, max_value=1.0, step=0.01,
    format="%.2f"
)
st.sidebar.caption("Scales down frames before detection; smaller = faster but less accurate.")

scale_factor = st.sidebar.number_input(
    "📏 Scale Factor (1.01 – 1.5)",
    min_value=1.01, max_value=1.5, step=0.01,
    format="%.2f"
)
st.sidebar.caption("Controls pyramid scaling; smaller = more accurate but slower.")

min_neighbors = st.sidebar.number_input(
    "🔍 Min Neighbors (1 – 15)",
    min_value=1, max_value=15, step=1
)
st.sidebar.caption("Sets how many nearby detections are required; higher = stricter detection.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset App"):
    for key in ["uploaded_file", "original_img", "processed_img", "temp_video_path", "processed_video", "file_type"]:
        st.session_state[key] = None
    st.info("✅ Reset successful. Please upload a new file to start again.")
    st.stop()

# Validate sidebar inputs
if resize_scale is None or scale_factor is None or min_neighbors is None:
    st.warning("⚠️ Please set values for Resize Scale, Scale Factor, and Min Neighbors in the sidebar.")
    st.stop()

# --- Select Upload Type ---
upload_type = st.radio("Select upload type:", ["Photo", "Video"], index=None)

# --- Haarcascade classifier ---
number_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# --- Object Detection Function (Logic unchanged) ---
def detect_numberplate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    plates = number_plate.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img, len(plates)

# -------------------------
# PHOTO UPLOAD & PROCESSING
# -------------------------
if upload_type == "Photo":
    uploaded_file = st.file_uploader("📂 Choose a photo...", type=["jpg", "jpeg", "png"], key="photo_uploader")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_type = "photo"
        
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not read the uploaded image.")
            st.stop()
        
        st.session_state.original_img = img
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Photo", use_container_width=True)

        if st.button("▶️ Start Detection"):
            # Resize for detection
            small_img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
            processed_small, count = detect_numberplate(small_img.copy())
            processed_img = cv2.resize(processed_small, (img.shape[1], img.shape[0]))
            st.session_state.processed_img = processed_img
            st.session_state.detect_count = count

        if st.session_state.processed_img is not None:
            st.image(cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_BGR2RGB),
                     caption=f"Processed Photo - Plates Detected: {st.session_state.detect_count}", use_container_width=True)
            if st.button("💾 Save Processed Image"):
                _, buffer = cv2.imencode(".png", st.session_state.processed_img)
                st.download_button(
                    "⬇️ Download Processed Image",
                    buffer.tobytes(),
                    file_name="processed_photo.png"
                )

            if st.button("🔄 Re-run Detection with New Parameters"):
                small_img = cv2.resize(st.session_state.original_img, (0,0), fx=resize_scale, fy=resize_scale)
                processed_small, count = detect_numberplate(small_img.copy())
                processed_img = cv2.resize(processed_small, (st.session_state.original_img.shape[1], st.session_state.original_img.shape[0]))
                st.session_state.processed_img = processed_img
                st.session_state.detect_count = count
                st.experimental_rerun()

# -------------------------
# VIDEO UPLOAD & PROCESSING
# -------------------------
if upload_type == "Video":
    uploaded_file = st.file_uploader("📂 Choose a video...", type=["mp4", "mov", "avi"], key="video_uploader")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_type = "video"
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        st.session_state.temp_video_path = tfile.name
        st.video(tfile.name)
        st.caption("Preview of the uploaded video before processing.")

        if st.button("▶️ Start Detection"):
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not out_writer.isOpened():
                st.error("Failed to initialize video writer.")
                st.stop()

            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                processed_small, _ = detect_numberplate(small_frame.copy())
                processed_frame = cv2.resize(processed_small, (width, height))
                out_writer.write(processed_frame)

                if frame_count % 3 == 0:
                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out_writer.release()
            st.session_state.processed_video = out_path

        if st.session_state.processed_video:
            st.subheader("🎬 Processed Video")
            st.video(st.session_state.processed_video)
            if st.button("💾 Save Processed Video"):
                with open(st.session_state.processed_video, "rb") as f:
                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            if st.button("🔄 Re-run Detection with New Parameters"):
                cap = cv2.VideoCapture(st.session_state.temp_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                stframe = st.empty()
                progress_bar = st.progress(0)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                    processed_small, _ = detect_numberplate(small_frame.copy())
                    processed_frame = cv2.resize(processed_small, (width, height))
                    out_writer.write(processed_frame)

                    if frame_count % 3 == 0:
                        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out_writer.release()
                st.session_state.processed_video = out_path
                st.experimental_rerun()
