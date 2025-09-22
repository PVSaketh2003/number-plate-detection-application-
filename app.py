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
    1. **Enter detection parameters in the sidebar and click Submit**.  
    2. **Select whether you want to process a photo or video**.  
    3. **Upload the file** using the uploader.  
    4. Click **Start Detection** to detect number plates frame by frame.  
    5. After processing, download the processed result directly.  
    """)

# --- Session State ---
for key in ["params_submitted", "uploaded_file", "original_img", "processed_img", "processed_video", "file_type", "temp_video_path", "resize_scale", "scale_factor", "min_neighbors"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Detection Settings (Mandatory)")

resize_scale_input = st.sidebar.selectbox(
    "🖼️ Resize Scale (0.1 – 1.0)",
    options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
)
scale_factor_input = st.sidebar.number_input(
    "📏 Scale Factor (1.01 – 1.5)",
    min_value=1.01, max_value=1.5, step=0.01,
    format="%.2f"
)
min_neighbors_input = st.sidebar.number_input(
    "🔍 Min Neighbors (1 – 10)",
    min_value=1, max_value=10, step=1
)

if st.sidebar.button("✅ Submit Parameters"):
    st.session_state.params_submitted = True
    st.session_state.resize_scale = resize_scale_input
    st.session_state.scale_factor = scale_factor_input
    st.session_state.min_neighbors = min_neighbors_input
    st.sidebar.success("Parameters submitted successfully!")

# Only show upload type selection after parameters are submitted
if st.session_state.params_submitted:
    # --- Select Upload Type ---
    upload_type = st.radio("Select upload type:", ["Photo", "Video"], index=None)

    # --- Haarcascade classifier ---
    number_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    # --- Object Detection Function (Your Logic Intact) ---
    def detect_numberplate(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        plates = number_plate.detectMultiScale(
            gray,
            scaleFactor=st.session_state.scale_factor,
            minNeighbors=st.session_state.min_neighbors
        )
        for (x, y, w, h) in plates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img

    # -------------------------
    # PHOTO UPLOAD & PROCESSING
    # -------------------------
    if upload_type == "Photo":
        uploaded_file = st.file_uploader("📂 Choose a photo...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_type = "photo"
            
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not read the uploaded image.")
                st.stop()
            
            st.session_state.original_img = img

            if st.button("▶️ Start Detection"):
                small_img = cv2.resize(img, (0,0), fx=st.session_state.resize_scale, fy=st.session_state.resize_scale)
                processed_small = detect_numberplate(small_img.copy())
                processed_img = cv2.resize(processed_small, (img.shape[1], img.shape[0]))
                st.session_state.processed_img = processed_img

            # Show processed image if exists
            if st.session_state.processed_img is not None:
                st.image(cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_BGR2RGB),
                         caption="Processed Photo", use_container_width=True)
                if st.button("💾 Save Processed Image"):
                    _, buffer = cv2.imencode(".png", st.session_state.processed_img)
                    st.download_button(
                        "⬇️ Download Processed Image",
                        buffer.tobytes(),
                        file_name="processed_photo.png"
                    )

    # -------------------------
    # VIDEO UPLOAD & PROCESSING
    # -------------------------
    if upload_type == "Video":
        uploaded_file = st.file_uploader("📂 Choose a video...", type=["mp4", "mov", "avi"])
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
                    small_frame = cv2.resize(frame, (0, 0), fx=st.session_state.resize_scale, fy=st.session_state.resize_scale)
                    processed_small = detect_numberplate(small_frame.copy())
                    processed_frame = cv2.resize(processed_small, (width, height))
                    out_writer.write(processed_frame)

                    if frame_count % 3 == 0:
                        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out_writer.release()
                st.session_state.processed_video = out_path

            # Show processed video if exists
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
