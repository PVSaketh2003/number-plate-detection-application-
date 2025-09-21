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
    1. **Upload a video** using the file uploader below.  
    2. **Adjust settings** in the sidebar (resize scale affects speed/accuracy).  
    3. Once uploaded, you’ll see a **preview of your video**.  
    4. The app will process frame by frame and show a **live preview with progress bar**.  
    5. Use **Stop, Resume, or Reset** controls to manage processing.  
    6. After completion, watch the **processed video** and **download it** if needed.  
    """)

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
st.caption("Upload a video file to begin processing.")

# --- Sidebar: Show message if no video uploaded ---
if uploaded_file is None:
    st.sidebar.header("⚙️ Processing Options")
    st.sidebar.info("Upload a video to access processing options.")
else:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    # --- Upload video with progress ---
    st.info("📤 Uploading video...")
    upload_progress = st.progress(0)
    chunk_size = 1024 * 1024
    uploaded_file.seek(0, os.SEEK_END)
    total_size = uploaded_file.tell()
    uploaded_file.seek(0)
    bytes_written = 0
    while True:
        chunk = uploaded_file.read(chunk_size)
        if not chunk:
            break
        tfile.write(chunk)
        bytes_written += len(chunk)
        upload_progress.progress(min(int(bytes_written / total_size * 100), 100))
    tfile.flush()
    st.success("✅ Upload complete!")
    st.video(tfile.name)
    st.caption("Preview of the uploaded video before processing.")

    # --- Sidebar Controls (Shown Only After Upload) ---
    st.sidebar.header("⚙️ Processing Options")
    st.sidebar.markdown("Customize detection speed and accuracy for your video:")

    resize_scale = st.sidebar.slider(
        "🖼️ Resize Scale",
        0.3, 1.0, 0.7, step=0.1,
        help="Smaller = faster processing, less accuracy. Larger = slower, more accurate."
    )
    st.sidebar.markdown("<small>Scale down frames before detection to improve speed.</small>", unsafe_allow_html=True)

    min_neighbors = st.sidebar.slider(
        "🔍 Detection Strictness",
        3, 10, 6,
        help="Higher value = stricter detection, fewer false positives."
    )
    st.sidebar.markdown("<small>Controls how many neighbors a rectangle needs to be considered a valid detection.</small>", unsafe_allow_html=True)

    show_preview = st.sidebar.checkbox("👁️ Show live frame preview", value=True)

    st.sidebar.markdown("---")
    with st.sidebar.expander("🛠️ Advanced Options", expanded=False):
        st.markdown("- Advanced detection parameters can go here.\n- e.g., different cascade models, logging, preprocessing.")

    if st.sidebar.button("🔄 Reset App"):
        st.session_state.stop_processing = False
        st.session_state.resume_processing = False
        st.session_state.last_frame = 0
        if st.session_state.out_path and os.path.exists(st.session_state.out_path):
            os.unlink(st.session_state.out_path)
        st.session_state.out_path = None
        st.experimental_rerun()
    st.sidebar.caption("Resets progress and allows uploading a new video.")

    # --- Haarcascade classifier ---
    number_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    def detect_numberplate(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return number_plate.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=min_neighbors, minSize=(30, 30))

    # --- Video Properties ---
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if st.session_state.resume_processing and st.session_state.last_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.last_frame)

    if st.session_state.out_path is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        st.session_state.out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_writer = cv2.VideoWriter(st.session_state.out_path, fourcc, fps, (width, height))

    # --- Controls ---
    st.subheader("🎛️ Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏹ Stop"):
            st.session_state.stop_processing = True
            st.session_state.resume_processing = False
        st.caption("Stop processing at the current frame.")
    with col2:
        if st.button("▶️ Resume"):
            st.session_state.resume_processing = True
            st.session_state.stop_processing = False
        st.caption("Resume processing from where it was stopped.")
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.stop_processing = False
            st.session_state.resume_processing = False
            st.session_state.last_frame = 0
            if st.session_state.out_path and os.path.exists(st.session_state.out_path):
                os.unlink(st.session_state.out_path)
            st.session_state.out_path = None
            st.success("✅ Reset successful. Upload a new video to start again.")
            st.caption("Clears progress and allows uploading a new video.")  # 1-line explanation
            st.stop()

    # --- Placeholders ---
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    preview_update_rate = 3  # update st.image every 3 frames

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
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        plates = detect_numberplate(small_frame)

        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        for (x, y, w, h) in plates:
            cv2.rectangle(frame,
                          (int(x * scale_x), int(y * scale_y)),
                          (int((x + w) * scale_x), int((y + h) * scale_y)),
                          (0, 0, 255), 2)

        out_writer.write(frame)

        if show_preview and frame_count % preview_update_rate == 0:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

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

    st.subheader("🎬 Processed Video")
    st.video(st.session_state.out_path)
    st.caption("This is the fully processed video with number plates highlighted.")

    with open(st.session_state.out_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="processed_video.mp4")
        st.caption("Download the processed video for offline use.")
