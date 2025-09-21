# compatible with python 3.10.13
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# --- App UI ---
st.set_page_config(page_title="Number Plate Detection", layout="wide")
st.title("🚗 Number Plate Detection")

# --- User Guide ---
with st.expander("📖 How to use this app", expanded=True):
    st.markdown("""
    1. **Select whether you want to process a photo or video**.  
    2. **Adjust settings** in the sidebar.  
    3. **Upload the file** using the uploader.  
    4. Click **Start Processing** to detect number plates frame by frame.  
    5. After processing, choose whether you want to save/download the result.  
    """)

# --- Session State ---
for key in ["last_frame", "out_path", "processing_started"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key=="last_frame" else None

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Processing Options")
with st.sidebar.expander("🛠️ Advanced Settings", expanded=True):
    resize_scale = st.slider("🖼️ Resize Scale", 0.3, 1.0, 0.7, step=0.1)
    st.caption("Scales down frames before detection to speed up processing; smaller values are faster but less accurate.")

    scale_factor = st.slider("📏 Scale Factor", 1.01, 1.5, 1.05, step=0.01)
    st.caption("Specifies how much the image size is reduced at each image scale; smaller = more accurate, slower.")

    min_neighbors = st.slider("🔍 Min Neighbors", 1, 15, 6)
    st.caption("Specifies how many neighbors each rectangle should have to retain it; higher = stricter detection.")

show_preview = st.sidebar.checkbox("👁️ Show live frame preview (video only)", value=True)
st.sidebar.caption("Displays processed frames while detection is running; uncheck to improve processing speed.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset App"):
    st.session_state.last_frame = 0
    st.session_state.processing_started = False
    if st.session_state.out_path and os.path.exists(st.session_state.out_path):
        os.unlink(st.session_state.out_path)
    st.session_state.out_path = None
    st.info("✅ Reset successful. Please upload a new file to start again.")
    st.stop()

# --- Select Upload Type ---
upload_type = st.radio("Select upload type:", ["Photo", "Video"], index=None)

# --- Haarcascade classifier ---
number_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

def detect_numberplate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    plates = number_plate.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img

# --- Process based on type ---
if upload_type:

    # --- Photo Upload ---
    if upload_type == "Photo":
        uploaded_file = st.file_uploader("📂 Choose a photo...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                # Resize for detection
                small_img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
                processed_small = detect_numberplate(small_img.copy())
                # Scale back to original size
                processed_img = cv2.resize(processed_small, (img.shape[1], img.shape[0]))
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                         caption="Processed Photo", use_container_width=True)

                save_photo = st.checkbox("💾 Save this photo?")
                if save_photo:
                    _, buffer = cv2.imencode(".png", processed_img)
                    st.download_button("⬇️ Download Processed Photo", buffer.tobytes(), file_name="processed_photo.png")
            else:
                st.error("Could not read the uploaded image.")

    # --- Video Upload ---
    else:
        uploaded_file = st.file_uploader("📂 Choose a video...", type=["mp4", "mov", "avi"])
        tfile = None
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            st.video(tfile.name)
            st.caption("Preview of the uploaded video before processing.")

        if uploaded_file is None:
            st.warning("Please upload a video to enable processing.")
        else:
            if st.button("▶️ Start Processing"):
                st.session_state.processing_started = True

        if uploaded_file is not None and st.session_state.processing_started:
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Failed to open video. Please upload a valid video file.")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width == 0 or height == 0:
                st.error("Error reading video dimensions. Please try another video.")
                st.stop()

            if st.session_state.out_path is None:
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                st.session_state.out_path = out_file.name
                out_file.close()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(st.session_state.out_path, fourcc, fps, (width, height))
            if not out_writer.isOpened():
                st.error("Failed to initialize video writer. Check file permissions.")
                st.stop()

            stframe = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_count = 0
            preview_update_rate = 3

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Resize for detection
                small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                processed_small = detect_numberplate(small_frame.copy())
                # Scale back to original size
                processed_frame = cv2.resize(processed_small, (width, height))

                out_writer.write(processed_frame)

                if show_preview and frame_count % preview_update_rate == 0:
                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)

                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(min(progress, 100))
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")

            cap.release()
            out_writer.release()
            os.unlink(tfile.name)

            st.subheader("🎬 Processed Video")
            st.video(st.session_state.out_path)
            st.caption("This is the fully processed video with number plates highlighted.")

            save_video = st.checkbox("💾 Save this video?")
            if save_video:
                with open(st.session_state.out_path, "rb") as f:
                    st.download_button("⬇️ Download Processed Video", f, file_name="processed_video.mp4")
