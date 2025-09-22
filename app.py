import cv2
import numpy as np
import streamlit as st
from io import BytesIO
import tempfile
import os

# -------------------------
# Number plate detection function
# -------------------------
def detect_numberplate(frame, plate_cascade, scale_factor, min_neighbors):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# -------------------------
# Streamlit UI
# -------------------------
st.title("Number Plate Detection App 🚗")

# Sidebar options
st.sidebar.header("⚙️ Detection Settings")

resize_scale = st.sidebar.slider(
    "Resize Scale", 0.1, 1.0, 0.5, 0.1,
    help="Scales down frames before detection to speed up processing; smaller values are faster but less accurate."
)
scale_factor = st.sidebar.slider(
    "Scale Factor", 1.01, 1.5, 1.1, 0.01,
    help="Parameter specifying how much the image size is reduced at each image scale."
)
min_neighbors = st.sidebar.slider(
    "Min Neighbors", 1, 10, 5, 1,
    help="Specifies how many neighbors each candidate rectangle should have to retain it."
)

# File uploader
file_type = st.radio("Choose what to upload:", ("Image", "Video"), index=None)
uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Load cascade
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# -------------------------
# IMAGE PROCESSING
# -------------------------
if uploaded_file and file_type == "Image":
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize for detection
    small_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
    processed_small = detect_numberplate(small_img, plate_cascade, scale_factor, min_neighbors)

    # Scale back to original size
    processed_img = cv2.resize(processed_small, (img.shape[1], img.shape[0]))

    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
             caption="Processed Photo", use_container_width=True)

    # Save option
    if st.button("Save Image"):
        success, buffer = cv2.imencode(".jpg", processed_img)
        if success:
            st.download_button(
                label="Download Processed Image",
                data=buffer.tobytes(),
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )

# -------------------------
# VIDEO PROCESSING
# -------------------------
if uploaded_file and file_type == "Video":
    # Save uploaded video to temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary output file
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    progress = st.progress(0)
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for detection
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        processed_small = detect_numberplate(small_frame, plate_cascade, scale_factor, min_neighbors)

        # Scale back
        processed_frame = cv2.resize(processed_small, (width, height))
        out.write(processed_frame)

        current_frame += 1
        progress.progress(min(current_frame / frame_count, 1.0))

    cap.release()
    out.release()

    # Load processed video into memory (for instant download)
    with open(out_path, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes)

    if st.button("Save Video"):
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Cleanup temp files
    os.remove(out_path)
    os.remove(tfile.name)
