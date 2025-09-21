# compatible with python 3.10.13
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Title and description
st.title("Russian Number Plate Detection")
st.write("Upload a video file to detect Russian number plates using OpenCV.")

# Upload video file
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Load the cascade classifier (using OpenCV's built-in data path)
    number_plate = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    )

    def detect_numberplate(img):
        russian_plate = img.copy()
        gray = cv2.cvtColor(russian_plate, cv2.COLOR_BGR2GRAY)
        face_rects = number_plate.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(russian_plate, (x, y), (x + w, y + h), (0, 0, 255), 10)
        return russian_plate

    # Open video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    frame_skip = 2  # process every 2nd frame (faster without losing much quality)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # skip frame to save processing time

        # Resize frame for speed (reduce to half resolution)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Detect number plate in frame
        frame = detect_numberplate(frame)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show frame in Streamlit
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    os.unlink(tfile.name)  # Optional: delete temp file after use
