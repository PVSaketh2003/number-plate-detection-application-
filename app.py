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

    # Load the Haar cascade classifier (using OpenCV's built-in path)
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    number_plate = cv2.CascadeClassifier(cascade_path)

    def detect_numberplate(img):
        russian_plate = img.copy()
        gray = cv2.cvtColor(russian_plate, cv2.COLOR_BGR2GRAY)
        plates = number_plate.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in plates:
            cv2.rectangle(russian_plate, (x, y), (x + w, y + h), (0, 0, 255), 3)
        return russian_plate

    # Open video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect number plate in frame
        frame = detect_numberplate(frame)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show frame in Streamlit
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    os.unlink(tfile.name)  # Delete temp file after use
