import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Title
st.title("Russian Number Plate Detection on Uploaded Video")
st.write("Upload a video file, and we'll detect Russian number plates using OpenCV.")

# Upload video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Save option
save_video = st.radio("Do you want to save the processed video?", ["No", "Yes"])

if uploaded_file is not None:
    # Temporary input file
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    input_temp.write(uploaded_file.read())

    # Temp file for output video (only if user wants to save)
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') if save_video == "Yes" else None

    # Load cascade
    number_plate = cv2.CascadeClassifier(
        "/Users/pvsairamsaketh/Desktop/opencvudemy/Computer Vision with Python Course/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_russian_plate_number.xml"
    )

    def detect_numberplate(img):
        russian_plate = img.copy()
        gray = cv2.cvtColor(russian_plate, cv2.COLOR_BGR2GRAY)
        face_rects = number_plate.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(russian_plate, (x, y), (x + w, y + h), (0, 0, 255), 10)
        return russian_plate

    # Open video
    cap = cv2.VideoCapture(input_temp.name)

    # Get video properties for saving
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if needed
    if save_video == "Yes":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

    # Display frames
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_numberplate(frame)

        # Save frame if needed
        if save_video == "Yes":
            out.write(processed_frame)

        # Show frame in Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    if save_video == "Yes":
        out.release()
        # Show download button
        with open(output_temp.name, "rb") as file:
            btn = st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_number_plate_video.mp4",
                mime="video/mp4"
            )

    # Clean up temp input file
    os.unlink(input_temp.name)
