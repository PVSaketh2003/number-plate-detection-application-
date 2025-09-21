# app.py
# Compatible with Python 3.10.13
import streamlit as st
import cv2
import tempfile
import os
import numpy as np

st.title("Number Plate Detection from Video")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Open video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Temporary file for output if saved
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Example: draw dummy box (simulate detection)
        cv2.rectangle(frame, (100, 100), (300, 150), (0, 255, 0), 2)
        cv2.putText(frame, "Number Plate", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

        # Convert BGR to RGB for Streamlit display
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()

    # Option to download
    with open(out_file.name, "rb") as f:
        st.download_button("💾 Download Processed Video", f, file_name="detected_video.mp4")
