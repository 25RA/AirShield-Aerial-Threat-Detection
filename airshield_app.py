import streamlit as st
import cv2
import librosa
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os

# --- Configuration & Model Loading (Crucial to do this only once) ---
# Use Streamlit's cache decorator to load the model only once
@st.cache_resource
def load_models():
    # 1. Load the trained Audio Classifier (Random Forest)
    try:
        audio_model = joblib.load('models/audio_drone_detector.pkl')
    except FileNotFoundError:
        st.error("Audio Model Error: 'models/audio_drone_detector.pkl' not found. Did you run train_audio.py?")
        audio_model = None
    
    # 2. Load the YOLOv8 Video Detector
    try:
        # Note: I corrected the path separator to be OS-agnostic
        video_model = YOLO(os.path.join('models', 'Final_drone.pt')) 
    except Exception as e:
        st.error(f"Video Model Error: Could not load YOLO model. Check path and file: {e}")
        video_model = None

    return audio_model, video_model

audio_model, video_model = load_models()

# --- 1. AUDIO CLASSIFICATION LOGIC (Uses the trained model) ---
def extract_audio_features(audio_path, duration=3):
    """Extracts MFCC features in the same way the model was trained."""
    try:
        # Load the uploaded file data using a BytesIO object
        y, sr = librosa.load(audio_path, sr=22050, duration=duration) 
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Transpose and take the mean across the time axis (axis=0)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_audio(audio_file):
    if audio_model is None:
        return "Model Unavailable"
        
    features = extract_audio_features(audio_file)
    if features is None or features.size == 0:
        return "Feature extraction failed"
        
    # Predict requires a 2D array: [[feature_1, feature_2, ...]]
    prediction = audio_model.predict(features.reshape(1, -1))
    
    # Map classID (0 or 1) back to class name
    class_name = "Drone Detected" if prediction[0] == 1 else "Normal (Non-Drone)"
    
    return class_name

# --- 2. VIDEO DETECTION + FLIGHT PATH LOGIC ---
def process_video_and_track(video_path):
    if video_model is None:
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file. Check file integrity or format.")
        return

    st_frame = st.empty() # Placeholder for the video feed
    trajectory_data = [] 
    frame_count = 0
    
    progress_bar = st.progress(0, text="Processing video frames...")
    
    # Run analysis for a short duration to keep demo quick
    max_frames = 300 

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # Run YOLOv8 detection and tracking (using the default ByteTrack)
        # Set a confidence threshold (e.g., 0.5) to avoid clutter
        results = video_model.track(frame, persist=True, conf=0.5)
        annotated_frame = results[0].plot()
        
        # --- Flight Path Data Capture ---
        if results[0].boxes:
            # We assume the first detected box is the drone
            box = results[0].boxes.xyxy.cpu().numpy()[0]
            
            # Calculate the center (x_center, y_center)
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            trajectory_data.append({
                'frame': frame_count,
                'x_coord': x_center,
                'y_coord': y_center,
            })

        # Display in Streamlit
        st_frame.image(annotated_frame, channels="BGR")
        
        frame_count += 1
        progress_bar.progress(min(frame_count / max_frames, 1.0))

    cap.release()
    progress_bar.empty()
    st.success("Video Analysis Complete.")
    
    # --- Plot the Trajectory ---
    if trajectory_data:
        df_trajectory = pd.DataFrame(trajectory_data)
        st.subheader("✈️ Flight Path Simulation (2D Trajectory)")
        
        # Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_trajectory['x_coord'], df_trajectory['y_coord'], 
                marker='o', linestyle='-', color='red', markersize=3, alpha=0.7)
        ax.set_title("Drone Movement Across Frame Coordinates")
        ax.set_xlabel("X-Coordinate (Horizontal)")
        ax.set_ylabel("Y-Coordinate (Vertical)")
        ax.invert_yaxis() # Y=0 is at the top of the video frame
        ax.grid(True, alpha=0.3)
        st.pyplot(fig) # Display the plot in Streamlit

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AirShield AI", layout="wide")
st.title("🛡️ AirShield: AI Malicious Drone Detection")

col1, col2 = st.columns(2)

with col1:
    st.header("🎧 Audio Analysis")
    audio_file = st.file_uploader("Upload Drone Sound (WAV)", type=['wav'])
    if audio_file:
        st.audio(audio_file)
        if st.button("Classify Audio"):
            with st.spinner('Analyzing audio file...'):
                result = predict_audio(audio_file)
                st.info(f"Analysis Result: **{result}**")

with col2:
    st.header("📹 Video Surveillance")
    # Using 'mp4' as the type, assuming conversion (or new files) was done
    video_file = st.file_uploader("Upload Surveillance Footage (MP4 format is best)", type=['mp4', 'avi', 'mov'])
    if video_file:
        # Display video player before analysis
        st.video(video_file) 
        
        if st.button("Start Video Analysis"):
            with st.spinner('Running YOLOv8 Detection and Tracking...'):
                # Save temp file to process
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.read())
                
                # Start the combined detection and tracking
                process_video_and_track(temp_video_path)
                
                # Clean up the temp file
                os.remove(temp_video_path)