# Project: LogoSense Analytics
# File: app.py
# Description: 
# This is the main application file for LogoSense Analytics, 
# a Streamlit-based web app for real-time logo detection in videos.
# It allows users to upload a video file (MP4, AVI, MOV) or provide a YouTube link, 
# configure detection settings, and display analytics like logo durations, 
# logo frequencies, video metadata, and system performance.
# Integrated with a Django API for frame data.

import streamlit as st
import cv2
from logos_detection import detect_logos
from datetime import datetime, timedelta
import os
import tempfile
import yt_dlp
import validators
import requests
from pathlib import Path

# Configure page and grayish theme
st.set_page_config(page_title="LogoSense Analytics", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #e5e7eb; color: #1f2937; }
    .sidebar .sidebar-content { background-color: #374151; color: #d1d5db; }
    .stButton>button {
        background-color: #6b7280;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover { background-color: #4b5563; }
    h1, h2, h3 { color: #1f2937; font-family: 'Arial', sans-serif; }
    .stMarkdown { font-family: 'Arial', sans-serif; }
    .stat-box {
        background-color: #f3f4f6;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        text-align: left;
        width: 100%;
        box-sizing: border-box;
        overflow: hidden;
    }
    .stat-value {
        font-size: 16px;
        margin-top: 5px;
        word-wrap: break-word;
    }
    .stFileUploader label, .stSlider label, .stRadio label, .stTextInput label { color: #d1d5db; font-weight: 500; }
    .stDownloadButton>button {
        background-color: #6b7280;
        color: white;
        border-radius: 8px;
        font-weight: 500;
    }
    .stDownloadButton>button:hover { background-color: #4b5563; }
    .stSpinner > div > div { border-color: #6b7280 transparent transparent transparent; }
    .stProgress > div > div { background-color: #6b7280; }
    .chart-container {
        min-height: 450px;
        max-width: 1200px;
        position: relative;
        transition: none;
        overflow-x: auto;
        overflow-y: hidden;
        display: block;
    }
    .chart-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        padding: 15px;
        border: 1px solid #d1d5db;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logo_stats' not in st.session_state:
    st.session_state.logo_stats = {}
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def main():
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Control Panel")
        st.markdown("**Configure LogoSense Analytics**")
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Set the minimum confidence for logo detection."
        )
        save_output = st.radio(
            "Save Processed Video?",
            ("No", "Yes"),
            help="Save the output video with logo detections."
        )
        nosave = save_output == "No"
        
        # Input method selection
        input_method = st.radio(
            "Video Input Method",
            ("Upload Video", "YouTube Link"),
            help="Choose to upload a video file or provide a YouTube link."
        )
        
        video_file = None
        youtube_url = None
        if input_method == "Upload Video":
            allowed_types = ["mp4", "avi", "mov", "mpeg", "m4v"]  # Expanded video formats
            video_file = st.file_uploader(
                "🎬 Upload Video",
                type=allowed_types,
                help="Upload an MP4, AVI, MOV, MPEG, or M4V file.",
                key="video_uploader"
            )
        else:
            youtube_url = st.text_input(
                "📺 YouTube Video URL",
                placeholder="e.g., https://www.youtube.com/watch?v=example",
                help="Enter a valid YouTube video URL."
            )

        # Login for Django API
        st.markdown("**Django API Login**")
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        if st.button("Login"):
            login(username, password)

    # Main layout
    st.title("🎥 LogoSense Analytics")
    st.markdown("**Real-time logo detection and analytics for your video content.**")

    # Containers for sections
    video_stream_container = st.container()
    logo_duration_container = st.container()
    logo_frequency_container = st.container()
    system_performance_container = st.container()
    metadata_container = st.container()

    # Django API functions
    def login(username, password):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/login/",
                json={"username": username, "password": password},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.user_data = {
                    "token": data.get("token"),
                    "is_section_admin": data.get("is_section_admin", False),
                    "is_hq_admin": data.get("is_hq_admin", False),
                    "section": data.get("section"),
                    "expires_at": data.get("expires_at")
                }
                st.success("Login successful!")
            else:
                st.error(f"Login failed: {response.json().get('error', 'Unknown error')}")
        except requests.RequestException as e:
            st.error(f"Login failed: {str(e)}")

    def check_and_renew_token():
        user_data = st.session_state.user_data
        token = user_data.get("token")
        expires_at = user_data.get("expires_at")
        if not token or not expires_at:
            return False
        try:
            expires_at_dt = datetime.fromisoformat(expires_at)
            if expires_at_dt - datetime.utcnow() < timedelta(minutes=5):
                response = requests.post(
                    "http://127.0.0.1:8000/api/renew-token/",
                    headers={"Authorization": f"Token {token}"},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.user_data.update({
                        "token": data.get("token"),
                        "expires_at": data.get("expires_at")
                    })
                else:
                    return False
        except (ValueError, requests.RequestException):
            return False
        return True

    def fetch_frames():
        if not check_and_renew_token():
            st.error("Session expired. Please log in again.")
            return None
        token = st.session_state.user_data.get("token")
        try:
            response = requests.get(
                "http://127.0.0.1:8000/api/frames/",
                headers={"Authorization": f"Token {token}"},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("Session expired. Please log in again.")
                return None
            else:
                st.error(f"Failed to fetch frames: {response.status_code}")
                return None
        except requests.RequestException as e:
            st.error(f"Error fetching frames: {str(e)}")
            return None

    # Process video
    def get_youtube_stream_url(url):
        """Get the direct streaming URL for a YouTube video."""
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',  # Prefer MP4 format
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info['url']
        except Exception as e:
            st.error(f"Failed to fetch YouTube stream: {str(e)}. Please check the URL or try another video.")
            return None

    def get_video_metadata(video_source):
        """Extract metadata from the video."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "Resolution": f"{width}x{height}",
            "Duration": f"{duration:.2f} sec",
            "FPS": f"{fps:.1f}"
        }

    if st.sidebar.button("🚀 Start Analysis"):
        video_source = None
        temp_file = None
        if input_method == "Upload Video" and video_file:
            # Validate file extension
            try:
                file_extension = Path(video_file.name).suffix.lower().lstrip(".")
                allowed_types = ["mp4", "avi", "mov", "mpeg", "m4v"]
                if file_extension not in allowed_types:
                    st.error(f"Invalid file type: {file_extension}. Please upload a {', '.join(allowed_types)} file.")
                    st.session_state.uploaded_file = None
                    st.rerun()
                else:
                    # Handle uploaded video
                    temp_file = os.path.join(tempfile.gettempdir(), f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}")
                    with open(temp_file, "wb") as f:
                        f.write(video_file.read())
                    video_source = temp_file
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                st.session_state.uploaded_file = None
                st.rerun()
        elif input_method == "YouTube Link" and youtube_url:
            # Validate and get YouTube stream URL
            if not validators.url(youtube_url):
                st.error("Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=example).")
                return
            with st.spinner("Fetching YouTube video stream..."):
                video_source = get_youtube_stream_url(youtube_url)
            if not video_source:
                return
        else:
            st.error("Please upload a video file or provide a valid YouTube URL.")
            return

        if video_source:
            # Reset session state
            st.session_state.logo_stats = {}

            # Show sections when analysis starts
            with video_stream_container:
                st.subheader("Video Stream")
                stframe = st.empty()
                st.markdown(
                    "<p style='text-align: center; color: #6b7280;'>Live logo detections</p>",
                    unsafe_allow_html=True
                )

            with logo_duration_container:
                st.subheader("Logo Duration")
                duration_container = st.container()

            with logo_frequency_container:
                st.subheader("Logo Frequency")
                chart_placeholder = st.empty()

            with system_performance_container:
                st.subheader("System Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fps_container = st.container()
                with col2:
                    mem_container = st.container()
                with col3:
                    cpu_container = st.container()

            with metadata_container:
                st.subheader("Video Metadata")
                metadata_inner_container = st.container()
                # Display video metadata
                metadata = get_video_metadata(video_source)
                if metadata:
                    with metadata_inner_container:
                        for key, value in metadata.items():
                            st.markdown(f"<div class='stat-box'><b>{key}</b><br><span class='stat-value'>{value}</span></div>", unsafe_allow_html=True)
                else:
                    with metadata_inner_container:
                        st.markdown("<div class='stat-box'><b>No metadata available</b><br><span class='stat-value'>Unable to retrieve video metadata.</span></div>", unsafe_allow_html=True)

            # Check model file
            model_path = os.path.join("model", "best.pt")
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}. Please ensure 'model/best.pt' exists.")
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                return

            with st.spinner("Processing video with GPU acceleration..."):
                st.write("Debug: Starting analysis for video source")
                logo_stats = detect_logos(
                    source=video_source,
                    model_path=model_path,
                    stframe=stframe,
                    duration_container=duration_container,
                    fps_container=fps_container,
                    mem_container=mem_container,
                    cpu_container=cpu_container,
                    chart_placeholder=chart_placeholder,
                    conf_thres=conf_thres,
                    nosave=nosave
                )
            st.success("Processing Complete! All logos have been analyzed.")
            
            # Debug logo_stats
            with logo_frequency_container:
                if logo_stats:
                    st.write(f"Debug: Final logo_stats = {logo_stats}")
                else:
                    st.warning("No logos detected. Frequency chart cannot be displayed. Try lowering the confidence threshold (e.g., 0.1) or using a YOLO model trained for logos in this video (e.g., Coca-Cola, FIFA).")

            # Clean up temporary file for uploaded video
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            # Final download option
            if not nosave and os.path.exists("output.mp4"):
                with open("output.mp4", "rb") as f:
                    st.download_button(
                        label="📹 Download Processed Video",
                        data=f,
                        file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )

            # Display frames from Django API
            if st.session_state.user_data.get("token"):
                frames = fetch_frames()
                if frames:
                    st.subheader("Frame Analysis")
                    for frame in frames:
                        st.write(f"Frame ID: {frame['id']}")
                        if frame.get("image_raw"):
                            try:
                                temp_path = "temp_frame.jpg"
                                urllib.request.urlretrieve(frame["image_raw"], temp_path)
                                img = cv2.imread(temp_path)
                                if img is not None:
                                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Frame {frame['id']}")
                            except Exception as e:
                                st.error(f"Error processing frame {frame['id']}: {str(e)}")

if __name__ == "__main__":
    main()
