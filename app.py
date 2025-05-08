# Project: LogoSense Analytics
# File: app.py
# Description:
# This is the main Streamlit application file for LogoSense Analytics.
# It provides a professional user interface for processing videos (local uploads or real-time YouTube streams),
# uploading a custom YOLOv8 model (.pt file), setting a confidence threshold,
# and displaying real-time logo detection results with video metadata at the bottom, below system stats, shown after analysis.
# YouTube streaming starts only when "Start Analysis" is clicked, with the loader in the main body.
# Supports saving the processed video with a download button.
# Uses professional emojis/icons for enhanced UI.

import streamlit as st
import yt_dlp
import os
import cv2
import tempfile
import validators
from datetime import datetime
from logos_detection import detect_logos

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #3498db;
        color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .chart-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        max-width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        position: relative;
        display: block;
    }
    .stat-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-card h3 {
        margin: 0;
        font-size: 16px;
        color: #2c3e50;
    }
    .stat-card p {
        margin: 5px 0 0;
        font-size: 20px;
        font-weight: bold;
        color: #3498db;
    }
    .metadata-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #dfe6e9;
        margin-top: 20px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .metadata-card h3 {
        margin: 0 0 15px 0;
        font-size: 18px;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }
    .metadata-card table {
        width: 100%;
        border-collapse: collapse;
    }
    .metadata-card th, .metadata-card td {
        padding: 10px;
        text-align: left;
        font-size: 14px;
        color: #2c3e50;
    }
    .metadata-card th {
        font-weight: bold;
        width: 40%;
    }
    .metadata-card td {
        font-weight: normal;
        color: #3498db;
    }
    .metadata-card tr:not(:last-child) {
        border-bottom: 1px solid #dfe6e9;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Roboto', sans-serif;
    }
    .stSlider > div > div > div > div {
        background-color: #3498db;
    }
    .stTextInput > div > div {
        background-color: #ffffff;
        border-radius: 5px;
        border: 1px solid #dfe6e9;
    }
    .stFileUploader label, .stRadio label {
        color: #111111;
        font-weight: 500;
    }
    .stDownloadButton>button {
        background-color: #3498db;
        color: #ffffff;
        border-radius: 5px;
        font-weight: bold;
    }
    .stDownloadButton>button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Function to get YouTube stream URL
def get_youtube_stream_url(url):
    try:
        with st.spinner("📡 Fetching YouTube video stream..."):
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
                'no_warnings': True,
                'simulate': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    st.error("No video information retrieved. The video may be unavailable or the URL is invalid.")
                    return None
                stream_url = info.get('url') or info.get('direct_url')
                if not stream_url:
                    st.error("No valid stream URL found. The video may be private, age-restricted, or region-locked.")
                    return None
                # Validate stream with OpenCV
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    st.error("OpenCV could not open the stream. Ensure ffmpeg is installed and the video is compatible.")
                    cap.release()
                    return None
                cap.release()
            return stream_url
    except yt_dlp.utils.DownloadError as de:
        st.error(f"Download error: {str(de)}. Ensure the video is public and accessible, or try another URL.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}. Check your internet connection, ffmpeg installation, or try a different video.")
        return None

# Function to get video metadata
def get_video_metadata(video_source, is_streaming=False):
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.warning("Could not open video stream for metadata extraction.")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30 if is_streaming else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = "N/A (Streaming)" if is_streaming or total_frames <= 0 else f"{total_frames / fps:.2f} seconds"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "Duration": duration,
            "Resolution": f"{width} × {height}" if width and height else "N/A",
            "FPS": f"{fps:.2f}" if fps else "N/A",
            "Total Frames": total_frames if total_frames > 0 and not is_streaming else "N/A (Streaming)"
        }
    except Exception as e:
        st.warning(f"Error retrieving video metadata: {str(e)}")
        return None

# Main app
def main():
    st.title("🎥 LogoSense Analytics")
    st.markdown("**Professional Logo Detection and Analysis for Videos**")

    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        # Video input
        st.subheader("Video Source")
        video_source_type = st.radio("Select video source", ["Upload Video", "YouTube URL"])
        
        video_input = None
        temp_file = None
        if video_source_type == "Upload Video":
            uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
            if uploaded_video:
                temp_file = os.path.join(tempfile.gettempdir(), f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                with open(temp_file, "wb") as f:
                    f.write(uploaded_video.read())
                video_input = temp_file
        else:
            video_input = st.text_input("Enter YouTube URL", help="e.g., https://www.youtube.com/watch?v=VIDEO_ID")

        # Model selection
        st.subheader("Model Selection")
        uploaded_model = st.file_uploader("Upload YOLOv8 Model (.pt)", type=["pt"])
        model_path = None
        if uploaded_model:
            with st.spinner("Loading model..."):
                model_path = os.path.join(tempfile.gettempdir(), f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.read())
        else:
            model_path = os.path.join("model", "yolov12_new_dataset.pt")

        conf_thres = st.slider(
            "Confidence Threshold",
            0.1, 1.0, 0.25, 0.05,
            help="Set the minimum confidence for logo detections."
        )
        save_output = st.radio(
            "💾 Save Processed Video?",
            ("No", "Yes"),
            help="Save the output video with logo detections."
        )
        nosave = save_output == "No"
        start_button = st.button("🚀 Start Analysis")

    # Main content
    if video_input and model_path and start_button:
        with st.spinner("Processing video..."):
            # Resolve video source
            video_source = None
            is_streaming = False
            if video_source_type == "Upload Video":
                video_source = video_input
            else:
                if not validators.url(video_input):
                    st.error("Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=example).")
                    return
                video_source = get_youtube_stream_url(video_input)
                is_streaming = True
                if not video_source:
                    return

            # Containers for UI sections
            video_container = st.container()
            charts_container = st.container()
            stats_container = st.container()
            metadata_container = st.container()

            # Get metadata before analysis
            metadata = get_video_metadata(video_source, is_streaming)

            # Metadata display (before analysis, to ensure visibility)
            with metadata_container:
                st.markdown("### 📋 Video Metadata")
                if metadata:
                    st.markdown(
                        f"""
                        <div class="metadata-card">
                            <h3>Video Metadata</h3>
                            <table>
                                <tr>
                                    <th>Duration</th>
                                    <td>{metadata['Duration']}</td>
                                </tr>
                                <tr>
                                    <th>Resolution</th>
                                    <td>{metadata['Resolution']}</td>
                                </tr>
                                <tr>
                                    <th>FPS</th>
                                    <td>{metadata['FPS']}</td>
                                </tr>
                                <tr>
                                    <th>Total Frames</th>
                                    <td>{metadata['Total Frames']}</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="metadata-card">
                            <h3>Video Metadata</h3>
                            <p style='text-align: center; color: #2c3e50;'>No metadata available</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Video output
            with video_container:
                st.markdown("### Video Output")
                video_placeholder = st.empty()
                st.markdown(
                    "<p style='text-align: center; color: #2c3e50;'>Live logo detections</p>",
                    unsafe_allow_html=True
                )

            # Charts
            with charts_container:
                st.markdown("### Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    frequency_placeholder = st.empty()
                with col2:
                    duration_placeholder = st.empty()

            # System stats
            with stats_container:
                st.markdown("### System Performance")
                cols = st.columns(3)
                fps_container = cols[0].empty()
                mem_container = cols[1].empty()
                cpu_container = cols[2].empty()

            # Run detection
            logo_stats = detect_logos(
                source=video_source,
                model_path=model_path,
                stframe=video_placeholder,
                duration_container=duration_placeholder,
                fps_container=fps_container,
                mem_container=mem_container,
                cpu_container=cpu_container,
                chart_placeholder=frequency_placeholder,
                conf_thres=conf_thres,
                nosave=nosave
            )

            if logo_stats:
                st.success("✅ Analysis completed!")
            else:
                st.warning("⚠️ No logos detected. Try lowering the confidence threshold (e.g., 0.1) or using a YOLO model trained for logos in this video.")

            # Download processed video
            if not nosave and os.path.exists("output.mp4"):
                with open("output.mp4", "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )

            # Clean up temporary files
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            if model_path and model_path != os.path.join("model", "best.pt") and os.path.exists(model_path):
                os.remove(model_path)

if __name__ == "__main__":
    # Initialize session state for logo stats
    if "logo_stats" not in st.session_state:
        st.session_state.logo_stats = {}
    main()