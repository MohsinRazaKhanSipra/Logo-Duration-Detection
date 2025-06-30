import streamlit as st
import yt_dlp
import os
import cv2
import tempfile
import validators
from datetime import datetime
from logos_detection import detect_logos
import pandas as pd
from io import BytesIO
import json
import uuid

# Custom CSS with Tailwind for professional styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            padding: 0;
            margin: 0;
        }
        .header {
            background-color: #1e3a8a;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .header img {
            height: 2.5rem;
        }
        .header h1 {
            color: white;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.375rem;
            padding: 0.75rem;
            font-weight: 600;
            transition: background-color 0.2s;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
        }
        .delete-button {
            background-color: #dc2626;
        }
        .delete-button:hover {
            background-color: #b91c1c;
        }
        .stSelectbox > div > div {
            background-color: #ffffff;
            border-radius: 0.375rem;
            border: 1px solid #d1d5db;
        }
        .stSlider > div > div > div > div {
            background-color: #2563eb;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border-radius: 0.375rem;
            border: 1px solid #d1d5db;
        }
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .stat-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-card h3 {
            margin: 0;
            font-size: 0.875rem;
            color: #1f2937;
        }
        .stat-card p {
            margin: 0.5rem 0 0;
            font-size: 1.25rem;
            font-weight: 600;
            color: #2563eb;
        }
        .metadata-card table {
            width: 100%;
            border-collapse: collapse;
        }
        .metadata-card th, .metadata-card td {
            padding: 0.75rem;
            text-align: left;
            font-size: 0.875rem;
            color: #1f2937;
        }
        .metadata-card th {
            font-weight: 600;
            width: 40%;
        }
        .metadata-card td {
            color: #2563eb;
        }
        .metadata-card tr:not(:last-child) {
            border-bottom: 1px solid #e5e7eb;
        }
        .stDownloadButton>button {
            background-color: #2563eb;
            color: white;
            padding: 1rem;
            border-radius: 0.375rem;
            font-weight: 600;
        }
        .stDownloadButton>button:hover {
            background-color: #1d4ed8;
        }
        .stProgress > div > div {
            background-color: #2563eb;
        }
    </style>
""", unsafe_allow_html=True)

# Ensure processed_videos directory exists
if not os.path.exists("processed_videos"):
    os.makedirs("processed_videos")

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

# Function to save analysis record
def save_analysis_record(video_source, metadata, logo_stats, output_video_path=None):
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_source": video_source,
        "metadata": metadata,
        "logo_stats": logo_stats,
        "processed_video_path": output_video_path
    }
    try:
        if os.path.exists("analysis_records.json"):
            with open("analysis_records.json", "r") as f:
                records = json.load(f)
        else:
            records = []
        records.append(record)
        with open("analysis_records.json", "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        st.error(f"Error saving analysis record: {str(e)}")

# Function to load records
def load_records():
    if os.path.exists("analysis_records.json"):
        with open("analysis_records.json", "r") as f:
            return json.load(f)
    return []

# Function to save records
def save_records(records):
    with open("analysis_records.json", "w") as f:
        json.dump(records, f, indent=2)

# Main app
def main():
    # Header
    st.markdown("""
<div class="header">
    <img src="https://image.made-in-china.com/202f0j00fditTurWBRzG/Night-Vision-300m-Smart-Auto-Tracking-and-Analysis-IP-PTZ-Camera.webp" alt="Logo">
    <div>
        <h1 class="text-2xl font-bold">LogoSense Analytics</h1>
        <p class="text-sm text-gray-100">AI-Powered Logo Detection and Video Intelligence</p>
    </div>
</div>
""", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["Logo Detection", "Past Records"])

    # Tab 1: Logo Detection (from original app.py)
    with tab1:
        # Sidebar for inputs
        with st.sidebar:
            st.markdown('<h3 class="text-sm font-medium text-dark">Video Source</h3>', unsafe_allow_html=True)
            video_source_type = st.selectbox(
                "Select video source",
                ["Upload Video", "YouTube URL"],
                help="Choose to upload a local video or stream from a YouTube URL",
                key="video_source_type"
            )
            
            video_input = None
            temp_file = None
            if video_source_type == "Upload Video":
                uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], help="Supported formats: MP4, AVI, MOV")
                if uploaded_video:
                    temp_file = os.path.join(tempfile.gettempdir(), f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_video.read())
                    video_input = temp_file
            else:
                video_input = st.text_input("Enter YouTube URL", help="e.g., https://www.youtube.com/watch?v=VIDEO_ID")

            model_path = r"model\weights.pt"
            st.markdown('<h3 class="text-sm font-medium text-dark">Detection Settings</h3>', unsafe_allow_html=True)
            conf_thres = st.slider(
                "Confidence Threshold",
                0.1, 1.0, 0.5, 0.05,
                help="Set the minimum confidence for logo detections (0.1 to 1.0)"
            )
            save_output = st.selectbox(
                "Save Processed Video?",
                ("No", "Yes"),
                help="Choose whether to save the output video with logo detections"
            )
            nosave = save_output == "No"
            start_button = st.button("🚀 Start Analysis", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not video_input and not start_button:
            st.markdown('<div class="card"><h2 class="text-lg font-semibold text-gray-800">Upload video for analysis</h2></div>', unsafe_allow_html=True)
                    
        # Main content for logo detection
        if video_input and model_path and start_button:
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

                # Layout containers
                video_container = st.container()
                progress_container = st.container()
                stats_container = st.container()
                chart_container = st.container()
                metadata_container = st.container()
                download_container = st.container()

                # Progress bar
                with progress_container:
                    st.markdown('<h2 class="text-lg font-semibold text-gray-800">Processing Progress</h2>', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
              

                # Get metadata before analysis
                metadata = get_video_metadata(video_source, is_streaming)

                # Video output
                with video_container:
                    st.markdown('<h2 class="text-lg font-semibold text-gray-800">Video Output</h2>', unsafe_allow_html=True)
                    video_placeholder = st.empty()
                    st.markdown('<p class="text-sm text-gray-600 text-center">Live logo detections</p>', unsafe_allow_html=True)

                # System stats
                with stats_container:
                    st.markdown('<h2 class="text-lg font-semibold text-gray-800">System Performance</h2>', unsafe_allow_html=True)
                    cols = st.columns(3)
                    fps_container = cols[0].empty()
                    mem_container = cols[1].empty()
                    cpu_container = cols[2].empty()
                

                # Charts
                with chart_container:
                    st.markdown('<h2 class="text-lg font-semibold text-gray-800">Logo Presence Timeline</h2>', unsafe_allow_html=True)
                    frequency_placeholder = st.empty()
                 

                # Metadata display
                with metadata_container:
                    st.markdown('<h2 class="text-lg font-semibold text-gray-800">Video Metadata</h2>', unsafe_allow_html=True)
                    if metadata:
                        st.markdown(
                            f"""
                            <div class="metadata-card">
                                <table>
                                    <tr><th>Duration</th><td>{metadata['Duration']}</td></tr>
                                    <tr><th>Resolution</th><td>{metadata['Resolution']}</td></tr>
                                    <tr><th>FPS</th><td>{metadata['FPS']}</td></tr>
                                    <tr><th>Total Frames</th><td>{metadata['Total Frames']}</td></tr>
                                </table>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="metadata-card"><p class="text-center text-gray-600">No metadata available</p></div>',
                            unsafe_allow_html=True
                        )
                

                # Run detection
                output_video_path = f"processed_videos/processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4" if not nosave else None
                logo_stats = detect_logos(
                    source=video_source,
                    model_path=model_path,
                    stframe=video_placeholder,
                    fps_container=fps_container,
                    mem_container=mem_container,
                    cpu_container=cpu_container,
                    chart_placeholder=frequency_placeholder,
                    conf_thres=conf_thres,
                    nosave=nosave,
                    output_video_path=output_video_path,
                    progress_bar=progress_bar,
                    progress_text=progress_text
                )

                # Download buttons
                with download_container:
                 
                    if logo_stats:
                        st.success("✅ Analysis completed successfully!")
                        # Save analysis record
                        save_analysis_record(video_source, metadata, logo_stats, output_video_path)

                        # Create Excel file in memory
                        rows = []
                        for logo, data in logo_stats.items():
                            for start, end in data['appearances']:
                                rows.append({
                                    "Logo": logo,
                                    "Start Time (s)": start,
                                    "End Time (s)": end,
                                    "Duration (s)": round(end - start, 2),
                                    "Total Duration (s)": round(data['duration'], 2),
                                    "Frequency": data['frequency']
                                })
                        df = pd.DataFrame(rows)
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Logo Analysis')
                        output.seek(0)

                        # Download button for Excel
                        st.download_button(
                            label="📊 Download Logo Analysis (Excel)",
                            data=output,
                            file_name=f"logo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Download button for processed video
                        if not nosave and os.path.exists(output_video_path):
                            with open(output_video_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=f,
                                    file_name=os.path.basename(output_video_path),
                                    mime="video/mp4"
                                )
                    else:
                        st.warning("⚠️ No logos detected. Try lowering the confidence threshold or check the video content.")
                

                # Clean up temporary files
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)

    # Tab 2: Past Records (from records.py)
    with tab2:
  
        st.markdown('<h2 class="text-lg font-semibold text-gray-800">Past Analysis Records</h2>', unsafe_allow_html=True)

        records = load_records()
        if not records:
            st.info("No analysis records found.")
            return

        # Display records as a selectable list
        record_options = [f"{r['timestamp']} - {r['video_source']}" for r in records]
        selected_record = st.selectbox("Select a record to view details", record_options, key="record_select")

        if selected_record:
            record_index = record_options.index(selected_record)
            record = records[record_index]

            # Display metadata
            st.markdown('<h3 class="text-md font-semibold text-gray-800">Video Metadata</h3>', unsafe_allow_html=True)
            metadata = record.get('metadata', {})
            if metadata:
                st.markdown(
                    f"""
                    <div class="card metadata-card">
                        <table>
                            <tr><th>Duration</th><td>{metadata.get('Duration', 'N/A')}</td></tr>
                            <tr><th>Resolution</th><td>{metadata.get('Resolution', 'N/A')}</td></tr>
                            <tr><th>FPS</th><td>{metadata.get('FPS', 'N/A')}</td></tr>
                            <tr><th>Total Frames</th><td>{metadata.get('Total Frames', 'N/A')}</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="card metadata-card"><p class="text-center text-gray-600">No metadata available</p></div>', unsafe_allow_html=True)

            # Display logo analysis
            st.markdown('<h3 class="text-md font-semibold text-gray-800">Logo Analysis</h3>', unsafe_allow_html=True)
            logo_stats = record.get('logo_stats', {})
            if logo_stats:
                rows = []
                for logo, data in logo_stats.items():
                    for start, end in data.get('appearances', []):
                        rows.append({
                            "Logo": logo,
                            "Start Time (s)": start,
                            "End Time (s)": end,
                            "Duration (s)": round(end - start, 2),
                            "Total Duration (s)": round(data.get('duration', 0), 2),
                            "Frequency": data.get('frequency', 0)
                        })
                df = pd.DataFrame(rows)
                st.dataframe(df)

                # Download Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Logo Analysis')
                output.seek(0)
                st.download_button(
                    label="📊 Download Logo Analysis (Excel)",
                    data=output,
                    file_name=f"logo_analysis_{record['timestamp'].replace(' ', '_').replace(':', '-')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No logo analysis data available.")

            # Download processed video
            processed_video_path = record.get('processed_video_path')
            if processed_video_path and os.path.exists(processed_video_path):
                with open(processed_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name=os.path.basename(processed_video_path),
                        mime="video/mp4"
                    )
            else:
                st.info("No processed video available for this record.")

            # Delete record
            if st.button("🗑️ Delete Record", key=f"delete_record_{record_index}", help="Permanently delete this analysis record"):
                if processed_video_path and os.path.exists(processed_video_path):
                    try:
                        os.remove(processed_video_path)
                    except Exception as e:
                        st.error(f"Error deleting video file: {str(e)}")
                records.pop(record_index)
                save_records(records)
                st.success("Record deleted successfully!")
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()