# Project: LogoSense Analytics
# File: logos_detection.py
# Description: 
# This file contains the core logic for detecting logos in videos using the YOLOv8 model from Ultralytics. 
# It processes video frames, draws bounding boxes around detected logos, tracks logo durations, 
# monitors system performance (FPS, memory, CPU), and updates a single real-time logo frequency chart using Altair. 
# The processed frames and analytics are displayed in real-time via Streamlit, 
# and the output video can optionally be saved.

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import defaultdict
import psutil
import torch
import streamlit as st
import pandas as pd
import altair as alt

def draw_boxes(img, bboxes, labels):
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (107, 114, 128), 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            img,
            (x1, y1 - label_size[1] - 12),
            (x1 + label_size[0] + 12, y1),
            (107, 114, 128),
            -1
        )
        cv2.putText(
            img,
            label,
            (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    return img

def plot_logo_frequency(chart_placeholder, frame_num):
    """Update the Altair bar chart for logo frequency with professional styling."""
    logo_stats = st.session_state.logo_stats
    if not logo_stats:
        chart_placeholder.warning("No logos detected yet.")
        return
    
    logos = sorted(logo_stats.keys())
    frequencies = [logo_stats[logo]["frames"] for logo in logos]
    num_logos = len(logos)
    
    # Debug number of logos
    #st.write(f"Debug: Number of logos: {num_logos}")
    
    # Dynamic bar width: smaller for fewer logos, scaling up as more are detected
    if num_logos <= 3:
        bar_width = 0.05  # Narrower for 1-3 logos (25px)
    else:
        bar_width = min(0.6 / num_logos, 0.15)  # Scale inversely, max 0.15 (25-75px)
    
    # Create DataFrame for chart
    data = pd.DataFrame({
        'Logo': logos,
        'Frames': frequencies,
        'BarWidth': [bar_width * 500] * num_logos  # Scaling yields 25-75px
    })
    
    # Dynamic chart width: ~80px per logo, capped at 1200px
    chart_width = min(1200, 80 * num_logos)
    
    # Debug bar width and chart width
    #st.write(f"Debug: Frame {frame_num} - BarWidth: {data['BarWidth'].iloc[0]}px, BarWidthRaw: {bar_width}, ChartWidth: {chart_width}px")
    
    # Create Altair chart
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Logo:N', title='Logos', sort=None, axis=alt.Axis(
            labelAngle=45,
            labelFont='Arial',
            labelFontSize=10,
            labelColor='#1f2937',
            titleColor='#1f2937',
            labelOverlap=False,  # Force all labels to display
            labelPadding=10,
            labelLimit=200  # Support longer labels
        )),
        y=alt.Y('Frames:Q', title='Frame Count', axis=alt.Axis(grid=True, gridColor='#e5e7eb', titleColor='#1f2937')),
        size=alt.Size('BarWidth:Q', scale=None, legend=None),
        color=alt.condition(
            alt.datum._hover,
            alt.ColorValue('#6b7280'),  # Lighten on hover
            alt.ColorValue('#4b5563')   # Default blue-gray
        ),
        tooltip=['Logo:N', 'Frames:Q']
    ).properties(
        title=f"Logo Frequency (Frame {frame_num})",
        width=chart_width,  # Dynamic width
        height=400
    ).configure(
        background='transparent'
    ).configure_axis(
        titleFont='Arial',
        titleFontSize=14,
        grid=True,
        gridColor='#e5e7eb',
        gridOpacity=0.5
    ).configure_title(
        font='Arial',
        fontSize=18,
        color='#1f2937',
        anchor='middle'
    ).configure_mark(
        opacity=0.9,
        stroke='#4b5563',
        strokeWidth=1
    ).configure_view(
        stroke=None
    ).interactive()  # Enable hover effects
    
    # Display chart in placeholder
    try:
        chart_placeholder.altair_chart(chart)
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")

def detect_logos(source, model_path, stframe, duration_container, fps_container, mem_container, cpu_container,
                 chart_placeholder, conf_thres=0.25, nosave=True):
    # Force CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        stframe.warning("GPU not detected. Running on CPU.")

    # Load model
    try:
        model = YOLO(model_path).to(device)
       # st.write(f"Debug: Model loaded. Classes: {model.names}")
    except Exception as e:
        stframe.error(f"Error: Failed to load YOLO model: {str(e)}")
        return {}

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        stframe.error("Error: Could not open video.")
        return {}

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if total_frames > 0 else 1.0
    save_path = "output.mp4" if not nosave else None
    vid_writer = None

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Track logo stats in session state
    logo_start_frame = defaultdict(int)
    frame_num = 0
    prev_time = time.time()

    # Placeholders for UI updates
    logo_placeholders = {}
    fps_placeholder = fps_container.empty()
    fps_bar = fps_container.progress(0)
    mem_placeholder = mem_container.empty()
    mem_bar = mem_container.progress(0)
    cpu_placeholder = cpu_container.empty()
    cpu_bar = cpu_container.progress(0)

    with duration_container:
        st.markdown("**Detected Logos**")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, conf=conf_thres, verbose=False)
        detected_logos = set()

        # Process detections
        bboxes = []
        labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = box.conf.item()
                detected_logos.add(label)
                st.session_state.logo_stats.setdefault(label, {"duration": 0.0, "frames": 0})["frames"] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
                labels.append(f"{label} {conf:.2f}")

            frame = draw_boxes(frame, bboxes, labels)

        # Update durations
        current_frame_time = frame_num / fps
        for logo in st.session_state.logo_stats.keys():
            if logo in detected_logos:
                if logo_start_frame[logo] == 0:
                    logo_start_frame[logo] = frame_num
            elif logo_start_frame[logo] > 0:
                duration = (frame_num - logo_start_frame[logo]) / fps
                st.session_state.logo_stats[logo]["duration"] += duration
                logo_start_frame[logo] = 0

        for logo in detected_logos:
            if logo not in st.session_state.logo_stats:
                logo_start_frame[logo] = frame_num

        # Real-time analytics
        curr_time = time.time()
        fps_val = round(1 / (curr_time - prev_time), 1) if curr_time != prev_time else 0
        prev_time = curr_time
        mem_val = psutil.virtual_memory().percent
        cpu_val = psutil.cpu_percent()

        # Update system stats
        fps_placeholder.markdown(f"<div class='stat-box'><b>FPS</b><br><span class='stat-value'>{fps_val} FPS</span></div>", unsafe_allow_html=True)
        fps_bar.progress(min(int(fps_val / 30 * 100), 100))
        mem_placeholder.markdown(f"<div class='stat-box'><b>Memory</b><br><span class='stat-value'>{mem_val}%</span></div>", unsafe_allow_html=True)
        mem_bar.progress(int(mem_val))
        cpu_placeholder.markdown(f"<div class='stat-box'><b>CPU</b><br><span class='stat-value'>{cpu_val}%</span></div>", unsafe_allow_html=True)
        cpu_bar.progress(int(cpu_val))

        # Update logo durations with progress bars
        with duration_container:
            for logo in sorted(st.session_state.logo_stats.keys()):
                duration = st.session_state.logo_stats[logo]["duration"]
                if logo not in logo_placeholders:
                    placeholder = st.empty()
                    bar = st.progress(0)
                    logo_placeholders[logo] = {"text": placeholder, "bar": bar}
                logo_placeholders[logo]["text"].markdown(f"**{logo}**: {duration:.2f} sec")
                logo_placeholders[logo]["bar"].progress(min(int((duration / total_duration) * 100), 100))

        # Update frequency chart every 20 frames
        if frame_num % 20 == 0:
            plot_logo_frequency(chart_placeholder, frame_num)

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, width=640)

        # Save frame
        if vid_writer:
            vid_writer.write(frame)

        frame_num += 1

    # Finalize durations
    for logo, start_frame in logo_start_frame.items():
        if start_frame > 0:
            duration = (frame_num - start_frame) / fps
            st.session_state.logo_stats[logo]["duration"] += duration

    # Update final durations
    with duration_container:
        for logo in sorted(st.session_state.logo_stats.keys()):
            duration = st.session_state.logo_stats[logo]["duration"]
            logo_placeholders[logo]["text"].markdown(f"**{logo}**: {duration:.2f} sec")
            logo_placeholders[logo]["bar"].progress(min(int((duration / total_duration) * 100), 100))

    # Final frequency chart update
    plot_logo_frequency(chart_placeholder, frame_num)

    # Clean up
    cap.release()
    if vid_writer:
        vid_writer.release()
    return st.session_state.logo_stats
