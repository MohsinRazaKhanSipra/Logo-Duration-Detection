# Project: LogoSense Analytics
# File: logos_detection.py
# Description: 
# This file contains the core logic for detecting logos in videos using a custom YOLOv8 model. 
# It processes video frames, draws bounding boxes around detected logos, tracks logo durations, 
# monitors system performance (FPS, memory, CPU), and updates real-time charts using Plotly.
# The chart plots video timestamps along the x-axis (divided equally based on video length) and 
# logo appearances on the y-axis, with multiple bars per logo for different start/end times.
# Tooltips show start time, end time, duration, and frequency. Colors are generated dynamically.

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import defaultdict
import psutil
import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import hashlib

# Large color palette for dynamic class coloring
COLOR_PALETTE = [
    '#111827', '#b91c1c', '#15803d', '#1d4ed8', '#7e22ce', '#c2410c', '#a16207',
    '#0e7490', '#be123c', '#4d7c0f', '#6d28d9', '#b45309', '#047857', '#7f1d1d',
    '#3b82f6', '#9f1239', '#166534', '#5b21b6', '#d97706', '#0ea5e9', '#991b1b',
    '#065f46', '#7c3aed', '#b45309', '#14b8a6', '#7e22ce', '#ca8a04', '#1e40af',
    '#be185d', '#15803d'  # Add more colors if expecting many classes
]

def get_consistent_color(class_name):
    """Generate a consistent color for a class name using hashing."""
    hash_object = hashlib.md5(class_name.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    color_index = hash_int % len(COLOR_PALETTE)
    return COLOR_PALETTE[color_index]

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

def plot_combined_logo_analysis(chart_placeholder, frame_num, total_duration):
    """Update a Plotly horizontal bar chart showing multiple logo appearances with hover tooltips."""
    logo_stats = st.session_state.logo_stats
    if not logo_stats or not any("appearances" in stat for stat in logo_stats.values()):
        chart_placeholder.warning("No logos detected yet for analysis chart.")
        return
    
    # Prepare data for plotting
    data = []
    logos = sorted(logo_stats.keys(), key=lambda x: (-logo_stats[x]["frequency"], x))
    for logo in logos:
        stat = logo_stats[logo]
        total_logo_duration = stat["duration"]
        frequency = stat["frequency"]
        for start, end in stat.get("appearances", []):
            duration = end - start
            data.append({
                "Logo": f"{logo} ({total_logo_duration:.2f}s)",  # Updated label
                "Start": start,
                "End": end,
                "Duration": duration,
                "Frequency": frequency
            })
    
    if not data:
        chart_placeholder.warning("No logo appearances to display.")
        return
    
    # Create Plotly figure
    fig = go.Figure()
    
    for i, item in enumerate(data):
        fig.add_trace(go.Bar(
            y=[item["Logo"]],
            x=[item["End"] - item["Start"]],
            base=[item["Start"]],
            orientation='h',
            marker_color=get_consistent_color(item["Logo"].split(" (")[0]),  # Use base logo name for color
            text=[f"{item['Duration']:.2f}s"],
            textposition='outside',
            hovertemplate=
            '<b>Logo</b>: %{y}<br>'+
            '<b>Start</b>: %{base:.2f}s<br>'+
            '<b>End</b>: %{x:.2f}s<br>'+
            '<b>Duration</b>: %{customdata[0]:.2f}s<br>'+
            '<b>Frequency</b>: %{customdata[1]}<br>',
            customdata=[[item["Duration"]], [item["Frequency"]]],
            showlegend=False
        ))
    
    # Update layout with x-axis based on video duration
    fig.update_layout(
        title='Logo Analysis Results',
        xaxis_title='Video Time (seconds)',
        yaxis_title='',
        xaxis=dict(
            range=[0, total_duration],
            tickmode='linear',
            dtick=total_duration / 10,
            gridcolor='#e5e7eb',
            gridwidth=0.5,
            zeroline=False
        ),
        yaxis=dict(autorange="reversed"),
        bargap=0.2,
        height=max(200, len(data) * 40),
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12, color='#1f2937'),
        title_font=dict(size=18, color='#1f2937')
    )
    
    # Display chart
    try:
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        chart_placeholder.error(f"Error rendering combined chart: {str(e)}")


def detect_logos(source, model_path, stframe=None, duration_container=None,
                 fps_container=None, mem_container=None, cpu_container=None,
                 chart_placeholder=None, conf_thres=0.25, nosave=True):
    # Validate model input
    if not model_path:
        stframe.error("Error: No model uploaded. Please upload a custom YOLOv8 model (.pt).")
        return {}

    # Initialize device with enhanced GPU/CPU handling
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            stframe.info(f"Using GPU: {gpu_name}")
            torch.cuda.empty_cache()  # Clear GPU memory before loading model
        else:
            device = torch.device("cpu")
            stframe.warning("No GPU detected. Falling back to CPU.")
    except Exception as e:
        device = torch.device("cpu")
        stframe.error(f"Error initializing device: {str(e)}. Falling back to CPU.")

    # Load model
    try:
        model = YOLO(model_path).to(device)
    except Exception as e:
        stframe.error(f"Error: Failed to load YOLO model {model_path}: {str(e)}")
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
    if "logo_stats" not in st.session_state:
        st.session_state.logo_stats = defaultdict(lambda: {"duration": 0.0, "frames": 0, "frequency": 0, "appearances": []})
    logo_start_frame = defaultdict(int)
    logo_last_seen = defaultdict(lambda: -2)  # Track last second seen
    frame_num = 0
    prev_time = time.time()

    # Placeholders for UI updates
    fps_placeholder = fps_container.empty()
    mem_placeholder = mem_container.empty()
    cpu_placeholder = cpu_container.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current second
        current_second = frame_num / fps

        # YOLO inference
        try:
            results = model(frame, conf=conf_thres, verbose=False, device=device)
        except Exception as e:
            stframe.error(f"Error during inference: {str(e)}")
            break

        detected_logos = set()
        bboxes = []
        labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = box.conf.item()
                detected_logos.add(label)
                if label not in st.session_state.logo_stats:
                    st.session_state.logo_stats[label] = {"duration": 0.0, "frames": 0, "frequency": 0, "appearances": []}
                st.session_state.logo_stats[label]["frames"] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
                labels.append(f"{label} {conf:.2f}")

            frame = draw_boxes(frame, bboxes, labels)

        # Update appearances and durations
        for logo in st.session_state.logo_stats.keys():
            if logo in detected_logos:
                if logo_start_frame[logo] == 0:
                    logo_start_frame[logo] = frame_num
            elif logo_start_frame[logo] > 0:
                start_frame = logo_start_frame[logo]
                end_frame = frame_num
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                st.session_state.logo_stats[logo]["appearances"].append((start_time, end_time))
                st.session_state.logo_stats[logo]["duration"] += duration
                st.session_state.logo_stats[logo]["frequency"] = len(st.session_state.logo_stats[logo]["appearances"])
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

        # Update system stats with card styling
        fps_placeholder.markdown(
            f'<div class="stat-card"><h3>FPS</h3><p>{fps_val} FPS</p></div>',
            unsafe_allow_html=True
        )
        mem_placeholder.markdown(
            f'<div class="stat-card"><h3>Memory</h3><p>{mem_val}%</p></div>',
            unsafe_allow_html=True
        )
        cpu_placeholder.markdown(
            f'<div class="stat-card"><h3>CPU</h3><p>{cpu_val}%</p></div>',
            unsafe_allow_html=True
        )

        # Update combined chart every 20 frames
        if frame_num % 20 == 0:
            plot_combined_logo_analysis(chart_placeholder, frame_num, total_duration)

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, width=820)

        # Save frame
        if vid_writer:
            vid_writer.write(frame)

        frame_num += 1

        # Clear GPU memory periodically to prevent memory leaks
        if device.type == "cuda" and frame_num % 100 == 0:
            torch.cuda.empty_cache()

    # Finalize durations for any ongoing appearances
    for logo, start_frame in logo_start_frame.items():
        if start_frame > 0:
            end_frame = frame_num
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            st.session_state.logo_stats[logo]["appearances"].append((start_time, end_time))
            st.session_state.logo_stats[logo]["duration"] += duration
            st.session_state.logo_stats[logo]["frequency"] = len(st.session_state.logo_stats[logo]["appearances"])

    # Final chart update
    plot_combined_logo_analysis(chart_placeholder, frame_num, total_duration)

    # Clean up
    cap.release()
    if vid_writer:
        vid_writer.release()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Final GPU memory cleanup
    return st.session_state.logo_stats