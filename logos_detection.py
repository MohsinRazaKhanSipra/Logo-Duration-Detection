# Project: LogoSense Analytics
# File: logos_detection.py
# Description: 
# This file contains the core logic for detecting logos in videos using a custom YOLOv8 model. 
# It processes video frames, draws bounding boxes around detected logos, tracks logo durations, 
# monitors system performance (FPS, memory, CPU), and updates real-time charts using Altair.
# The frequency chart is a horizontal bar chart based on the number of distinct appearances of a logo, 
# where an appearance is a continuous sequence of seconds in which the logo is present. 
# Consecutive seconds count as one appearance. The chart is sorted by frequency (highest to lowest) 
# with alphabetical tie-breaking, displayed top to bottom.
# The duration chart is a horizontal bar chart sorted by duration (highest to lowest) with alphabetical tie-breaking,
# displayed top to bottom. Both charts use a fixed width with scrolling supported via CSS in app.py.
# Colors for each logo class are generated dynamically using a hash-based approach to ensure consistency
# across charts and runs, regardless of the number of classes.

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
    # Hash the class name to get a consistent index
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

def plot_logo_frequency(chart_placeholder, frame_num):
    """Update the Altair horizontal bar chart for logo frequency with consistent colors per logo."""
    logo_stats = st.session_state.logo_stats
    if not logo_stats:
        chart_placeholder.warning("No logos detected yet for frequency chart.")
        return
    
    # Sort logos by frequency (descending) and alphabetically for ties
    logos = sorted(
        logo_stats.keys(),
        key=lambda x: (-logo_stats[x]["frequency"], x)
    )
    frequencies = [logo_stats[logo]["frequency"] for logo in logos]
    num_logos = len(logos)
    
    # Assign consistent colors using hash-based mapping
    if "logo_colors" not in st.session_state:
        st.session_state.logo_colors = {}
    colors = []
    for logo in logos:
        if logo not in st.session_state.logo_colors:
            st.session_state.logo_colors[logo] = get_consistent_color(logo)
        colors.append(st.session_state.logo_colors[logo])
    
    # Dynamic bar height
    bar_height = max(20, min(40, 400 / num_logos))
    chart_height = max(200, num_logos * bar_height)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Logo': logos,
        'Appearances': frequencies
    })
    
    # Create Altair chart with consistent colors
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y('Logo:N', title='', sort=None, axis=alt.Axis(
            labelFont='Roboto',
            labelFontSize=10,
            labelColor='#1f2937',
            titleColor='#1f2937',
            labelPadding=10,
            labelLimit=200
        )),
        x=alt.X('Appearances:Q', title='Appearance Count', axis=alt.Axis(grid=True, gridColor='#e5e7eb', titleColor='#1f2937')),
        color=alt.Color('Logo:N', scale=alt.Scale(
            domain=logos,
            range=colors
        ), legend=None),  # Remove legend for cleaner look
        tooltip=['Logo:N', 'Appearances:Q']
    ).properties(
        title=f"Logo Frequency",
        width=1000,
        height=chart_height
    ).configure(
        background='transparent'
    ).configure_axis(
        titleFont='Roboto',
        titleFontSize=14,
        grid=True,
        gridColor='#e5e7eb',
        gridOpacity=0.5
    ).configure_title(
        font='Roboto',
        fontSize=18,
        color='#1f2937',
        anchor='middle'
    ).configure_mark(
        opacity=0.9,
        stroke='#4b5563',
        strokeWidth=1
    ).configure_view(
        stroke=None
    ).interactive()
    
    # Display chart
    try:
        chart_placeholder.altair_chart(chart, use_container_width=True)
    except Exception as e:
        chart_placeholder.error(f"Error rendering frequency chart: {str(e)}")

def plot_logo_duration(duration_placeholder, frame_num, total_duration):
    """Update the Altair horizontal bar chart for logo durations with consistent colors per logo."""
    logo_stats = st.session_state.logo_stats
    if not logo_stats:
        duration_placeholder.warning("No logos detected yet for duration chart.")
        return
    
    # Sort logos by duration (descending) and alphabetically for ties
    logos = sorted(
        logo_stats.keys(),
        key=lambda x: (-logo_stats[x]["duration"], x)
    )
    durations = [logo_stats[logo]["duration"] for logo in logos]
    num_logos = len(logos)
    
    # Assign consistent colors using hash-based mapping
    if "logo_colors" not in st.session_state:
        st.session_state.logo_colors = {}
    colors = []
    for logo in logos:
        if logo not in st.session_state.logo_colors:
            st.session_state.logo_colors[logo] = get_consistent_color(logo)
        colors.append(st.session_state.logo_colors[logo])
    
    # Dynamic bar height
    bar_height = max(20, min(40, 400 / num_logos))
    chart_height = max(200, num_logos * bar_height)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Logo': logos,
        'Duration': durations
    })
    
    # Create Altair chart with consistent colors
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y('Logo:N', title='', sort=None, axis=alt.Axis(
            labelFont='Roboto',
            labelFontSize=10,
            labelColor='#1f2937',
            titleColor='#1f2937',
            labelPadding=10,
            labelLimit=200
        )),
        x=alt.X('Duration:Q', title='Duration (seconds)', axis=alt.Axis(grid=True, gridColor='#e5e7eb', titleColor='#1f2937')),
        color=alt.Color('Logo:N', scale=alt.Scale(
            domain=logos,
            range=colors
        ), legend=None),  # Remove legend for cleaner look
        tooltip=['Logo:N', 'Duration:Q']
    ).properties(
        title=f"Logo Durations",
        width=1000,
        height=chart_height
    ).configure(
        background='transparent'
    ).configure_axis(
        titleFont='Roboto',
        titleFontSize=14,
        grid=True,
        gridColor='#e5e7eb',
        gridOpacity=0.5
    ).configure_title(
        font='Roboto',
        fontSize=18,
        color='#1f2937',
        anchor='middle'
    ).configure_mark(
        opacity=0.9,
        stroke='#4b5563',
        strokeWidth=1
    ).configure_view(
        stroke=None
    ).interactive()
    
    # Display chart
    try:
        duration_placeholder.altair_chart(chart, use_container_width=True)
    except Exception as e:
        duration_placeholder.error(f"Error rendering duration chart: {str(e)}")

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
    logo_start_frame = defaultdict(int)
    logo_last_seen = defaultdict(lambda: -2)  # Track last second seen; -2 ensures first detection counts
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
        current_second = int(frame_num / fps)  # Floor to get the current second

        # YOLO inference
        try:
            results = model(frame, conf=conf_thres, verbose=False, device=device)
        except Exception as e:
            stframe.error(f"Error during inference: {str(e)}")
            break

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
                # Initialize logo stats
                if label not in st.session_state.logo_stats:
                    st.session_state.logo_stats[label] = {"duration": 0.0, "frames": 0, "frequency": 0}
                st.session_state.logo_stats[label]["frames"] += 1
                # Update frequency for new appearance
                if current_second > logo_last_seen[label] + 1:  # New appearance if not seen in previous second
                    st.session_state.logo_stats[label]["frequency"] += 1
                logo_last_seen[label] = current_second

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

        # Update duration chart every 10 frames
        if frame_num % 20 == 0:
            plot_logo_duration(duration_container, frame_num, total_duration)

        # Update frequency chart every 10 frames
        if frame_num % 20 == 0:
            plot_logo_frequency(chart_placeholder, frame_num)

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

    # Finalize durations
    for logo, start_frame in logo_start_frame.items():
        if start_frame > 0:
            duration = (frame_num - start_frame) / fps
            st.session_state.logo_stats[logo]["duration"] += duration

    # Final chart updates
    plot_logo_duration(duration_container, frame_num, total_duration)
    plot_logo_frequency(chart_placeholder, frame_num)

    # Clean up
    cap.release()
    if vid_writer:
        vid_writer.release()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Final GPU memory cleanup
    return st.session_state.logo_stats