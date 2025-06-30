from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import defaultdict
import psutil
import torch
import streamlit as st
import plotly.graph_objects as go
import hashlib
import os

# Large color palette for dynamic class coloring
COLOR_PALETTE = [
    '#111827', '#b91c1c', '#15803d', '#1d4ed8', '#7e22ce', '#c2410c', '#a16207',
    '#0e7490', '#be123c', '#4d7c0f', '#6d28d9', '#b45309', '#047857', '#7f1d1d',
    '#3b82f6', '#9f1239', '#166534', '#5b21b6', '#d97706', '#0ea5e9', '#991b1b',
    '#065f46', '#7c3aed', '#b45309', '#14b8a6', '#7e22ce', '#ca8a04', '#1e40af',
    '#be185d', '#15803d'
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

def plot_logo_timeline(chart_placeholder, frame_num, total_duration, fps, chart_key):
    """Update a Plotly line plot showing multiple logo appearances with duration-based legend."""
    logo_stats = st.session_state.logo_stats
    if not logo_stats or not any("appearances" in stat for stat in logo_stats.values()):
        chart_placeholder.warning("No logos detected yet for timeline chart.")
        return

    time_points = np.arange(0, total_duration + 1 / fps, 1 / fps)
    logos = sorted(logo_stats.keys(), key=lambda x: (-logo_stats[x]["frequency"], x))
    sorted_logos = sorted(logos, key=lambda x: logo_stats[x]["duration"], reverse=True)
    duration_labels = {logo: f"{logo}-{int(logo_stats[logo]['duration'])}s" for logo in sorted_logos}
    
    fig = go.Figure()
    for i, logo in enumerate(sorted_logos):
        y_level = i + 1
        appearances = logo_stats[logo].get("appearances", [])
        for j, (start, end) in enumerate(appearances):
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[y_level, y_level],
                mode='lines',
                name=duration_labels[logo],
                line=dict(color=get_consistent_color(logo), width=5),
                hovertemplate=
                    f"<b>Logo</b>: {logo}<br>" +
                    f"<b>Start</b>: {start:.2f}s<br>" +
                    f"<b>End</b>: {end:.2f}s<br>" +
                    f"<b>Duration</b>: {(end - start):.2f}s<br>" +
                    f"<b>Frequency</b>: {logo_stats[logo]['frequency']}",
                showlegend=(j == 0)
            ))

    fig.update_layout(
        title=dict(text="", font=dict(size=16, color="#1f2937")),
        xaxis_title="Time (seconds)",
        yaxis_title="Logos",
        xaxis=dict(
            range=[0, total_duration],
            tickmode='linear',
            dtick=total_duration / 10,
            gridcolor='#e5e7eb',
            gridwidth=0.5,
            zeroline=False
        ),
        yaxis=dict(
            tickvals=list(range(1, len(sorted_logos) + 1)),
            ticktext=sorted_logos,
            range=[0, len(sorted_logos) + 1]
        ),
        height=400,
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='#1f2937'),
        showlegend=True,
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb",
            borderwidth=1
        ),
        margin=dict(l=100, r=20, t=50, b=20)
    )

    try:
        chart_placeholder.plotly_chart(fig, key=chart_key, use_container_width=True)
    except Exception as e:
        chart_placeholder.error(f"Error rendering timeline chart: {str(e)}")

def detect_logos(source, model_path, stframe=None, fps_container=None, mem_container=None, cpu_container=None,
                 chart_placeholder=None, conf_thres=0.5, nosave=True, output_video_path=None, progress_bar=None, progress_text=None, chart_key=None):
    if not os.path.exists(model_path):
        stframe.error(f"Error: Model file {model_path} not found.")
        return {}

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            stframe.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
            stframe.warning("No GPU detected. Using CPU.")
    except Exception as e:
        device = torch.device("cpu")
        stframe.error(f"Device init error: {str(e)}")

    try:
        model = YOLO(model_path).to(device)
    except Exception as e:
        stframe.error(f"Failed to load YOLO model: {str(e)}")
        return {}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        stframe.error("Could not open video source.")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000  # Fallback for streaming
    total_duration = total_frames / fps if total_frames > 0 else 1.0

    if "logo_stats" not in st.session_state:
        st.session_state.logo_stats = defaultdict(lambda: {"duration": 0.0, "frames": 0, "frequency": 0, "appearances": []})

    logo_start_frame = defaultdict(lambda: None)
    frame_num = 0
    prev_time = time.time()

    video_writer = None
    if not nosave and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_second = frame_num / fps
        try:
            results = model(frame, conf=conf_thres, verbose=False, device=device)
        except Exception as e:
            stframe.error(f"Inference error: {str(e)}")
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

        for logo in st.session_state.logo_stats.keys():
            if logo in detected_logos:
                if logo_start_frame[logo] is None:
                    logo_start_frame[logo] = frame_num
            elif logo_start_frame[logo] is not None:
                start_time = logo_start_frame[logo] / fps
                end_time = frame_num / fps
                duration = end_time - start_time
                if duration > 0:
                    st.session_state.logo_stats[logo]["appearances"].append((start_time, end_time))
                    st.session_state.logo_stats[logo]["duration"] += duration
                    st.session_state.logo_stats[logo]["frequency"] = len(st.session_state.logo_stats[logo]["appearances"])
                logo_start_frame[logo] = None

        if video_writer:
            video_writer.write(frame)

        curr_time = time.time()
        fps_val = round(1 / (curr_time - prev_time), 1) if curr_time != prev_time else 0
        prev_time = curr_time
        mem_val = psutil.virtual_memory().percent
        cpu_val = psutil.cpu_percent()

        fps_container.markdown(f"<div class='stat-card'><h3>FPS</h3><p>{fps_val} FPS</p></div>", unsafe_allow_html=True)
        mem_container.markdown(f"<div class='stat-card'><h3>Memory</h3><p>{mem_val}%</p></div>", unsafe_allow_html=True)
        cpu_container.markdown(f"<div class='stat-card'><h3>CPU</h3><p>{cpu_val}%</p></div>", unsafe_allow_html=True)

        # Update progress
        progress = min(frame_num / total_frames, 1.0) if total_frames > 0 else 0
        progress_bar.progress(progress)
        progress_text.markdown(f"<p class='text-sm text-gray-600'>Processed {frame_num}/{total_frames} frames ({progress*100:.1f}%)</p>", unsafe_allow_html=True)

        if frame_num % 20 == 0:
            plot_logo_timeline(chart_placeholder, frame_num, total_duration, fps, chart_key)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=820)
        frame_num += 1

        if device.type == "cuda" and frame_num % 100 == 0:
            torch.cuda.empty_cache()

    for logo, start_frame in logo_start_frame.items():
        if start_frame is not None:
            start_time = start_frame / fps
            end_time = frame_num / fps
            duration = end_time - start_time
            st.session_state.logo_stats[logo]["appearances"].append((start_time, end_time))
            st.session_state.logo_stats[logo]["duration"] += duration
            st.session_state.logo_stats[logo]["frequency"] = len(st.session_state.logo_stats[logo]["appearances"])


    progress = 1
    progress_bar.progress(progress)
    progress_text.markdown(f"<p class='text-sm text-gray-600'>Processed {total_frames}/{total_frames} frames ({progress*100:.1f}%)</p>", unsafe_allow_html=True)
    plot_logo_timeline(chart_placeholder, frame_num, total_duration, fps, chart_key)
    cap.release()
    if video_writer:
        video_writer.release()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return st.session_state.logo_stats