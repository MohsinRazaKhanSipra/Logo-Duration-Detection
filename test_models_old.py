import os
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import time
import torch
from logos_detection import detect_logos  # Assumes logos_detection.py is in the same directory

# Ground truth durations (in seconds) for each logo
GROUND_TRUTH_DURATIONS = {
    'Abha': 0,
    'Aldiyar arabia': 0,
    'Alhilal': 0,
    'Almajed oud': 0,
    'Altazij': 0,
    'BSF': 0,
    'Bank Aljazira': 0,
    'Basic': 0,
    'Batuni': 0,
    'Berain': 0,
    'Cheek Rib': 0,
    'Direct': 0,
    'Dojan': 6.5,
    'FIFA': 0,
    'Flow': 0,
    'Flynas': 0,
    'Gree': 0,
    'Jahez': 0,
    'Jeddah': 0,
    'Kyan': 0,
    'Lateen': 1.5,
    'Like card': 0,
    'Mazda': 0,
    'Medgulf': 0,
    'Medplus': 0,
    'Pepsi': 0,
    'Qaid': 10.7,
    'Reef': 0,
    'Riyadh bank': 0,
    'Roshn Saudi League': 13,
    'SBL': 0,
    'SPL': 0,
    'STC': 0,
    'Safwa aljwar': 0,
    'Saudia': 0,
    'Savy': 2.66,
    'Tadavi': 0,
    'Tamini': 13.8,
    'Tree': 0.5,
    'United pharm': 0,
    'V2M': 0,
    'Yadh': 0,
    'osus': 0,
    'rsl': 0
}

# List of models to test
MODELS = {
    'yolov8s': r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\model\yolov8s_old_dataset.pt',
    'yolov8m': r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\model\yolov8m_new_dataset.pt',
    'yolov8l': r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\model\yolov8l_new.pt',
    'yolov8x': r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\model\yolov8x_new.pt',
    'roboflow_model': r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\model\weights.pt'
}

# Video file for testing
VIDEO_PATH = r'C:\Users\shop with hope\Desktop\Development\Computer Vision\Sir Sultan\Logo-Duration-Detection\test sample\Small Video.mp4'

def modified_detect_logos(source, model_path, conf_thres=0.25):
    """Modified detect_logos function to work without Streamlit dependencies."""
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load model
    try:
        model = YOLO(model_path).to(device)
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return {}

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video {source}")
        return {}

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.5
    frame_num = 0
    logo_start_frame = defaultdict(int)
    logo_last_seen = defaultdict(lambda: -2)
    logo_stats = defaultdict(lambda: {"duration": 0.0, "frames": 0, "frequency": 0})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_second = int(frame_num / fps)
        try:
            results = model(frame, conf=conf_thres, verbose=False, device=device)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            break

        detected_logos = set()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                detected_logos.add(label)
                logo_stats[label]["frames"] += 1
                if current_second > logo_last_seen[label] + 1:
                    logo_stats[label]["frequency"] += 1
                logo_last_seen[label] = current_second

        for logo in logo_stats.keys():
            if logo in detected_logos:
                if logo_start_frame[logo] == 0:
                    logo_start_frame[logo] = frame_num
            elif logo_start_frame[logo] > 0:
                duration = (frame_num - logo_start_frame[logo]) / fps
                logo_stats[logo]["duration"] += duration
                logo_start_frame[logo] = 0

        for logo in detected_logos:
            if logo not in logo_stats:
                logo_start_frame[logo] = frame_num

        frame_num += 1

    # Finalize durations
    for logo, start_frame in logo_start_frame.items():
        if start_frame > 0:
            duration = (frame_num - start_frame) / fps
            logo_stats[logo]["duration"] += duration

    cap.release()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logo_stats

def test_models():
    """Test each YOLOv8 model and export logo durations with logos as rows."""
    # Verify video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file {VIDEO_PATH} not found.")
        return

    # Initialize results dictionary: logo -> {model: duration}
    results_dict = {logo: {'Actual Duration (s)': duration} for logo, duration in GROUND_TRUTH_DURATIONS.items()}
    for model_name in MODELS:
        for logo in GROUND_TRUTH_DURATIONS:
            results_dict[logo][f'{model_name} (duration)'] = 0.0

    # Test each model
    for model_name, model_path in MODELS.items():
        print(f"Testing {model_name}...")
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Skipping.")
            continue

        # Run detection
        start_time = time.time()
        logo_stats = modified_detect_logos(VIDEO_PATH, model_path, conf_thres=0.25)
        end_time = time.time()

        # Update results with predicted durations
        for logo in GROUND_TRUTH_DURATIONS:
            predicted_duration = logo_stats.get(logo, {"duration": 0.0})["duration"]
            results_dict[logo][f'{model_name} (duration)'] = round(predicted_duration, 2)

        print(f"Completed testing {model_name} in {end_time - start_time:.2f} seconds.")

    # Create DataFrame with logos as rows
    df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    df = df.rename(columns={'index': 'Logo'})
    df = df[['Logo', 'Actual Duration (s)', 'yolov8s (duration)', 'yolov8m (duration)', 
             'yolov8l (duration)', 'yolov8x (duration)', 'roboflow_model (duration)']]

    # Sort by Logo for consistency
    df = df.sort_values('Logo')

    # Save results to CSV
    output_file = f"model_test_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    test_models()