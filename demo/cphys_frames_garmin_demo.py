import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
from datetime import datetime, timedelta
import torch
from tqdm import tqdm
import os
import logging

from PhysNetModel import PhysNet
from utils_sig import butter_bandpass, hr_fft

def load_frames_and_times(directory):
    """Load all frames from directory and extract their timestamps."""
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpeg')])
    frames = []
    timestamps = []

    for file in files:
        timestamp_str = os.path.splitext(os.path.basename(file))[0].split('_')[1:]
        timestamp_text = '_'.join(timestamp_str)
        try:
            timestamp = datetime.strptime(timestamp_text, "%Y-%m-%d_%H-%M-%S.%f")
        except ValueError:
            continue
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        timestamps.append(timestamp)

    return np.array(frames), timestamps

def process_frames_with_model(frames, fs, model_path):
    """Process frames through PhysNet model to estimate rPPG and heart rate."""
    device = torch.device('cpu')
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        tensor_frames = torch.tensor(frames.transpose((3, 0, 1, 2)), dtype=torch.float32).unsqueeze(0).to(device)
        rppg = model(tensor_frames)[:,-1, :].squeeze().detach().cpu().numpy()
        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fs)
        hr, _, _ = hr_fft(rppg, fs=fs)
    return rppg, hr

def moving_window_inference(directory, garmin_path, fs, model_path, video_id):
    """Perform moving window inference matching Garmin data for heart rate comparison."""
    frames, timestamps = load_frames_and_times(directory)
    garmin_data = pd.read_csv(garmin_path, parse_dates=['timestamp'])
    results = []
    start_idx = 0
    while start_idx + 300 <= len(frames):
        if len(timestamps) < start_idx + 210:
            break  # Ensure at least 210 frames are processed
        end_idx = min(start_idx + 300, len(frames))
        window_frames = frames[start_idx:end_idx]
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx-1]

        # Add tqdm for progress display during processing
        for _ in tqdm(range(1), desc=f"Processing window {start_idx//300 + 1}"):
            rppg, hr = process_frames_with_model(window_frames, fs, model_path)
            avg_hr = average_garmin_hr(garmin_data, start_time, end_time - start_time)

            results.append({
                'video_id': video_id,
                'start_time': start_time,
                'end_time': end_time,
                'model_hr': hr,
                'garmin_hr': avg_hr
            })

        start_idx += 300  # Move to the next window

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f'{video_id}_cphys_comparison_results_{timestamp}.npy', results)
    return results
    
def average_garmin_hr(garmin_data, start_time, duration):
    """Average Garmin heart rate over the given duration starting from start_time."""
    end_time = start_time + duration
    interval_data = garmin_data[(garmin_data['timestamp'] >= start_time) & (garmin_data['timestamp'] <= end_time)]
    return interval_data['heart_rate'].mean()


def main():
    parser = argparse.ArgumentParser(description='rPPG Heart Rate Estimation from Frames')
    parser.add_argument('--directory', type=str, default='/james/experiment_data/20240508_church-processed/01/images', help='Path to the directory containing frame images')
    parser.add_argument('--garmin_path', type=str, default='/james/experiment_data/garmin_data_processed/garmin_heartrate_01.csv', help='Path to the Garmin data CSV file')
    parser.add_argument('--model_path', type=str, default='/james/contrastphys_node_code/model1/src/contrastphys/contrastphys/contrast-phys-demo/model_weights.pt', help='Path to the model weight file')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate of the video')
    args = parser.parse_args()
    
    video = '01'  # Change this for each run
    results = moving_window_inference(args.directory, args.garmin_path, args.frame_rate, args.model_path, video)
    print("Processing complete. Results saved to 'inference_comparison_results.npy'.")

if __name__ == '__main__':
    main()
