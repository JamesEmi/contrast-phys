import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
from datetime import datetime
import torch
from tqdm import tqdm  # Import tqdm
import os

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

def process_all_frames(directory, garmin_path, fs, model_path, video_id):
    """Process all frames and compare with average Garmin heart rate."""
    frames, timestamps = load_frames_and_times(directory)
    garmin_data = pd.read_csv(garmin_path, parse_dates=['timestamp'])
    start_time = timestamps[0]
    end_time = timestamps[-1]

    # Add tqdm progress bar
    with tqdm(total=len(frames), desc="Processing frames") as pbar:
        rppg, hr = process_frames_with_model(frames, fs, model_path)
        pbar.update(len(frames))

    avg_hr = average_garmin_hr(garmin_data, start_time, end_time - start_time)

    result = {
        'video_id': video_id,
        'start_time': start_time,
        'end_time': end_time,
        'model_hr': hr,
        'garmin_hr': avg_hr,
        'rppg_waveform': rppg
    }

    return result

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
    result = process_all_frames(args.directory, args.garmin_path, args.frame_rate, args.model_path, video)

    # Print and plot the results
    print(f"Video ID: {result['video_id']}")
    print(f"Start Time: {result['start_time']}")
    print(f"End Time: {result['end_time']}")
    print(f"Model Heart Rate: {result['model_hr']:.2f} BPM")
    print(f"Garmin Heart Rate: {result['garmin_hr']:.2f} BPM")

    plt.figure(figsize=(12, 4))
    plt.plot(result['rppg_waveform'])
    plt.title('rPPG Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    main()
