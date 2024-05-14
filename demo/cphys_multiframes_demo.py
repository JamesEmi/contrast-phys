import cv2
import numpy as np
import torch
from PhysNetModel import PhysNet
from utils_sig import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from facenet_pytorch import MTCNN
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast

def load_and_preprocess_frames(directory, start_frame, num_frames):
    start_time = datetime.now()
    files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpeg')])
    frames = [np.array(Image.open(files[i])) for i in range(start_frame, start_frame + num_frames)]
    elapsed_time = datetime.now() - start_time
    print(f"Loading files took: {elapsed_time.total_seconds()} seconds")
    return np.array(frames)

def face_detection(frames):
    # device = torch.device('cpu')
    device = torch.device('cuda')
    print(f'Device being used for face detection is {device}')
    mtcnn = MTCNN(device=device)
    boxes, _ = mtcnn.detect(frames[0])
    box_len = np.max([boxes[0,2]-boxes[0,0], boxes[0,3]-boxes[0,1]])
    box_half_len = np.round(box_len / 2 * 1.1).astype('int')
    box_mid_y = np.round((boxes[0,3] + boxes[0,1]) / 2).astype('int')
    box_mid_x = np.round((boxes[0,2] + boxes[0,0]) / 2).astype('int')

    face_list = []
    for frame in frames:
        cropped_face = frame[box_mid_y-box_half_len:box_mid_y+box_half_len, box_mid_x-box_half_len:box_mid_x+box_half_len]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        face_list.append(cropped_face)

    face_list = np.array(face_list) # Convert list to numpy array
    face_list = np.transpose(face_list, (3, 0, 1, 2)) # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]
    return face_list

def inference_pipeline(directory_path, window_size=300, step_size=120, num_windows=3):
    # device = torch.device('cpu')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    print(f'Device being used for inference is {device}')
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('/james/contrastphys_node_code/model1/src/contrastphys/contrastphys/contrast-phys-demo/model_weights.pt', map_location=device))
    model.half()  # Convert model to half-precision
    print('Using half precision!!!')


    all_rppg = []
    all_psd_y = []
    all_psd_x = []
    heart_rates = []

    for i in tqdm(range(num_windows), desc="Processing windows"):
        start_frame = i * step_size
        start_time = datetime.now()
        frames = load_and_preprocess_frames(directory_path, start_frame, window_size)
        inference_start_time = datetime.now()
        face_list = face_detection(frames)

        with torch.no_grad():
            with autocast():
                face_list = torch.tensor(face_list.astype('float16')).to(device)
                # face_tensor = torch.tensor(face_list, dtype=torch.float16).to(device)  # Ensure data is in half-precision
                # face_tensor = torch.from_numpy(face_list).float().to(device).half()  # Ensure data is on GPU and in half-precision
                # rppg = model(face_tensor)[:,-1, :]
                rppg = model(face_list)[:,-1, :]
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=30)

            hr, psd_y, psd_x = hr_fft(rppg, fs=30)
            
        inference_elapsed_time = datetime.now() - inference_start_time
        all_rppg.append(rppg)
        all_psd_y.append(psd_y)
        all_psd_x.append(psd_x)
        heart_rates.append(hr)
        print(f'Heart rate for window {i+1}: {hr:.2f} bpm')
        print(f"Inference for window {i+1} took: {inference_elapsed_time.total_seconds()} seconds")

    mean_hr = np.mean(heart_rates)
    std_hr = np.std(heart_rates)
    print(f'Mean heart rate: {mean_hr:.2f} bpm, Standard deviation: {std_hr:.2f}')


    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(2, 1, 1)
    for idx, rppg in enumerate(all_rppg):
        ax1.plot(np.arange(len(rppg))/30 + (idx * step_size / 30), rppg, label=f'{idx * 4}-{idx * 4 + 10}s window')
    ax1.set_xlabel('Time (sec)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('rPPG Waveforms', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    ground_truth_hr = 61.95  # Ground truth heart rate
    for idx, (psd_y, psd_x) in enumerate(zip(all_psd_y, all_psd_x)):
        ax2.plot(psd_x, psd_y, label=f'{idx * 4}-{idx * 4 + 10}s window')
        ax2.axvline(x=heart_rates[idx], color='r', linestyle='--')  # Inferred heart rate
    ax2.axvline(x=ground_truth_hr, color='black', linewidth=2, linestyle='--', label='Ground Truth HR')  # Ground truth heart rate
    ax2.set_xlabel('Heart Rate (bpm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power', fontsize=12, fontweight='bold')
    ax2.set_xlim([40, 200])
    ax2.set_title('Power Spectral Density', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'./results/01_AI2C_last540_results_{timestamp}.png')
    plt.show()

# directory_path = '/james/experiment_data/20240508_church-processed/03/first_150'
directory_path = '/james/experiment_data/20240508_church-processed/01/images_last540'
inference_pipeline(directory_path)
