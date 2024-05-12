import cv2
import numpy as np
import torch
from PhysNetModel import PhysNet
from utils_sig import *
import matplotlib.pyplot as plt
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from datetime import datetime

def load_and_preprocess_frames(directory):
    # List all files in the directory and sort them
    files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpeg')])
    frames = [np.array(Image.open(file)) for file in files]
    return np.array(frames)

def face_detection(frames):
    device = torch.device('cpu')
    mtcnn = MTCNN(device=device)
    face_list = []

    # Detect face in the first frame and apply the same crop to all frames
    boxes, _ = mtcnn.detect(frames[0])
    box_len = np.max([boxes[0,2]-boxes[0,0], boxes[0,3]-boxes[0,1]])
    box_half_len = np.round(box_len / 2 * 1.1).astype('int')
    box_mid_y = np.round((boxes[0,3] + boxes[0,1]) / 2).astype('int')
    box_mid_x = np.round((boxes[0,2] + boxes[0,0]) / 2).astype('int')

    for frame in frames:
        cropped_face = frame[box_mid_y-box_half_len:box_mid_y+box_half_len, box_mid_x-box_half_len:box_mid_x+box_half_len]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        face_list.append(cropped_face)

    face_list = np.array(face_list) # (T, H, W, C)
    face_list = np.transpose(face_list, (3,0,1,2)) # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]
    return face_list

directory_path = '/james/experiment_data/20240508_church-processed/03/first_150'
fps = 30  # Define FPS as needed based on your frame rate

frames = load_and_preprocess_frames(directory_path)
face_list = face_detection(frames)

print('\nrPPG estimation')
device = torch.device('cpu')

with torch.no_grad():
    face_list = torch.tensor(face_list.astype('float32')).to(device)
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('/james/contrastphys_node_code/model1/src/contrastphys/contrastphys/contrast-phys-demo/model_weights.pt', map_location=device))
    rppg = model(face_list)[:,-1, :]
    rppg = rppg[0].detach().cpu().numpy()
    rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

fig, (ax1, ax2) = plt.subplots(2, figsize=(20,10))

ax1.plot(np.arange(len(rppg))/fps, rppg)
ax1.set_xlabel('time (sec)')
ax1.grid('on')
ax1.set_title('rPPG waveform')

ax2.plot(psd_x, psd_y)
ax2.set_xlabel('heart rate (bpm)')
ax2.set_xlim([40,200])
ax2.grid('on')
ax2.set_title('PSD')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'./results/01_150f_results_{timestamp}.png')

print('heart rate: %.2f bpm'%hr)
