import os
import cv2
import numpy as np
import h5py
import cv2
import pandas as pd

                
def openface_h5(video_path, landmark_path, h5_path, store_size=128):
    """
    crop face from OpenFace landmarks and save a video as .h5 file.

    video_path: the face video path
    landmark_path: landmark .csv file generated by OpenFace.
    h5_path: the path to save the h5_file
    store_size: the cropped face is resized to 128 (default).
    """

    landmark = pd.read_csv(landmark_path)

    with h5py.File(h5_path, 'w') as f:

        total_num_frame = len(landmark)

        cap = cv2.VideoCapture(video_path)
        
        for frame_num in range(total_num_frame):

            if landmark['success'][frame_num]:

                lm_x = []
                lm_y = []
                for lm_num in range(68):
                    lm_x.append(landmark['x_%d'%lm_num][frame_num])
                    lm_y.append(landmark['y_%d'%lm_num][frame_num])

                lm_x = np.array(lm_x)
                lm_y = np.array(lm_y)

                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy-miny)*0.2
                miny = miny - y_range_ext


                cnt_x = np.round((minx+maxx)/2).astype('int')
                cnt_y = np.round((maxy+miny)/2).astype('int')
                
                break

        bbox_size=np.round(1.5*(maxy-miny)).astype('int')
        
        ########### init dataset in h5 ##################
        if store_size==None:
            store_size = bbox_size
            
        imgs = f.create_dataset('imgs', shape=(total_num_frame, store_size, store_size, 3), 
                                        dtype='uint8', chunks=(1,store_size, store_size,3),
                                        compression="gzip", compression_opts=4)

        for frame_num in range(total_num_frame):

            if landmark['success'][frame_num]:

                lm_x_ = []
                lm_y_ = []
                for lm_num in range(68):
                    lm_x_.append(landmark['x_%d'%lm_num][frame_num])
                    lm_y_.append(landmark['y_%d'%lm_num][frame_num])

                lm_x_ = np.array(lm_x_)
                lm_y_ = np.array(lm_y_)
                
                lm_x = 0.9*lm_x+0.1*lm_x_
                lm_y = 0.9*lm_y+0.1*lm_y_
                
                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy-miny)*0.2
                miny = miny - y_range_ext


                cnt_x = np.round((minx+maxx)/2).astype('int')
                cnt_y = np.round((maxy+miny)/2).astype('int')
                
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            ########## for bbox ################
            bbox_half_size = int(bbox_size/2)
            
            face = np.take(frame, range(cnt_y-bbox_half_size, cnt_y-bbox_half_size+bbox_size),0, mode='clip')
            face = np.take(face, range(cnt_x-bbox_half_size, cnt_x-bbox_half_size+bbox_size),1, mode='clip')
            
            if store_size==bbox_size:
                imgs[frame_num] = face
            else:
                imgs[frame_num] = cv2.resize(face, (store_size,store_size))

        cap.release()


def main(landmark_dir, video_base_dir, output_dir, store_size=128):
    """
    Processes all video and landmark pairs in given directories and stores the results.
    Parameters:
        landmark_dir (str): Directory containing landmark files.
        video_base_dir (str): Directory containing subject directories with videos.
        output_dir (str): Directory to store output .h5 files.
        store_size (int, optional): Size to which faces are resized. Default is 128.
    """
    # Iterate over each landmark file in the landmark directory
    for landmark_filename in os.listdir(landmark_dir):
        if landmark_filename.endswith('.csv'):
            subject_id = os.path.splitext(landmark_filename)[0]  # e.g., 'subject1'
            landmark_path = os.path.join(landmark_dir, landmark_filename)
            video_path = os.path.join(video_base_dir, subject_id, 'vid.avi')
            h5_filename = f"{subject_id[7:]}.h5"  # Extracts the number from 'subject1' and makes '1.h5'
            h5_path = os.path.join(output_dir, h5_filename)
            
            if os.path.exists(video_path) and os.path.exists(landmark_path):
                print(f"Processing {subject_id}...")
                openface_h5(video_path, landmark_path, h5_path, store_size)
            else:
                print(f"Missing files for {subject_id}, skipping...")

if __name__ == "__main__":
    landmark_dir = '/data-fast/james/triage/datasets/UBFC-rPPG-openface'
    video_base_dir = '/data-fast/james/triage/datasets/UBFC-rPPG'
    output_dir = '/data-fast/james/triage/datasets/UBFC-rPPG-cphyspreprocessed'
    main(landmark_dir, video_base_dir, output_dir)
