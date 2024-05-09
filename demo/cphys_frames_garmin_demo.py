import logging

def moving_window_inference(directory, garmin_path, fs, model_path, video_id):
    logging.basicConfig(level=logging.INFO)
    frames, timestamps = load_frames_and_times(directory)
    garmin_data = pd.read_csv(garmin_path, parse_dates=['timestamp'])
    
    results = []
    start_idx = 0

    while start_idx + 300 <= len(frames):
        end_idx = min(start_idx + 300, len(frames))
        if end_idx - start_idx < 210:
            logging.info("Not enough frames to process; stopping early.")
            break
        
        window_frames = frames[start_idx:end_idx]
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx - 1]

        if window_frames.size == 0:
            logging.warning(f"No frames to process between {start_time} and {end_time}.")
            start_idx += 300
            continue
        
        rppg, hr = process_frames_with_model(window_frames, fs, model_path)
        avg_hr = average_garmin_hr(garmin_data, start_time, end_time - start_time)
        
        results.append({
            'video_id': video_id,
            'start_time': start_time,
            'end_time': end_time,
            'model_hr': hr,
            'garmin_hr': avg_hr
        })

        start_idx += 300

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f'{video_id}_cphys_comparison_results_{timestamp}.npy', results)
    return results
