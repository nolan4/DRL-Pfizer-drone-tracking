import cv2
import pandas as pd
import numpy as np
import pdb

def match_csv_to_frames(video_path, csv_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    
    # Get video FPS and number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read CSV file
    try:
        timestamps_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise ValueError(f"Error: CSV file not found at {csv_path}.")
    
    pdb.set_trace()
    
    # Check for required 'TIMESTAMP' column
    if 'TIMESTAMP' not in timestamps_df.columns:
        raise ValueError("Error: CSV does not contain 'TIMESTAMP' column.")
    
    csv_timestamps = timestamps_df['TIMESTAMP'].values
    video_timestamps = np.array([(frame_index / fps) * 1000 for frame_index in range(total_frames)])
    
    pdb.set_trace()
    
    # Initialize placeholders for matches
    matched_frame_indices = set()
    matched_rows_reverse = [-1] * len(csv_timestamps)  # Initialize all as unmatched

    for csv_index, csv_time in enumerate(csv_timestamps):
        # Compute absolute differences between this CSV timestamp and all video timestamps
        diffs = np.abs(video_timestamps - csv_time)
        
        # Mask already matched video frames
        diffs = np.where(np.isin(range(len(diffs)), list(matched_frame_indices)), np.inf, diffs)
        
        # Find the closest video frame index
        closest_frame_index = diffs.argmin()
        
        # Assign if a valid match is found
        if diffs[closest_frame_index] != np.inf:
            matched_rows_reverse[csv_index] = closest_frame_index
            matched_frame_indices.add(closest_frame_index)
    
    cap.release()
    
    # Prepare output DataFrame
    results_reverse = pd.DataFrame({
        'CSVIndex': range(len(csv_timestamps)),
        'CSV_Timestamp': csv_timestamps,
        'MatchedFrame': matched_rows_reverse,
        'FrameTimestamp': [video_timestamps[i] if i != -1 else None for i in matched_rows_reverse]
    })

    # Save results
    results_reverse.to_csv('csv_to_frames_matching.csv', index=False)
    print("Matching complete. Results saved to 'csv_to_frames_matching.csv'.")

# Example usage
match_csv_to_frames(
    '/home/nolan4/projects/pfizerDRL/materials-en/practice_video.mp4', 
    '/home/nolan4/projects/pfizerDRL/materials-en/practice_groundtruth.csv'
)
