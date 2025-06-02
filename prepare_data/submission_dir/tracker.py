import os
import shutil
import argparse
import numpy as np
import cv2
import pandas as pd
from evaluate_network import evaluate

def match_csv_to_frames(video_path, csv_path, alignment_csv_path):
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
    
    #####################################################
    # # Check for required 'TIMESTAMP' column
    # if 'TIMESTAMP' not in timestamps_df.columns:
    #     raise ValueError("Error: CSV does not contain 'TIMESTAMP' column.")
    
    # csv_timestamps = timestamps_df['TIMESTAMP'].values
    # video_timestamps = np.array([(frame_index / fps) * 1000 for frame_index in range(total_frames)])
    #####################################################
    #####################################################
    # Check for required 'Ts' column
    if 'Ts' not in timestamps_df.columns:
        raise ValueError("Error: CSV does not contain 'Ts' column.")
    csv_timestamps = timestamps_df['Ts'].values
    video_timestamps = np.array([(frame_index / fps) * 1000 for frame_index in range(total_frames)])
    #####################################################

    # Initialize placeholders for matches
    matched_frame_indices = set()
    matched_rows = [-1] * len(csv_timestamps)  # Initialize all as unmatched

    for csv_index, csv_time in enumerate(csv_timestamps):
        # Compute absolute differences between this CSV timestamp and all video timestamps
        diffs = np.abs(video_timestamps - csv_time)
        
        # Mask already matched video frames
        diffs = np.where(np.isin(range(len(diffs)), list(matched_frame_indices)), np.inf, diffs)
        
        # Find the closest video frame index
        closest_frame_index = diffs.argmin()
        
        # Assign if a valid match is found
        if diffs[closest_frame_index] != np.inf:
            matched_rows[csv_index] = closest_frame_index
            matched_frame_indices.add(closest_frame_index)
    
    cap.release()
    
    # Prepare output DataFrame
    results = pd.DataFrame({
        'CSVIndex': range(len(csv_timestamps)),
        'CSV_Timestamp': csv_timestamps,
        'MatchedFrame': matched_rows,
        'FrameTimestamp': [video_timestamps[i] if i != -1 else None for i in matched_rows]
    })

    # Save results
    results.to_csv(alignment_csv_path, index=False)
    print(f"Matching complete. Results saved to {alignment_csv_path}.")


def process_video_with_matched_coordinates(video_path, alignment_csv_path, groundtruth_csv_path, dataset_path, start_frame, display=False):
    """
    Process a video by matching frames with rows from an alignment CSV and retrieving coordinates 
    from a groundtruth CSV. Saves matched frames and coordinates to specified directories.

    Args:
        video_path (str): Path to the input video file.
        alignment_csv_path (str): Path to the alignment CSV with FrameIndex and MatchedFrame.
        groundtruth_csv_path (str): Path to the groundtruth CSV with TIMESTAMP, X POSITION, Y POSITION.
        dataset_path (str): Path to the dataset directory where images and labels will be saved.

    Returns:
        None
    """
    # Clear and recreate dataset directories
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)  # Delete existing dataset directory
    os.makedirs(dataset_path, exist_ok=True)
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Error: Could not determine FPS.")

    # Load the alignment CSV
    alignment_df = pd.read_csv(alignment_csv_path)

    # Ensure alignment CSV has the required columns
    if 'MatchedFrame' not in alignment_df.columns or 'CSVIndex' not in alignment_df.columns:
        raise ValueError("Alignment CSV must contain 'MatchedFrame' and 'CSVIndex' columns.")

    # Load the groundtruth CSV
    groundtruth_df = pd.read_csv(groundtruth_csv_path)

    ##################################
    # # Ensure groundtruth CSV has the required columns
    # required_columns = ['TIMESTAMP']
    # if not all(col in groundtruth_df.columns for col in required_columns):
    #     raise ValueError(f"Groundtruth CSV must contain the following columns: {', '.join(required_columns)}")
    
    # # Extract groundtruth data as a dictionary {CSVIndex: (timestamp, x, y)}
    # groundtruth_data = {
    #     index: (row['TIMESTAMP'])
    #     for index, row in groundtruth_df.iterrows()
    # }
    ##################################
    ##################################
    # Ensure groundtruth CSV has the required columns
    required_columns = ['Ts'] #, ' X POSITION', ' Y POSITION']
    if not all(col in groundtruth_df.columns for col in required_columns):
        raise ValueError(f"Groundtruth CSV must contain the following columns: {', '.join(required_columns)}")

    groundtruth_data = {
        index: (row['Ts']) #, float(row[' X POSITION']), float(row[' Y POSITION']))
        for index, row in groundtruth_df.iterrows()
    }
    ##################################

    frame_index = start_frame  # Start processing frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame has a match in the alignment CSV
        matched_rows = alignment_df[alignment_df['MatchedFrame'] == frame_index]
        if not matched_rows.empty:
            for _, matched_row in matched_rows.iterrows():
                csv_index = matched_row['CSVIndex']

                # Get the coordinates from the groundtruth data
                if csv_index in groundtruth_data:
                    ###############################
                    # timestamp, x, y = groundtruth_data[csv_index] # use for processing training data
                    ###############################
                    ###############################
                    timestamp = groundtruth_data[csv_index]
                    ###############################
                    # Save the frame as an image
                    image_filename = f"frame_ms_{int(timestamp)}.png"
                    image_path = os.path.join(image_dir, image_filename)
                    cv2.imwrite(image_path, frame)

                    # # Save the coordinates as a text file
                    # label_filename = f"coords_ms_{int(timestamp)}.txt"
                    # label_path = os.path.join(label_dir, label_filename)
                    # with open(label_path, "w") as label_file:
                    #     label_file.write(f"Timestamp: {int(timestamp)} ms\n")
                    #     label_file.write(f"X Position: {x:.3f}\n")
                    #     label_file.write(f"Y Position: {y:.3f}\n")

                    # Overlay a green dot on the frame
                    # cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
                    # print(f"Frame {frame_index}: Saved image '{image_filename}' and coordinates '{label_filename}'.")
                    print(f"Frame {frame_index}: Saved image '{image_filename}'")


        if display:
            # Display the frame
            cv2.imshow("Processed Video", frame)

            # Break playback on 'q' key press
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

        frame_index += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()
    print("Video processing complete.")


def v2e(pf, f, linlog, on_thresh, off_thresh):
    """
    Generate events by comparing consecutive frames using a log intensity change model.
    Args:
        pf (np.ndarray): Previous frame.
        f (np.ndarray): Current frame.
        linlog (float): Linear-to-log intensity threshold.
        on_thresh (float): Threshold for ON events.
        off_thresh (float): Threshold for OFF events.
    Returns:
        np.ndarray: Binary event frame indicating ON (1) and OFF (1) events.
    """
    pf = pf.copy()
    f = f.copy()

    # Apply log-intensity transformation
    pf[pf <= linlog] *= np.log(linlog) / linlog
    pf[pf > linlog] = np.log(pf[pf > linlog])
    f[f <= linlog] *= np.log(linlog) / linlog
    f[f > linlog] = np.log(f[f > linlog])

    # Compute difference
    diff_frame = pf - f

    # Normalize ON and OFF events
    diff_frame[diff_frame > 0] = np.fix(diff_frame[diff_frame > 0] / on_thresh)
    diff_frame[diff_frame < 0] = np.fix(diff_frame[diff_frame < 0] / off_thresh)

    # Create binary event frame (merge on and off polarities)
    disp_frame = diff_frame.copy()
    disp_frame[disp_frame > 0] = 1  # ON events
    disp_frame[disp_frame == 0] = 0
    disp_frame[disp_frame < 0] = 1  # OFF events

    return disp_frame

def process_scene(scene_path, save=False, display=True):
    """
    Process a single scene folder to generate time-surface heatmaps, optionally display them as a video,
    and optionally save the heatmaps to a subfolder. Display can be toggled in real-time with a key press.

    Args:
        scene_path (str): Path to the scene directory containing images.
        save (bool): Whether to save the generated heatmaps to a subfolder. Defaults to False.
        display (bool): Whether to display the frames during processing. Defaults to True.
    """
    images_path = os.path.join(scene_path, "images")
    timesurfaces_path = os.path.join(scene_path, "timesurfaces")

    if not os.path.exists(images_path):
        print(f"No images directory found in {scene_path}. Skipping...")
        return

    # If saving is enabled, ensure the timesurfaces folder is clean
    if save:
        if os.path.exists(timesurfaces_path):
            for file in os.listdir(timesurfaces_path):
                os.remove(os.path.join(timesurfaces_path, file))
        else:
            os.makedirs(timesurfaces_path, exist_ok=True)

    # Read and sort image files by timestamp
    image_files = sorted(
        [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if len(image_files) < 1:
        print(f"No images to process in {scene_path}. Skipping...")
        return

    # Initialize parameters for v2e (hand crafted)
    linlog = 20
    # on_thresh = 0.06
    # off_thresh = 0.06
    on_thresh = 0.5
    off_thresh = 0.5
    tau = 0.5  # Decay constant

    # Initialize arrays for time surface computation
    first_image = cv2.imread(os.path.join(images_path, image_files[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    times_array = -tau * 3 * np.ones_like(first_image, dtype=np.float32)
    heatmap = np.zeros_like(first_image, dtype=np.float32)

    # Extract timestamps
    timestamps = [int(f.split('_')[-1].split('.')[0]) for f in image_files]  # Extract timestamp in ms

    # Process all frames
    for idx in range(len(image_files)):
        if idx == 0:
            # First frame: initialize with zeros
            heatmap.fill(0)
        else:
            prev_image_path = os.path.join(images_path, image_files[idx - 1])
            curr_image_path = os.path.join(images_path, image_files[idx])

            prev_frame = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            curr_frame = cv2.imread(curr_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            # Compute events using v2e
            events = v2e(prev_frame, curr_frame, linlog, on_thresh, off_thresh)

            # Update times_array with current timestamp for detected events
            t_i = timestamps[idx]
            times_array[events == 1] = t_i / 1000.0  # Convert to seconds for consistency

            # Compute heatmap
            time_diff = np.maximum(t_i / 1000.0 - times_array, 0)  # Ensure non-negative differences
            heatmap = np.exp(-time_diff / tau) * 255  # Scale for single-channel output

        # Save heatmap if enabled
        if save:
            output_filename = f"timesurf_ms_{timestamps[idx]}.png"
            print(f'Saving {output_filename} to {timesurfaces_path}')
            heatmap_filepath = os.path.join(timesurfaces_path, output_filename)
            cv2.imwrite(heatmap_filepath, heatmap.astype(np.uint8))  # Save as single-channel PNG

        # Display heatmap if enabled
        if display:
            heatmap_rgb = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Time Surface Heatmap', heatmap_rgb)
            key = cv2.waitKey(50)  # Wait 50ms for key input
            if key == ord('q'):  # Press 'q' to quit early
                break
            elif key == ord('d'):  # Press 'd' to toggle display mode
                display = not display
        else:
            # Non-display mode: Still allow toggling and quitting
            key = cv2.waitKey(1)  # Shorter wait to ensure responsiveness
            if key == ord('q'):  # Press 'q' to quit early
                break
            elif key == ord('d'):  # Press 'd' to toggle display mode
                display = not display

    # Close the OpenCV window after processing
    if display:
        cv2.destroyAllWindows()

    print(f"Processing complete for {scene_path}")
    if save:
        print(f"Timesurfaces saved to: {timesurfaces_path}")

def process_scenes(scene_dir_path, save=False, display=True):
    """
    Processes each scene in the scenes directory.
    Reads images in each scene folder sequentially, sorted by timestamp in the filename.

    Args:
        scene_dir_path (str): Path to the `scenes` directory containing scene subfolders.
        save (bool): Whether to save the generated heatmaps to subfolders. Defaults to False.
        display (bool): Whether to display the frames during processing. Defaults to True.
    """
    # Get all scene subdirectories sorted by scene number
    # scene_folders = sorted(
    #     [f for f in os.listdir(scene_dir_path) if os.path.isdir(os.path.join(scene_dir_path, f))],
    #     key=lambda x: int(x.replace("scene", ""))
    # )

    print(f"Processing {scene_dir_path}...")
    process_scene(scene_dir_path, save=save, display=display)
    print("All scenes processed.")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and groundtruth data.")
    parser.add_argument('video_path', help="Path to the video file")  # Positional argument for video path
    parser.add_argument('times_csv_path', help="Path to the groundtruth CSV file")  # Positional argument for groundtruth CSV path
    parser.add_argument('--alignment_csv_path', default="./csv_to_frames_matching.csv", help="Path to save the alignment CSV file (default: ./csv_to_frames_matching.csv)")
    parser.add_argument('--dataset_path', default="./test-dataset", help="Path to save the processed dataset (default: ./test-dataset)")
    parser.add_argument('--scene_name', default="test_scene", help="Scene name (folder for .pngs extracted from .mp4 video file)")
    parser.add_argument('--model_path', default="./epoch_29_loss_18.7249.pth", help="Path to model checkpoint")
    parser.add_argument('--start_frame', type=int, default=0, help="Starting frame number for alignment (default: 0)")

    args = parser.parse_args()

    # Correctly join paths
    # args.dataset_path = os.path.join(args.dataset_path, args.scene_name)

    # Call the functions in sequence
    # match_csv_to_frames(args.video_path, args.times_csv_path, args.alignment_csv_path)
    # process_video_with_matched_coordinates(args.video_path, args.alignment_csv_path, args.times_csv_path, args.dataset_path, args.start_frame)
    process_scenes(args.dataset_path, save=True, display=False)

    # evaluate model and save data to output.csv file
    evaluate(args.dataset_path, [args.scene_name], args.model_path, save_preds=False)