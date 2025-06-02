import os
import cv2
import pandas as pd
import shutil

def process_video_with_matched_coordinates(video_path, alignment_csv_path, groundtruth_csv_path, dataset_path, start_frame):
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

    # Ensure groundtruth CSV has the required columns
    required_columns = ['TIMESTAMP', ' X POSITION', ' Y POSITION']
    if not all(col in groundtruth_df.columns for col in required_columns):
        raise ValueError(f"Groundtruth CSV must contain the following columns: {', '.join(required_columns)}")

    # Extract groundtruth data as a dictionary {CSVIndex: (timestamp, x, y)}
    groundtruth_data = {
        index: (row['TIMESTAMP'], float(row[' X POSITION']), float(row[' Y POSITION']))
        for index, row in groundtruth_df.iterrows()
    }

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
                    timestamp, x, y = groundtruth_data[csv_index]
                    # Save the frame as an image
                    image_filename = f"frame_ms_{int(timestamp)}.png"
                    image_path = os.path.join(image_dir, image_filename)
                    cv2.imwrite(image_path, frame)

                    # Save the coordinates as a text file
                    label_filename = f"coords_ms_{int(timestamp)}.txt"
                    label_path = os.path.join(label_dir, label_filename)
                    with open(label_path, "w") as label_file:
                        label_file.write(f"Timestamp: {int(timestamp)} ms\n")
                        label_file.write(f"X Position: {x:.3f}\n")
                        label_file.write(f"Y Position: {y:.3f}\n")

                    # Overlay a green dot on the frame
                    cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
                    print(f"Frame {frame_index}: Saved image '{image_filename}' and coordinates '{label_filename}'.")

        # Display the frame
        cv2.imshow("Processed Video", frame)

        # Break playback on 'q' key press
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

# Example usage
if __name__ == "__main__":
    # Input video file and CSV file paths
    video_file_path = "/home/nolan4/projects/pfizerDRL/materials-en/practice_video.mp4"  # Replace with your video file path
    alignment_csv_path = "/home/nolan4/projects/pfizerDRL/code2/csv_to_frames_matching.csv"  # Replace with your alignment CSV file path
    groundtruth_csv_path = "/home/nolan4/projects/pfizerDRL/materials-en/practice_groundtruth.csv"  # Replace with your groundtruth CSV file path
    dataset_path = "/home/nolan4/projects/pfizerDRL/dataset"  # Replace with your desired dataset directory path

    start_frame = 1 # to fix positional alignment issue between labels and frames

    # Process the video
    process_video_with_matched_coordinates(video_file_path, alignment_csv_path, groundtruth_csv_path, dataset_path, start_frame)
