import os
import cv2
import numpy as np

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

    # Create binary event frame
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

    # Initialize parameters for v2e
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
    scene_folders = sorted(
        [f for f in os.listdir(scene_dir_path) if os.path.isdir(os.path.join(scene_dir_path, f))],
        key=lambda x: int(x.replace("scene", ""))
    )

    for scene_folder in scene_folders:
        scene_path = os.path.join(scene_dir_path, scene_folder)
        print(f"Processing {scene_folder}...")
        process_scene(scene_path, save=save, display=display)
    print("All scenes processed.")

if __name__ == "__main__":
    # Define the path to the scenes directory
    scene_dir_path = '/home/nolan4/projects/pfizerDRL/dataset/scenes'

    # Check if the path exists before processing
    if not os.path.exists(scene_dir_path):
        print(f"Scenes directory does not exist: {scene_dir_path}")
    else:
        # Process scenes with saving enabled and real-time display
        process_scenes(scene_dir_path, save=True, display=False)
