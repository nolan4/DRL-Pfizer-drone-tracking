import os
import shutil
import pandas as pd

def move_files_based_on_ranges(scene_boundaries_path, dataset_dir_path):
    """
    Organizes images and labels into scene folders based on time ranges in a CSV.
    Deletes and recreates the `scenes` directory to ensure it is clean.

    Args:
        scene_boundaries_path (str): Path to the CSV file with START_TIME and END_TIME.
        dataset_dir_path (str): Base directory containing `images` and `labels` subdirectories.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(scene_boundaries_path)

    # Define paths for images, labels, and scenes directory
    images_dir = os.path.join(dataset_dir_path, "images")
    labels_dir = os.path.join(dataset_dir_path, "labels")
    scenes_dir = os.path.join(dataset_dir_path, "scenes")

    # Delete the `scenes` directory if it exists, then recreate it
    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)
    os.makedirs(scenes_dir, exist_ok=True)

    for index, row in df.iterrows():
        scene_folder = f"scene{index + 1}"
        scene_images_dir = os.path.join(scenes_dir, scene_folder, "images")
        scene_labels_dir = os.path.join(scenes_dir, scene_folder, "labels")

        # Create scene subfolders
        os.makedirs(scene_images_dir, exist_ok=True)
        os.makedirs(scene_labels_dir, exist_ok=True)

        # Get start and end times
        start_time, end_time = row["START_TIME"], row["END_TIME"]

        # Copy images that fall within the range
        for file in os.listdir(images_dir):
            try:
                timestamp = int(file.split('_')[-1].split('.')[0])
                if start_time <= timestamp <= end_time:
                    shutil.copy(os.path.join(images_dir, file), os.path.join(scene_images_dir, file))
                    print(f'{file} written to scene folder {scene_images_dir}')
            except ValueError:
                continue

        # Copy labels that fall within the range
        for file in os.listdir(labels_dir):
            try:
                # Assuming label filenames are in the format 'timestamp.txt'
                timestamp = int(file.split('_')[-1].split('.')[0])
                if start_time <= timestamp <= end_time:
                    shutil.copy(os.path.join(labels_dir, file), os.path.join(scene_labels_dir, file))
                    print(f'{file} written to scene folder {scene_labels_dir}')
            except ValueError:
                continue

if __name__ == "__main__":
    # Define paths to the scene boundaries CSV and dataset directory
    scene_boundaries_path = '/path/to/projects/pfizerDRL/code2/scene_boundaries.txt'
    dataset_dir_path = '/path/to/projects/pfizerDRL/dataset'

    # Call the function to organize files
    move_files_based_on_ranges(scene_boundaries_path, dataset_dir_path)
