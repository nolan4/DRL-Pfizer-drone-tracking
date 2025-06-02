import os
import cv2
import re

def display_dataset_images_and_labels(dataset_path):
    """
    Load and display images and corresponding labels from a dataset directory,
    plotting the green dot based on coordinates in the label files.

    Args:
        dataset_path (str): Path to the dataset directory containing 'images' and 'labels'.

    Returns:
        None
    """
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise ValueError("Both 'images' and 'labels' directories must exist within the dataset path.")

    # Helper function to extract integers from filenames for sorting
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    # Get sorted lists of image and label file paths based on extracted numbers
    image_files = sorted(os.listdir(image_dir), key=extract_number)
    label_files = sorted(os.listdir(label_dir), key=extract_number)

    if not image_files or not label_files:
        raise ValueError("No images or labels found in the dataset directories.")

    if len(image_files) != len(label_files):
        raise ValueError("Number of images and labels must match.")

    print(f"Found {len(image_files)} images and {len(label_files)} labels.")

    index = 0
    total_files = len(image_files)

    while True:
        # Ensure the index is valid
        index = max(0, min(index, total_files - 1))

        image_path = os.path.join(image_dir, image_files[index])
        label_path = os.path.join(label_dir, label_files[index])

        # Load and display the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image '{image_path}'.")
            break

        # Extract coordinates from the label file
        with open(label_path, 'r') as label_file:
            label_content = label_file.readlines()
            x = None
            y = None
            for line in label_content:
                if line.startswith("X Position:"):
                    x = float(line.split(":")[1].strip())
                if line.startswith("Y Position:"):
                    y = float(line.split(":")[1].strip())

        # Plot the green dot if coordinates are found
        if x is not None and y is not None:
            cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

        # Display the image
        cv2.imshow("Image", image)

        # Print the label information to the terminal
        print(f"Image: {image_files[index]}")
        print(f"Label: {label_files[index]}\n      {'      '.join(label_content)}")
        # print(f"Label Content:\n{''.join(label_content)}")

        # Wait for user input
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break
        elif key == 81:  # Left arrow key
            index -= 1  # Go to previous image
        elif key == 83:  # Right arrow key
            index += 1  # Go to next image

    cv2.destroyAllWindows()
    print("Dataset browsing complete.")

# Example usage
if __name__ == "__main__":
    dataset_path = "/home/nolan4/projects/pfizerDRL/dataset"

    # Call the function
    display_dataset_images_and_labels(dataset_path)
