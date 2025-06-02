import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pdb

# Set PyTorch print options for higher precision
torch.set_printoptions(precision=8, sci_mode=False)

class drlpDataset(Dataset):
    def __init__(self, dataset_dir, scene_list, image_size, grid_size, augment=True):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        
        self.frame_paths, self.timesurf_paths, self.labels_paths = [], [], []
        for scene in scene_list:
            scene_path = os.path.join(dataset_dir, 'scenes', scene)
            self.frame_paths.extend(sorted(os.path.join(scene_path, "images", f) for f in os.listdir(os.path.join(scene_path, "images")) if f.endswith('.png')))
            self.timesurf_paths.extend(sorted(os.path.join(scene_path, "timesurfaces", f) for f in os.listdir(os.path.join(scene_path, "timesurfaces")) if f.endswith('.png')))
            self.labels_paths.extend(sorted(os.path.join(scene_path, "labels", f) for f in os.listdir(os.path.join(scene_path, "labels")) if f.endswith('.txt')))

    def __len__(self):
        return len(self.frame_paths)

    def load_label(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            x_pos = float(lines[1].split(':')[1].strip())
            y_pos = float(lines[2].split(':')[1].strip())
            # print(f'y_pos {y_pos}, x_coord {x_pos}')
        return np.array([y_pos, x_pos], dtype=np.float32) # [height_coord, width_coord]

    def augment_image_label(self, image, label):
        """
        Apply center-based flips and rotations to the image and normalize the label.

        Args:
            image: Input image tensor.
            label: Tuple of (x, y) coordinates (height, width).
            H: Image height.
            W: Image width.

        Returns:
            Augmented image and updated label.
        """
        # Center of the image
        H, W = self.image_size
        cy, cx = H / 2, W / 2

        # Augmentation transformations
        augmentations = [
            lambda img, lbl: (img, lbl),  # No change
            lambda img, lbl: (torch.flip(img, dims=[2]),  # Horizontal flip
                            np.array([lbl[0], 2 * cx - lbl[1]])),  # Adjust x-coordinate
            lambda img, lbl: (torch.flip(img, dims=[1]),  # Vertical flip
                            np.array([2 * cy - lbl[0], lbl[1]])),  # Adjust y-coordinate
            lambda img, lbl: (torch.flip(img, dims=[1, 2]),  # 180-degree rotation
                            np.array([2 * cy - lbl[0], 2 * cx - lbl[1]]))  # Adjust both coordinates
        ]

        # Apply random augmentation
        aug_idx = np.random.randint(0, len(augmentations))
        # print(f"Augmentation Index: {aug_idx}")  # Debugging to check applied augmentation
        augmented_image, augmented_label = augmentations[aug_idx](image, label)

        return augmented_image, augmented_label

    def __getitem__(self, idx):
        # Load images
        frame = cv2.imread(self.frame_paths[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        timesurface = cv2.imread(self.timesurf_paths[idx], cv2.IMREAD_GRAYSCALE)  # Grayscale timesurface

        # Convert to tensors
        input_image = torch.from_numpy(timesurface).unsqueeze(0).float() / 255.0 # originally timesurface_tensor

        # Load label and scale to resized image
        label = self.load_label(self.labels_paths[idx]) # idx instead of 0 # [y_coord, x_coord] aka (height_coord, width_coord)

        # Perform augmentation on both image and label
        if self.augment:
            input_image, label = self.augment_image_label(input_image, label)  # self.image(size) is [height, width]

        # normalize coordinates between [0,1] | label[0] is height_coord, self.image_size[0] is height
        scaled_label = np.array([label[0] / self.image_size[0], label[1] / self.image_size[1]])

        heatmap, offsets, mask = self.create_ground_truth(self.grid_size, scaled_label)

        return input_image, {
            "heatmap_gt": torch.tensor(heatmap, dtype=torch.float32),
            "offsets_gt": torch.tensor(offsets, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "scaled_label": torch.tensor(scaled_label, dtype=torch.float32)
        }          
          

    def create_ground_truth(self, grid_size, scaled_label):
        """
        Create ground truth heatmap, offsets, and mask for CenterNet-style training.

        Args:
            grid_size (tuple): Grid dimensions (H, W) of the feature map.
            image_size (tuple): Image dimensions (height, width).
            scaled_label (tuple): Normalized coordinates (x, y) of the keypoint, values in [0, 1].

        Returns:
            tuple: (heatmap, offsets, mask)
        """
        # Unpack grid size and image size
        grid_H, grid_W = grid_size

        # Initialize tensors
        heatmap = np.zeros((1, grid_H, grid_W), dtype=np.float32)
        offsets = np.zeros((2, grid_H, grid_W), dtype=np.float32)
        mask = np.zeros((1, grid_H, grid_W), dtype=np.float32)

        # Map keypoint to grid
        x_grid = int(scaled_label[0] * grid_W)  # Grid x-coordinate
        y_grid = int(scaled_label[1] * grid_H)  # Grid y-coordinate

        # Skip keypoints outside the grid
        if x_grid < 0 or x_grid >= grid_W or y_grid < 0 or y_grid >= grid_H:
            print('out of bounds:', x_grid, grid_W, y_grid, grid_H)
            return heatmap, offsets, mask

        # Set heatmap value
        heatmap[0, y_grid, x_grid] = 1

        # Compute offsets correctly
        dx = (scaled_label[0] * grid_W) - x_grid  # X-offset
        dy = (scaled_label[1] * grid_H) - y_grid  # Y-offset
        offsets[0, y_grid, x_grid] = dx  # Channel 0: X-offset
        offsets[1, y_grid, x_grid] = dy  # Channel 1: Y-offset

        # Set mask value
        mask[0, y_grid, x_grid] = 1

        return heatmap, offsets, mask
            
          
    @staticmethod
    def max_pooling(image, target_size):
        """
        Apply max pooling to downsample an image.

        Args:
            image (np.ndarray): Input grayscale image as a 2D numpy array.
            target_size (tuple): Target size as (height, width).

        Returns:
            np.ndarray: Downsampled image using max pooling.
        """
        orig_h, orig_w = image.shape
        target_h, target_w = target_size  # Interpret target_size as (height, width)

        # Calculate kernel size
        kernel_h = orig_h // target_h
        kernel_w = orig_w // target_w

        # Trim excess pixels to ensure exact pooling blocks
        trimmed_image = image[:kernel_h * target_h, :kernel_w * target_w]

        # Reshape and apply max pooling
        pooled_image = trimmed_image.reshape(target_h, kernel_h, target_w, kernel_w).max(axis=(1, 3))

        return pooled_image 
        
def visualize_batch(batch_images, batch_labels, batch_masks, batch_offsets, output_dir="batch_visualizations"):
    """
    Visualizes a batch of data, including RGB/jet-colored timesurfaces, masks, and offsets.

    Args:
        batch_images (torch.Tensor): Tensor of shape [batch_size, 4, H, W] (RGB + Grayscale).
        batch_labels (torch.Tensor): Tensor of shape [batch_size, 2] (X, Y coordinates).
        batch_masks (torch.Tensor): Tensor of shape [batch_size, 1, grid_h, grid_w].
        batch_offsets (torch.Tensor): Tensor of shape [batch_size, 2, grid_h, grid_w].
        output_dir (str): Directory to save visualizations.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    batch_images = batch_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [batch_size, H, W, C]
    batch_labels = batch_labels.cpu().numpy()  # Convert to numpy array
    batch_masks = batch_masks.cpu().numpy()  # Convert masks to numpy
    batch_offsets = batch_offsets.cpu().numpy()  # Convert offsets to numpy

    batch_size, H, W, C = batch_images.shape
    C -= 1  # Remove the grayscale channel from count

    for i in range(batch_size):
        sample_dir = os.path.join(output_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Extract grayscale timesurface
        timesurface = (batch_images[i, :, :, 0] * 255).astype(np.uint8)
        timesurface = np.ascontiguousarray(timesurface, dtype=np.uint8)

        # Convert grayscale to RGB using Jet colormap
        timesurface_color = cv2.applyColorMap(timesurface, cv2.COLORMAP_JET)

        # Draw the scaled label on the timesurface
        y, x = int(batch_labels[i, 0] * H), int(batch_labels[i, 1] * W)
        cv2.circle(timesurface_color, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        # Save the timesurface visualization
        timesurface_path = os.path.join(sample_dir, "timesurface.png")
        cv2.imwrite(timesurface_path, timesurface_color)

        # Save the mask visualization
        mask = (batch_masks[i, 0] * 255).astype(np.uint8)  # Assume single-channel mask
        mask_path = os.path.join(sample_dir, "mask.png")
        cv2.imwrite(mask_path, mask)

        # Save the offset visualization
        offsets = batch_offsets[i]

        dx = (offsets[0] * 255).astype(np.uint8)  # Normalize dx to [0, 255]
        dy = (offsets[1] * 255).astype(np.uint8)  # Normalize dy to [0, 255]
        offset_image = np.zeros((batch_masks.shape[2], batch_masks.shape[3], 3), dtype=np.uint8)
        offset_image[..., 0] = dx  # Red channel for dx
        offset_image[..., 1] = dy  # Green channel for dy
        offset_path = os.path.join(sample_dir, "offsets.png")
        cv2.imwrite(offset_path, offset_image)

        print(f"Saved visualizations for sample {i} in {sample_dir}")

    print(f"Batch visualizations saved in {output_dir}")



if __name__ == '__main__':
    # Dataset and DataLoader
    dataset_dir = '/home/nolan4/projects/pfizerDRL/dataset'
    scene_list = ['scene1']
    dataset = drlpDataset(dataset_dir, scene_list, image_size=(1080, 1920), grid_size=(34, 60), augment=True)

    print(f"Total dataset size: {dataset.__len__()} samples.")

    specific_index = 42  # Change this to select a specific sample

    # Fetch the specific sample
    images, targets = dataset[specific_index]

    # If the dataset returns single items, wrap them in a batch dimension
    images = images.unsqueeze(0)  # Convert (C, H, W) to (1, C, H, W)
    targets = {k: v.unsqueeze(0) for k, v in targets.items()}  # Convert dict of tensors

    print("Images Shape:", images.shape)  # Should be (1, C, H, W)
    print("Heatmap Shape:", targets["heatmap_gt"].shape)  # (1, 1, grid_h, grid_w)
    print("Offsets Shape:", targets["offsets_gt"].shape)  # (1, 2, grid_h, grid_w)
    print("Mask Shape:", targets["mask"].shape)  # (1, 1, grid_h, grid_w)
    print("Scaled Labels:", targets["scaled_label"])

    print('Visualizing the batch...')

    # Visualize the specific sample
    visualize_batch(
        batch_images=images,
        batch_labels=targets["scaled_label"],
        batch_masks=targets["mask"],
        batch_offsets=targets["offsets_gt"],
        output_dir="batch_visualizations"
    )