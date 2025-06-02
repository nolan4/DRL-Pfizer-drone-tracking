import os
import shutil
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pdb
from utils import *

class VisualizationCallback:
    """
    Callback to generate and save visualizations after each epoch with random image selection.

    Args:
        output_dir (str): Directory to save visualizations.
        val_loader (DataLoader): DataLoader for the test/validation set.
        model (torch.nn.Module): Trained model for generating predictions.
        device (torch.device): Device to run inference (e.g., "cpu" or "cuda").
        max_images (int): Maximum number of images to visualize per epoch.
        threshold (float): Confidence threshold for filtering heatmap peaks.
        top_k (int): Number of top predictions to visualize.
    """
    def __init__(self, output_dir, val_loader, model, device, max_images=5, threshold=0.1, top_k=5):
        self.output_dir = output_dir
        self.val_loader = val_loader
        self.model = model
        self.device = device
        self.max_images = max_images
        self.threshold = threshold
        self.top_k = top_k

        # Clear all visualization folders
        self._clear_output_dir()

    def _clear_output_dir(self):
        """
        Deletes the output directory and its contents if it exists.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Deletes the directory and its contents
        os.makedirs(self.output_dir)  # Recreates the empty directory

    def __call__(self, epoch):
        """
        Generate and save visualizations for the specified epoch.

        Args:
            epoch (int): Current epoch number.
        """
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        recreate_directory(epoch_dir)

        self.model.eval()

        # Convert the entire validation set to a list for random sampling
        val_data = list(self.val_loader)

        with torch.no_grad():
            for i in range(self.max_images):
                # Select a random batch and random sample within the batch
                random_batch_idx = random.randint(0, len(val_data) - 1)
                x, y = val_data[random_batch_idx]

                random_sample_idx = random.randint(0, x.size(0) - 1)
                x_sample = x[random_sample_idx].unsqueeze(0).to(self.device)  # Add batch dimension

                preds = self.model(x_sample)

                # Generate predictions for the selected image
                batch_predictions = merge_heads(
                    heatmap=preds['heatmap_pred'],
                    offsets=preds['offsets_pred'],
                    threshold=self.threshold,
                    top_k=self.top_k,
                    input_size=(x_sample.shape[2], x_sample.shape[3])
                )

                # Visualize predictions for the selected image
                visualize_predictions_with_heatmaps(
                    image=x_sample[0],
                    predictions=batch_predictions[0],
                    heatmap=preds['heatmap_pred'][0, 0],
                    output_path=os.path.join(epoch_dir, f"image_{i}.png")
                )

        print(f"Visualizations for epoch {epoch} saved to {epoch_dir}")
