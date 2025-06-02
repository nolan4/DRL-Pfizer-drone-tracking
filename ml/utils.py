import os
import shutil
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

def recreate_directory(path):
    """
    Deletes the directory if it exists and creates a new one.

    Args:
        path (str): Path of the directory to recreate.
    """
    if os.path.exists(path):
        shutil.rmtree(path)  # Deletes the directory and its contents
    os.makedirs(path)  # Recreates the empty directory


def merge_heads(heatmap, offsets, threshold=0.5, top_k=100, input_size=(1080, 1920)):
    """
    Merge heatmap and offset predictions into final coordinates.

    Args:
        heatmap (torch.Tensor): Heatmap output of shape [batch, num_classes, H, W].
        offsets (torch.Tensor): Offset output of shape [batch, 2, H, W].
        threshold (float): Threshold for filtering heatmap peaks.
        top_k (int): Number of top keypoints to retain after non-max suppression.
        input_size (tuple): Input image size as (height, width).

    Returns:
        List[List[Tuple[float, float, float]]]: List of lists of predictions for each batch.
    """
    batch_size, num_classes, H, W = heatmap.shape
    input_h, input_w = input_size
    scale_x = input_w / W  # Width scaling factor
    scale_y = input_h / H  # Height scaling factor

    final_predictions = []
    for b in range(batch_size):
        batch_preds = []
        for c in range(num_classes):
            class_heatmap = heatmap[b, c]
            pooled_heatmap = torch.nn.functional.max_pool2d(
                class_heatmap[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            nms_heatmap = (class_heatmap == pooled_heatmap).float() * class_heatmap
            mask = nms_heatmap > threshold
            y_idxs, x_idxs = torch.nonzero(mask, as_tuple=True)
            scores = nms_heatmap[y_idxs, x_idxs]

            if scores.numel() > 0:
                predictions = torch.stack([x_idxs, y_idxs, scores], dim=1)
                predictions = predictions[torch.argsort(predictions[:, 2], descending=True)]
                predictions = predictions[:top_k]

                for pred in predictions:
                    x_idx, y_idx, score = pred.int()
                    dx, dy = offsets[b, :, y_idx, x_idx]
                    x_abs = (x_idx + dx.item()) * scale_x
                    y_abs = (y_idx + dy.item()) * scale_y

                    # Ensure coordinates are within bounds
                    x_abs = max(0, min(input_w, x_abs))
                    y_abs = max(0, min(input_h, y_abs))

                    batch_preds.append((x_abs, y_abs, score.item()))

        final_predictions.append(batch_preds)

    return final_predictions


def visualize_predictions_with_heatmaps(image, predictions, heatmap, output_path=None):
    """
    Visualize predictions and heatmaps by overlaying them on the input image.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        predictions (list): List of predicted coordinates [(x, y, confidence), ...].
        heatmap (torch.Tensor): Heatmap tensor of shape (H, W).
        output_path (str, optional): Path to save the visualization. Defaults to None.
    """
    # Convert the input image to NumPy format
    image_np = image.permute(1, 2, 0).cpu().numpy()
    if image.max() <= 1:
        image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Normalize and resize the heatmap
    heatmap_np = heatmap.cpu().numpy()
    heatmap_np = cv2.resize(heatmap_np, (image_np.shape[1], image_np.shape[0]))
    if heatmap_np.max() > 1:
        heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
    else:
        heatmap_normalized = heatmap_np    
    
    heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

    # Draw predictions on the overlay
    for x, y, conf in predictions:
        cv2.circle(overlay, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(overlay, f"{conf:.2f}", (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Plot the images
    plt.figure()
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay (Image + Heatmap + Predictions)")
    plt.axis("off")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.close()