import os
import shutil
import torch
# from torch.utils.data import DataLoader
# from dataloaders.dataloader_Ts import drlpDataset
from net import CustomResNet18
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_model(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load a saved model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model architecture to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into. Defaults to None.
        device (str): Device to map the model to. Defaults to "cpu".

    Returns:
        dict: Loaded checkpoint data.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    return checkpoint


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
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Normalize and resize the heatmap
    heatmap_np = heatmap.cpu().numpy()
    heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
    heatmap_resized = cv2.resize(heatmap_normalized, (image_np.shape[1], image_np.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

    # Draw predictions on the overlay
    for x, y, conf in predictions:
        cv2.circle(overlay, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(overlay, f"{conf:.2f}", (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Plot the images
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Resized Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay (Image + Heatmap + Predictions)")
    plt.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print

def evaluate(dataset_dir, scene_list, checkpoint_path, save_preds=False):

    image_size = (1080, 1920)
    grid_size = (34, 60)
    threshold = 0.5
    top_k = 1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CustomResNet18(in_channels=1, num_classes=1, log_shapes=False).to(device)

    # Load the saved model checkpoint
    load_model(checkpoint_path, model, device=device)

    # Output directory for visualizations
    output_dir = "evaluation_visualizations"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Output CSV file for predictions
    predictions_csv = "./output.csv"
    with open(predictions_csv, "w") as f_out:
        # Write header
        f_out.write("Ts,x,y\n")

        # Manually load dataset paths as in drlpDataset
        timesurf_paths = []
        for scene in scene_list:
            scene_path = os.path.join(dataset_dir, scene)

            timesurf_paths = sorted(
                [os.path.join(scene_path, "timesurfaces", f) for f in os.listdir(os.path.join(scene_path, "timesurfaces")) if f.endswith('.png')],
                key=lambda x: int(os.path.basename(x).split('_ms_')[1].split('.png')[0])
            )

        # Initialize previous keypoint with the center of the image
        previous_keypoint = (image_size[1] // 2, image_size[0] // 2)  # (center_x, center_y)

        # Iterate over dataset manually
        model.eval()
        with torch.no_grad():
            for idx, timesurf_path in enumerate(timesurf_paths):
                # Extract the `ms` timestamp and format it as a zero-padded integer
                ms_timestamp = int(os.path.basename(timesurf_path).split('_ms_')[1].split('.png')[0])
                formatted_timestamp = f"{ms_timestamp:09d}"

                # Load the timesurface
                timesurface = cv2.imread(timesurf_path, cv2.IMREAD_GRAYSCALE)
                timesurface = torch.from_numpy(timesurface).unsqueeze(0).float() / 255.0

                network_input = timesurface.to(device).unsqueeze(0)
                preds = model(network_input)

                # Merge predictions (heatmap + offsets)
                batch_predictions = merge_heads(
                    heatmap=preds['heatmap_pred'],
                    offsets=preds['offsets_pred'],
                    threshold=threshold,
                    top_k=top_k,
                    input_size=image_size,
                )

                # Handle no detected keypoints
                if not batch_predictions[0]:
                    # No keypoints detected, use the previous keypoint
                    x, y = previous_keypoint
                else:
                    # Use the first detected keypoint and update the previous keypoint
                    x, y, _ = batch_predictions[0][0]
                    previous_keypoint = (x, y)

                # Write predictions to CSV file
                f_out.write(f"{formatted_timestamp},{x},{y}\n")
                f_out.flush()

                # Visualize predictions
                visualize_predictions_with_heatmaps(
                    image=network_input[0],  # Single image
                    predictions=[(x, y, 1.0)],  # Use the current prediction or fallback
                    heatmap=preds['heatmap_pred'][0, 0],  # Heatmap for the first class
                    output_path=os.path.join(output_dir, f"image_{idx}.png"),
                )

                print(f"{idx + 1}/{len(timesurf_paths)} inferences complete")

    print(f"Predictions saved to {predictions_csv}")
    print(f"Visualizations for all test images saved in {output_dir}")

if __name__ == "__main__":

    checkpoint_path = "/path/to/pfizerDRL/ml/checkpoints/epoch_29_loss_18.7249.pth"
    dataset_dir = "/path/to/pfizerDRL/prepare_data/gametime/test-dataset"
    scene_list = ["test_scene"]
    evaluate(dataset_dir, scene_list, checkpoint_path)