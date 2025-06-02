import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from dataloaders.dataloaderB import drlpDataset  # Replace with your dataset class
from dataloaders.dataloader_Ts import drlpDataset  # Replace with your dataset class
from nets.resnet18 import CustomResNet18
from nets.mobnetV2 import CustomMobileNetV2
from loss import CenterNetLoss
from callbacks import VisualizationCallback  # Import the callback
import pdb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig


def fuse_resnet18(model):
    # ResNet18 typically has Conv-BN-ReLU sequences in its blocks
    layers_to_fuse = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers_to_fuse.append(name)
    
    # Fuse layers: Conv -> BN -> ReLU where applicable
    for layer_name in layers_to_fuse:
        model = torch.quantization.fuse_modules(model, [[layer_name + ".conv1", layer_name + ".bn1", layer_name + ".relu"]], inplace=True)
        # Repeat for other layers (conv2, bn2, relu etc.) if they exist
        # Example:
        # model = torch.quantization.fuse_modules(model, [[layer_name + ".conv2", layer_name + ".bn2", layer_name + ".relu2"]], inplace=True)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0

    with tqdm(dataloader, unit="batch", desc=f"Training Epoch {epoch+1}") as t_epoch:
        for x, y in t_epoch:
            x = x.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds['heatmap_pred'], 
                             preds['offsets_pred'], 
                             y['heatmap_gt'].to(device), 
                             y['offsets_gt'].to(device), 
                             y['mask'].to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            t_epoch.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


def validate_epoch(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    with tqdm(dataloader, unit="batch", desc=f"Validation Epoch {epoch+1}") as v_epoch:
        with torch.no_grad():
            for x, y in v_epoch:
                x = x.to(device)
                preds = model(x)
                loss = criterion(preds['heatmap_pred'], 
                                 preds['offsets_pred'], 
                                 y['heatmap_gt'].to(device), 
                                 y['offsets_gt'].to(device), 
                                 y['mask'].to(device))
                running_loss += loss.item()
                v_epoch.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    writer.add_scalar("Validation/Loss", avg_loss, epoch)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, best_loss, checkpoint_dir):
    """
    Save the checkpoint if the current validation loss is the best so far.
    """
    if loss < best_loss:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, checkpoint_path)
        print(f"New best model saved: {checkpoint_path}")
        return loss
    return best_loss


def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 10
    lr = 5e-7
    image_size = (1080, 1920) # resolution of network input
    grid_size = (135, 240) # resolution of backbone output
    checkpoint_dir = "checkpoints"

    # Dataset setup
    dataset_dir = "/path/to/pfizerDRL/dataset"
    train_dataset = drlpDataset(
        dataset_dir=dataset_dir,
        scene_list=["scene2", "scene3", "scene4", "scene5", "scene6", "scene7", "scene8", "scene9", "scene10", "scene11"],
        image_size=image_size,
        grid_size=grid_size,
        augment=True,
    )
    val_dataset = drlpDataset(
        dataset_dir=dataset_dir,
        scene_list=["scene1"],
        image_size=image_size,
        grid_size=grid_size,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet18(in_channels=1, num_classes=1, N_blocks=6, log_shapes=False)

    # Fuse model layers for quantization
    # model = fuse_resnet18(model)

    model = model.to(device)
    criterion = CenterNetLoss(alpha=2.0, beta=0.10)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Prepare for Quantization Aware Training
    model.qconfig = get_default_qat_qconfig('fbgemm')  # Or 'qnnpack' for mobile devices
    prepare_qat(model, inplace=True)

    # TensorBoard logging
    writer = SummaryWriter("tb_logs/timesurf_centernet")

    # Visualization Callback
    visualization_callback = VisualizationCallback(
        output_dir="viz_after_epoch",
        val_loader=val_loader,  # Validation DataLoader
        model=model,  # Trained model
        device=device,  # Device ("cuda" or "cpu")
        max_images=1,  # Number of images to visualize per epoch
        threshold=0.5,  # Confidence threshold
        top_k=1  # Number of top predictions to visualize
    )

    # Track the best validation loss
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch, writer)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step()  # Update the learning rate

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Train/Learning Rate", current_lr, epoch)

        # Save the checkpoint if validation loss improves
        best_val_loss = save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, checkpoint_dir)
        
        visualization_callback(epoch + 1)

    writer.close()

    # After training, convert the model to a quantized model
    model.to('cpu')
    model.eval()
    quantized_model = convert(model, inplace=False)

    # Save the quantized model
    torch.save(quantized_model.state_dict(), "quantized_model.pth")


if __name__ == "__main__":
    main()
