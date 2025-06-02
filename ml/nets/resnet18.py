import torch
import torch.nn as nn
from torchvision import models


class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_keypoints=2):
        super(RegressionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # Flatten to (batch_size, in_channels)
            nn.Linear(in_channels, 128),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(128, num_keypoints)  # Predict x, y coordinates
        )

    def forward(self, x):
        return self.layers(x)


class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(KeypointHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),  # Heatmap output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class OffsetHead(nn.Module):
    def __init__(self, in_channels):
        super(OffsetHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1)  # Offsets (dx, dy)
        )

    def forward(self, x):
        return self.layers(x)


class CustomResNet18(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, N_blocks=12, log_shapes=False):
        super(CustomResNet18, self).__init__()
        self.log_shapes = log_shapes

        # Load the ResNet-18 backbone
        resnet = models.resnet18()

        # Modify the first convolutional layer to accept custom input channels
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias
        )

        # Remove the final fully connected layer and global pooling
        resnet_layers = list(resnet.children())[:-2]

        # Slice the backbone up to N_blocks or the full network
        if N_blocks > 0 and N_blocks < len(resnet_layers):
            self.backbone = nn.Sequential(*resnet_layers[:N_blocks])
        else:
            self.backbone = nn.Sequential(*resnet_layers)

        # Determine the output channels dynamically
        dummy_input = torch.randn(1, in_channels, 1080, 1920)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        last_channel = dummy_output.shape[1]

        # Add separate heads
        self.keypoint_head = KeypointHead(last_channel, num_classes)
        self.offset_head = OffsetHead(last_channel)
        self.regression_head = RegressionHead(last_channel, 2)

        self.initialize_weights()

    def forward(self, x):
        if self.log_shapes:
            print(f"Input shape: {x.shape}")

        # Extract features using the backbone
        x = self.backbone(x)
        if self.log_shapes:
            print(f"Backbone Output shape: {x.shape}")

        # Compute outputs for separate heads
        heatmap = self.keypoint_head(x)
        offsets = self.offset_head(x)
        coords = self.regression_head(x)

        if self.log_shapes:
            print(f"Heatmap shape: {heatmap.shape}")
            print(f"Offsets shape: {offsets.shape}")
            print(f"Coords shape: {coords.shape}")

        return {
            "heatmap_pred": heatmap,
            "offsets_pred": offsets,
            "coords_pred": coords,
        }

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the custom ResNet-18 model
    model = CustomResNet18(in_channels=1, num_classes=1, N_blocks=6, log_shapes=True)
    model = model.to(device)
    model.eval()

    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # Test the model
    test_input = torch.randn(1, 1, 1080, 1920).to(device)
    with torch.no_grad():
        output = model(test_input)

    print("Heatmap Output Shape:", output["heatmap_pred"].shape)  # Expected: [batch, num_classes, h, w]
    print("Offsets Output Shape:", output["offsets_pred"].shape)  # Expected: [batch, 2, h, w]
    print("Coords Output Shape:", output["coords_pred"].shape)  # Expected: [batch, 2]