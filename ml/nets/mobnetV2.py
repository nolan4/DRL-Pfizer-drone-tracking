import torch
import torch.nn as nn
from torchvision import models
import pdb


class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_keypoints=2):
        """
        Regression head for directly regressing keypoint coordinates.
        Args:
            in_channels (int): Number of input channels from the backbone.
            num_keypoints (int): Number of keypoints to regress (default is 2 for x, y coordinates).
        """
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
        """
        Keypoint head for heatmap prediction.
        Args:
            in_channels (int): Number of input channels from the backbone.
            num_classes (int): Number of classes for keypoint heatmaps.
        """
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
        """
        Offset head for offset regression.
        Args:
            in_channels (int): Number of input channels from the backbone.
        """
        super(OffsetHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1)  # Offsets (dx, dy)
        )

    def forward(self, x):
        return self.layers(x)


class CustomMobileNetV2(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, N_blocks=-1, log_shapes=False):
        """
        Custom MobileNetV2 with separate heads for keypoint detection and offsets.
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of keypoint classes for heatmap prediction.
            log_shapes (bool): Whether to log the shapes of tensors.
        """
        super(CustomMobileNetV2, self).__init__()
        self.log_shapes = log_shapes

        # Load the MobileNetV2 backbone
        mobilenet = models.mobilenet_v2(pretrained=False)

        # Modify the first convolutional layer to accept custom input channels
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mobilenet.features[0][0].out_channels,
            kernel_size=mobilenet.features[0][0].kernel_size,
            stride=mobilenet.features[0][0].stride,
            padding=mobilenet.features[0][0].padding,
            bias=mobilenet.features[0][0].bias
        )


        if N_blocks > 0 and N_blocks < 18:
            # Extract features (up to the Nth convolutional block)
            self.features = nn.Sequential(*mobilenet.features[:N_blocks])
        else:
            # Extract features (up to the last convolutional block)
            self.features = mobilenet.features


        # Determine the number of output channels from the backbone
        # Extract features and determine output channels
        if N_blocks > 0 and N_blocks < len(mobilenet.features):
            self.features = nn.Sequential(*mobilenet.features[:N_blocks])
        else:
            self.features = mobilenet.features

        # Get the output channel count of the sliced backbone
        dummy_input = torch.randn(1, in_channels, 1080, 1920)  # Dummy input to infer output shape
        with torch.no_grad():
            out = self.features(dummy_input)
        last_channel = out.shape[1]

        # pdb.set_trace()

        # Add separate heads
        self.keypoint_head = KeypointHead(last_channel, num_classes)
        self.offset_head = OffsetHead(last_channel)
        self.regression_head = RegressionHead(last_channel, 2)

        self.initialize_weights()

    def forward(self, x):
        if self.log_shapes:
            print(f"Input shape: {x.shape}")

        # Extract features using MobileNetV2 backbone
        x = self.features(x)
        if self.log_shapes:
            print(f"Features shape: {x.shape}")

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
        """
        Initialize the weights of the model using specified initialization methods.
        """
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

    # Initialize the custom MobileNetV2 model
    model = CustomMobileNetV2(in_channels=1, num_classes=1, N_blocks=13, log_shapes=True)
    model.eval()  # Set model to evaluation mode
    
    model = model.to(device)
    
    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # Test input
    test_input = torch.randn(1, 1, 1080, 1920).to(device)  # Batch size = 1, in_channels = 1, image size = 1080x1920
    
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(test_input)

    print("Heatmap Output Shape:", output["heatmap_pred"].shape)  # Expected: [batch, num_classes, h, w]
    print("Offsets Output Shape:", output["offsets_pred"].shape)  # Expected: [batch, 2, h, w]