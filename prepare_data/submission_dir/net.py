import torch
import torch.nn as nn
from torchvision import models


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


class CustomResNet18(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, log_shapes=False):
        """
        Custom ResNet-18 with separate heads for keypoint detection and offsets.
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of keypoint classes for heatmap prediction.
            log_shapes (bool): Whether to log the shapes of tensors.
        """
        super(CustomResNet18, self).__init__()
        self.log_shapes = log_shapes

        # Load the ResNet-18 backbone
        self.resnet = models.resnet18()

        # Modify the first convolutional layer to accept the custom number of input channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias
        )

        # Remove the final fully connected layer and global pooling
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Add separate heads
        self.keypoint_head = KeypointHead(512, num_classes)
        self.offset_head = OffsetHead(512)
        self.regression_head = RegressionHead(512, 2)

        self.initialize_weights()

    def forward(self, x):
        if self.log_shapes:
            print(f"Input shape: {x.shape}")

        # Extract features using ResNet backbone
        for i, layer in enumerate(self.resnet):
            x = layer(x)
            if self.log_shapes:
                print(f"ResNet Layer {i} - Output Shape: {x.shape}")

        if self.log_shapes:
            print(f"Features shape: {x.shape}")

        # Compute outputs for separate heads
        heatmap = self.keypoint_head(x)
        offsets = self.offset_head(x)
        coords = self.regression_head(x)

        if self.log_shapes:
            print(f"Heatmap shape: {heatmap.shape}")
            print(f"Offsets shape: {offsets.shape}")
            print(f"Offsets shape: {coords.shape}")

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Initialize the custom ResNet-18 model
    model = CustomResNet18(in_channels=4, num_classes=1, log_shapes=True)
    model.eval()  # Set model to evaluation mode
    
    # Move the model to GPU
    model = model.to('cuda')
    # model = model.to('cpu')
    
    # Create a test input tensor and move it to GPU
    test_input = torch.randn(1, 4, 960, 540).to('cuda')  # Batch size = 1, in_channels = 4, image size = 960x540
    # test_input = torch.randn(1, 4, 960, 540).to('cpu')  # Batch size = 1, in_channels = 4, image size = 960x540
    
    # Forward pass
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(test_input)

    print("Heatmap Output Shape:", output["heatmap"].shape)  # Expected: [batch, num_classes, h, w]
    print("Offsets Output Shape:", output["offsets"].shape)  # Expected: [batch, 2, h, w]
