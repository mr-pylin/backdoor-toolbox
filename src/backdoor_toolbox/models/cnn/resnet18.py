import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    """
    A ResNet18-based neural network model with customizable input features, number of classes, and optional pretrained weights.

    Attributes:
        model (nn.Module): The ResNet18 model with customized input and output layers.
    """

    def __init__(self, in_features: int, num_classes: int, weights: ResNet18_Weights | str | None, device: str, verbose: bool):
        """
        Initializes the ResNet18 model.

        Args:
            in_features (int): Number of input channels.
            num_classes (int): Number of output classes.
            weights (ResNet18_Weights | str | None): Specifies pretrained weights.
                - If `ResNet18_Weights`, loads default pretrained weights.
                - If a `.pth` file path is provided, loads weights from the file.
                - If `None`, initializes the model without pretrained weights.
            device (torch.device | str): Device to move the model to ('cpu' or 'cuda').
            verbose (bool): Whether to print logs (e.g., when weights are loaded).

        Raises:
            RuntimeError: If the weights file path is invalid or loading fails.
        """
        super().__init__()
        if isinstance(weights, str) and weights.endswith((".pth", ".pt")):
            self.model = resnet18(weights=None)
            self.model.load_state_dict(torch.load(weights, weights_only=True))
            if verbose:
                print(f"loaded weights from {weights}")
        else:
            self.model = resnet18(weights=weights)

        # modify the model to fit the desired dataset
        self.model.conv1 = nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # move model to device
        self.model = self.model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)


if __name__ == "__main__":
    model_1 = ResNet18(in_features=1, num_classes=10, weights=ResNet18_Weights.IMAGENET1K_V1, device="cpu", verbose=True)
    model_2 = ResNet18(in_features=1, num_classes=10, weights=None, device="cpu", verbose=True)
    model_3 = ResNet18(in_features=1, num_classes=10, weights="temp.pth", device="cpu", verbose=True)
