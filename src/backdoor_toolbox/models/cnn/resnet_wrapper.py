import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


class CustomResNet(nn.Module):
    """
    A general ResNet-based neural network model with customizable input features, number of classes, and optional pretrained weights.

    Attributes:
        model (nn.Module): The selected ResNet model with customized input and output layers.
    """

    RESNET_MODELS = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    }

    def __init__(
        self,
        arch: str,
        in_features: int,
        num_classes: int,
        weights: str | None,
        device: str,
        verbose: bool,
    ):
        """
        Initializes the ResNet model.

        Args:
            model_name (str): Name of the ResNet model to use ('resnet18', 'resnet34', etc.).
            in_features (int): Number of input channels.
            num_classes (int): Number of output classes.
            weights (str | None): Specifies pretrained weights.
                - If a predefined weight object is passed, initializes with those weights.
                - If a `.pth` file path is provided, loads weights from the file.
                - If `None`, initializes the model without pretrained weights.
            device (str): Device to move the model to ('cpu' or 'cuda').
            verbose (bool): Whether to print logs (e.g., when weights are loaded).

        Raises:
            ValueError: If the model name is not in the supported ResNet variants.
        """
        super().__init__()

        if arch not in self.RESNET_MODELS:
            raise ValueError(f"Unsupported model_name '{arch}'. Choose from {list(self.RESNET_MODELS.keys())}.")

        model_fn, model_w = self.RESNET_MODELS[arch]

        # load the model
        if isinstance(weights, str) and weights.endswith((".pth", ".pt")):
            self.model = model_fn(weights=None)
            self.model.load_state_dict(torch.load(weights, weights_only=True))
            if verbose:
                print(f"Loaded pretrained weights: {weights}")
        elif weights is None:
            self.model = model_fn(weights=weights)
            if verbose:
                print(f"No pretrained weights selected.")
        else:
            self.model = model_fn(weights=model_w)
            if verbose:
                print(f"Loaded pretrained weights: {weights}")

        # Modify the model to fit the desired dataset
        self.model.conv1 = nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Move model to device
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
    model_1 = CustomResNet(
        arch="resnet18",
        in_features=1,
        num_classes=10,
        weights=ResNet18_Weights.IMAGENET1K_V1,
        device="cpu",
        verbose=True,
    )
    model_2 = CustomResNet(
        arch="resnet50",
        in_features=3,
        num_classes=1000,
        weights=ResNet50_Weights.IMAGENET1K_V1,
        device="cuda",
        verbose=True,
    )
    model_3 = CustomResNet(
        arch="resnet101",
        in_features=1,
        num_classes=10,
        weights=None,
        device="cpu",
        verbose=True,
    )
    model_4 = CustomResNet(
        arch="resnet152",
        in_features=3,
        num_classes=5,
        weights="temp.pth",
        device="cpu",
        verbose=True,
    )

    print(model_1)
    print(model_2)
    print(model_3)
    print(model_4)
