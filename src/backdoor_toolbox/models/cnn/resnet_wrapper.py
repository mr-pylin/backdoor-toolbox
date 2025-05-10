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
    A ResNet wrapper for small images that also drops the heaviest stage (layer4)
    to reduce parameters and speed up training.
    - 3×3 stride=1 conv1
    - No initial max-pool
    - Remove layer4 entirely
    """

    RESNET_MODELS = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, 256),
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, 256),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, 1024),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 1024),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2, 1024),
    }
    # The third tuple element is the # output channels of layer3

    def __init__(
        self,
        arch: str,
        in_channels: int,
        num_classes: int,
        weights: str | None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__()

        if arch not in self.RESNET_MODELS:
            raise ValueError(f"Unsupported arch '{arch}'. Choose from {list(self.RESNET_MODELS)}")

        model_fn, default_w, layer3_out = self.RESNET_MODELS[arch]

        # Load pretrained or from file
        if isinstance(weights, str) and weights.endswith((".pth", ".pt")):
            self.model = model_fn(weights=None)
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            if verbose:
                print(f"Loaded weights from {weights}")
        elif weights is None:
            self.model = model_fn(weights=None)
            if verbose:
                print("No pretrained weights")
        else:
            self.model = model_fn(weights=default_w)
            if verbose:
                print(f"Loaded default pretrained weights for {arch}")

        # Replace conv1 for small images
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        # Remove layer4 to slim down the network
        self.model.layer4 = nn.Identity()

        # Replace final FC to match layer3's output channels
        self.model.fc = nn.Linear(layer3_out, num_classes)

        # Move to device
        self.model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    model_1 = CustomResNet(
        arch="resnet18",
        in_channels=1,
        num_classes=10,
        weights=True,
        device="cpu",
        verbose=True,
    )
    model_2 = CustomResNet(
        arch="resnet50",
        in_channels=3,
        num_classes=1000,
        weights=True,
        device="cuda",
        verbose=True,
    )
    model_3 = CustomResNet(
        arch="resnet101",
        in_channels=1,
        num_classes=10,
        weights=None,
        device="cpu",
        verbose=True,
    )
    model_4 = CustomResNet(
        arch="resnet152",
        in_channels=3,
        num_classes=5,
        weights="temp.pth",
        device="cpu",
        verbose=True,
    )

    print(model_1)
    print(model_2)
    print(model_3)
    print(model_4)
