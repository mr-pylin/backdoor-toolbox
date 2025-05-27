from pathlib import Path

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
from torchvision.models.resnet import BasicBlock, ResNet


class CustomResNet(ResNet):
    RESNET_MODELS = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, 256),
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, 256),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, 1024),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 1024),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2, 1024),
    }

    def __init__(
        self,
        arch: str,
        in_channels: int,
        num_classes: int,
        weights: str | None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        if arch not in self.RESNET_MODELS:
            raise ValueError(f"Unsupported arch '{arch}'. Choose from {list(self.RESNET_MODELS)}")

        model_fn, default_w, layer3_out = self.RESNET_MODELS[arch]

        # Initialize base ResNet with or without pretrained weights
        if weights is None:
            model = model_fn(weights=None)
        elif isinstance(weights, (str, Path)) and weights.endswith((".pt", ".pth")):
            model = model_fn(weights=None)  # for compatibility with our checkpoint
        else:
            model = model_fn(weights=default_w)
            if verbose:
                print(f"Loaded default pretrained weights for {arch}")

        # Init base class
        super().__init__(BasicBlock, [len(b) for b in [model.layer1, model.layer2, model.layer3, model.layer4]])

        # Copy weights from base model
        self.load_state_dict(model.state_dict())

        # Adjust for small images
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        self.layer4 = nn.Identity()
        self.fc = nn.Linear(layer3_out, num_classes)

        # Load checkpoint if path is given
        if isinstance(weights, (str, Path)) and weights.endswith((".pt", ".pth")):
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict)
            if verbose:
                print(f"Loaded weights from {weights}")
        elif weights is None and verbose:
            print("No pretrained weights")

        self.to(device)


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
