from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class CustomMobileNetV2(nn.Module):
    MOBILENET_CONFIG = {
        "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
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
        super().__init__()

        if arch not in self.MOBILENET_CONFIG:
            raise ValueError(f"Unsupported arch '{arch}'. Choose from {list(self.MOBILENET_CONFIG)}")

        model_fn, default_weights = self.MOBILENET_CONFIG[arch]

        # Load model (pretrained or from checkpoint)
        if weights is None:
            model = model_fn(weights=None)
        elif isinstance(weights, (str, Path)) and weights.endswith((".pt", ".pth")):
            model = model_fn(weights=None)  # For manual checkpoint loading
        else:
            model = model_fn(weights=default_weights)
            if verbose:
                print(f"Loaded default pretrained weights for {arch}")

        # Modify input conv if input channels ≠ 3
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove classifier
        # self.features = nn.Sequential(*list(model.features.children())[:18])  # up to features[17]
        # last_conv_out_channels = list(self.features.children())[-1].conv[-2].out_channels
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(last_conv_out_channels, num_classes)

        self.features = model.features
        last_conv_out_channels = model.classifier[1].in_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_conv_out_channels, num_classes)

        # Load custom weights if provided
        if isinstance(weights, (str, Path)) and weights.endswith((".pt", ".pth")):
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict)
            if verbose:
                print(f"Loaded weights from checkpoint: {weights}")
        elif weights is None and verbose:
            print("No pretrained weights used")

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CustomMobileNetV2(
        arch="mobilenet_v2",
        in_channels=1,
        num_classes=10,
        weights=None,
        device="cpu",
        verbose=True,
    )

    # Print named parameters
    for name, param in model.named_parameters():
        print(f"{name}: {tuple(param.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
