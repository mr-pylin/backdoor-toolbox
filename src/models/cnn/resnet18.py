from torchvision.models import resnet18
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, weights, in_features: int, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
