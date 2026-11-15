"""
CNN Model Implementations: MobileNetV3-Small, EfficientNetB0, and ResNet18.
All backbones are frozen (feature extractors) with a trained linear probe classification head.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

from .base_model import BaseClassifier


class MobileNetV3SmallClassifier(BaseClassifier):
    """MobileNetV3-Small: Lightweight CNN optimized for low-latency CPU inference."""

    def __init__(self, device: str = "cpu", num_classes: int = 2):
        super().__init__(name="MobileNetV3-Small", device=device)
        self.num_classes = num_classes
        self.embedding_dim = 576
        self.build_model()
        if self.model is not None:
            self.model.to(self.device)

    def build_model(self) -> nn.Module:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        for param in base.parameters():
            param.requires_grad = False

        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, self.num_classes)
        )
        self.model = base
        return self.model

    def extract_features(self, loader: DataLoader) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()
        features = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                x = self.model.features(images)
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = nn.functional.normalize(x, dim=-1)
                features.append(x.cpu().numpy())
        return np.concatenate(features, axis=0)


class EfficientNetB0Classifier(BaseClassifier):
    """EfficientNetB0: Standard baseline using depthwise separable convolutions."""

    def __init__(self, device: str = "cpu", num_classes: int = 2):
        super().__init__(name="EfficientNetB0", device=device)
        self.num_classes = num_classes
        self.embedding_dim = 1280
        self.build_model()
        if self.model is not None:
            self.model.to(self.device)

    def build_model(self) -> nn.Module:
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        for param in base.parameters():
            param.requires_grad = False

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.45),
            nn.Linear(in_features, self.num_classes)
        )
        self.model = base
        return self.model

    def extract_features(self, loader: DataLoader) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()
        features = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                x = self.model.features(images)
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = nn.functional.normalize(x, dim=-1)
                features.append(x.cpu().numpy())
        return np.concatenate(features, axis=0)


class ResNet18Classifier(BaseClassifier):
    """ResNet18: Residual network architecture providing classic convolutional inductive bias."""

    def __init__(self, device: str = "cpu", num_classes: int = 2):
        super().__init__(name="ResNet18", device=device)
        self.num_classes = num_classes
        self.embedding_dim = 512
        self.build_model()
        if self.model is not None:
            self.model.to(self.device)

    def build_model(self) -> nn.Module:
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)
        for param in base.parameters():
            param.requires_grad = False

        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, self.num_classes)
        )
        self.model = base
        return self.model

    def extract_features(self, loader: DataLoader) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()
        features = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                # Forward pass up to avgpool
                x = self.model.conv1(images)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)

                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)

                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = nn.functional.normalize(x, dim=-1)
                features.append(x.cpu().numpy())
        return np.concatenate(features, axis=0)
