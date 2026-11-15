"""
Vision Transformer (ViT) implementation: ViT-B/16.
Self-attention transformer architecture for vision representation.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, vit_b_16

from .base_model import BaseClassifier


class ViTClassifier(BaseClassifier):
    """Vision Transformer ViT-B/16 classifier with frozen backbone."""

    def __init__(self, device: str = "cpu", num_classes: int = 2):
        super().__init__(name="ViT-B/16", device=device)
        self.num_classes = num_classes
        self.embedding_dim = 768
        self.build_model()
        if self.model is not None:
            self.model.to(self.device)

    def build_model(self) -> nn.Module:
        weights = ViT_B_16_Weights.DEFAULT
        base = vit_b_16(weights=weights)
        for param in base.parameters():
            param.requires_grad = False

        in_features = base.heads.head.in_features
        base.heads.head = nn.Sequential(
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
                # Reshape and project patches
                x = self.model._process_input(images)
                n = x.shape[0]

                batch_class_token = self.model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.model.encoder(x)

                # Class token representation
                cls_token = x[:, 0]
                cls_token = nn.functional.normalize(cls_token, dim=-1)
                features.append(cls_token.cpu().numpy())
        return np.concatenate(features, axis=0)
