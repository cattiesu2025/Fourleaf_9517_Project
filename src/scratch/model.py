"""Scratch ResNet18 model definition."""

from __future__ import annotations

import torch
from torchvision import models


def build_resnet18_scratch(num_classes: int = 500) -> torch.nn.Module:
    """Build ResNet18 with random initialization and a 500-way classifier."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
