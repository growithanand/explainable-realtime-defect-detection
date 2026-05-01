import torch
from torch import nn
from torchvision import models


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 classifier for normal/defective classification."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Load a trained ResNet18 model from a checkpoint."""
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
