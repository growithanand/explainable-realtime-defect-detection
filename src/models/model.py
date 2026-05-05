import torch.nn as nn
from torchvision import models


def create_resnet18_binary_model(pretrained: bool = True):
    """
    Creates a ResNet18 model for binary classification.

    Classes:
        0 = normal
        1 = defective
    """

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # ResNet18 originally outputs 1000 classes.
    # We replace the final layer so it outputs 2 classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    return model