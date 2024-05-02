import timm
import torch.nn as nn


def create_model(architecture: str, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
    img_size = kwargs.get("img_size", 512)
    try:
        model = timm.create_model(architecture, num_classes=num_classes, pretrained=pretrained, img_size=img_size)
    except TypeError:
        model = timm.create_model(architecture, num_classes=num_classes, pretrained=pretrained)

    return model
