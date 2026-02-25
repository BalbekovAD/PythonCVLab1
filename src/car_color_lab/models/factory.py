from __future__ import annotations

from typing import cast

from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet

from car_color_lab.constants import ModelName, PRETRAIN_FREEZE_STAGES
from car_color_lab.models.resnet18_scratch import ResNet18Scratch


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc"):
        fc_layer: nn.Module = cast(nn.Module, getattr(model, "fc"))
        if isinstance(fc_layer, nn.Linear):
            in_features: int = fc_layer.in_features
            setattr(model, "fc", nn.Linear(in_features, num_classes))
            return model
    raise ValueError("Model does not expose an fc classifier")


def _freeze_resnet_stages(model: ResNet, freeze_stages: int) -> None:
    if freeze_stages <= 0:
        return

    if freeze_stages >= 1:
        for parameter in model.conv1.parameters():
            parameter.requires_grad = False
        for parameter in model.bn1.parameters():
            parameter.requires_grad = False

    if freeze_stages >= 2:
        for parameter in model.layer1.parameters():
            parameter.requires_grad = False

    if freeze_stages >= 3:
        for parameter in model.layer2.parameters():
            parameter.requires_grad = False

    if freeze_stages >= 4:
        for parameter in model.layer3.parameters():
            parameter.requires_grad = False


def build_model(model_name: ModelName, num_classes: int) -> nn.Module:
    if model_name == "resnet18_scratch":
        return ResNet18Scratch(num_classes=num_classes)

    if model_name == "resnet18_pretrain":
        model18: ResNet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model18 = cast(ResNet, _replace_classifier(model18, num_classes=num_classes))
        _freeze_resnet_stages(model18, PRETRAIN_FREEZE_STAGES[model_name])
        return cast(nn.Module, model18)

    if model_name == "resnet50_pretrain":
        model50: ResNet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model50 = cast(ResNet, _replace_classifier(model50, num_classes=num_classes))
        _freeze_resnet_stages(model50, PRETRAIN_FREEZE_STAGES[model_name])
        return cast(nn.Module, model50)

    raise ValueError(f"Unsupported model name: {model_name}")
