from __future__ import annotations

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18Scratch(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_channels: int = 64

        self.conv1: nn.Conv2d = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(self.in_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1: nn.Sequential = self._make_layer(64, blocks=2, stride=1)
        self.layer2: nn.Sequential = self._make_layer(128, blocks=2, stride=2)
        self.layer3: nn.Sequential = self._make_layer(256, blocks=2, stride=2)
        self.layer4: nn.Sequential = self._make_layer(512, blocks=2, stride=2)

        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None

        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers: list[nn.Module] = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
