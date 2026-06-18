from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .baseline import build_cifar_resnet18


@dataclass(frozen=True)
class ExitSpec:
    name: str
    channels: int
    flops_ratio: float


class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EarlyExitResNet18(nn.Module):
    exit_specs = [
        ExitSpec("exit_1", 64, 0.22),
        ExitSpec("exit_2", 128, 0.42),
        ExitSpec("exit_3", 256, 0.68),
        ExitSpec("final_exit", 512, 1.00),
    ]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        backbone = build_cifar_resnet18(num_classes=num_classes)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.exit1 = ExitHead(64, num_classes)
        self.exit2 = ExitHead(128, num_classes)
        self.exit3 = ExitHead(256, num_classes)
        self.final_pool = backbone.avgpool
        self.final_fc = backbone.fc

    @property
    def flops_ratios(self) -> list[float]:
        return [spec.flops_ratio for spec in self.exit_specs]

    @property
    def exit_names(self) -> list[str]:
        return [spec.name for spec in self.exit_specs]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        logits, _ = self.forward_with_intermediates(x)
        return logits

    def forward_all_exits(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward(x)

    def forward_with_intermediates(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feature_maps: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []

        x = self.stem(x)
        x = self.layer1(x)
        feature_maps.append(x)
        logits.append(self.exit1(x))

        x = self.layer2(x)
        feature_maps.append(x)
        logits.append(self.exit2(x))

        x = self.layer3(x)
        feature_maps.append(x)
        logits.append(self.exit3(x))

        x = self.layer4(x)
        feature_maps.append(x)
        pooled = self.final_pool(x)
        flattened = torch.flatten(pooled, 1)
        logits.append(self.final_fc(flattened))

        return logits, feature_maps
