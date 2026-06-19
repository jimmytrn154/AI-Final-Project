from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .baseline import build_cifar_resnet18
from ..metrics import softmax_confidence


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

    def forward_until(self, x: torch.Tensor, exit_index: int) -> torch.Tensor:
        if exit_index < 0 or exit_index >= len(self.exit_specs):
            raise ValueError(f"Invalid exit index: {exit_index}")

        x = self.stem(x)
        x = self.layer1(x)
        if exit_index == 0:
            return self.exit1(x)

        x = self.layer2(x)
        if exit_index == 1:
            return self.exit2(x)

        x = self.layer3(x)
        if exit_index == 2:
            return self.exit3(x)

        x = self.layer4(x)
        pooled = self.final_pool(x)
        flattened = torch.flatten(pooled, 1)
        return self.final_fc(flattened)

    def _forward_routed(self, x: torch.Tensor, should_exit_fn) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        num_classes = self.final_fc.out_features
        final_index = len(self.exit_specs) - 1

        chosen_indices = torch.full((batch_size,), final_index, dtype=torch.long, device=device)
        chosen_logits = torch.empty((batch_size, num_classes), dtype=x.dtype, device=device)
        active_indices = torch.arange(batch_size, device=device)

        x = self.stem(x)
        x = self.layer1(x)

        logits = self.exit1(x)
        exit_mask = should_exit_fn(logits, 0)
        if exit_mask.any():
            chosen_indices[active_indices[exit_mask]] = 0
            chosen_logits[active_indices[exit_mask]] = logits[exit_mask]
        continue_mask = ~exit_mask
        if not continue_mask.any():
            return chosen_logits, chosen_indices

        active_indices = active_indices[continue_mask]
        x = self.layer2(x[continue_mask])

        logits = self.exit2(x)
        exit_mask = should_exit_fn(logits, 1)
        if exit_mask.any():
            chosen_indices[active_indices[exit_mask]] = 1
            chosen_logits[active_indices[exit_mask]] = logits[exit_mask]
        continue_mask = ~exit_mask
        if not continue_mask.any():
            return chosen_logits, chosen_indices

        active_indices = active_indices[continue_mask]
        x = self.layer3(x[continue_mask])

        logits = self.exit3(x)
        exit_mask = should_exit_fn(logits, 2)
        if exit_mask.any():
            chosen_indices[active_indices[exit_mask]] = 2
            chosen_logits[active_indices[exit_mask]] = logits[exit_mask]
        continue_mask = ~exit_mask
        if not continue_mask.any():
            return chosen_logits, chosen_indices

        active_indices = active_indices[continue_mask]
        x = self.layer4(x[continue_mask])
        pooled = self.final_pool(x)
        flattened = torch.flatten(pooled, 1)
        final_logits = self.final_fc(flattened)
        chosen_logits[active_indices] = final_logits
        return chosen_logits, chosen_indices

    def forward_with_policy(
        self,
        x: torch.Tensor,
        strategy: str,
        threshold: float,
        alpha: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def should_exit(logits: torch.Tensor, exit_index: int) -> torch.Tensor:
            confidence = softmax_confidence(logits)
            flops_ratio = self.flops_ratios[exit_index]
            if strategy == "fixed":
                current_threshold = threshold
            elif strategy == "accuracy_first":
                current_threshold = threshold + alpha * flops_ratio
            elif strategy == "budget_first":
                current_threshold = threshold - alpha * flops_ratio
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            return confidence >= current_threshold

        return self._forward_routed(x, should_exit)

    def forward_with_supervised_controller(
        self,
        x: torch.Tensor,
        controller: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ..controller import build_controller_features

        def should_exit(logits: torch.Tensor, exit_index: int) -> torch.Tensor:
            features = build_controller_features(logits, exit_index + 1, self.flops_ratios[exit_index]).to(logits.device)
            return controller(features).argmax(dim=1) == 1

        return self._forward_routed(x, should_exit)

    def forward_with_reinforce_controller(
        self,
        x: torch.Tensor,
        controller: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ..controller import build_reinforce_state

        def should_exit(logits: torch.Tensor, exit_index: int) -> torch.Tensor:
            state = build_reinforce_state(
                logits=logits,
                exit_index=exit_index,
                flops_ratio=self.flops_ratios[exit_index],
                num_exits=len(self.exit_specs),
            ).to(logits.device)
            return controller.policy(state).probs.argmax(dim=1) == 1

        return self._forward_routed(x, should_exit)

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
