from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .metrics import entropy_from_logits, margin_from_logits, softmax_confidence


class ExitController(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_controller_features(logits: torch.Tensor, exit_index: int, flops_ratio: float) -> torch.Tensor:
    confidence = softmax_confidence(logits)
    entropy = entropy_from_logits(logits)
    margin = margin_from_logits(logits)
    exit_column = torch.full_like(confidence, float(exit_index))
    flops_column = torch.full_like(confidence, float(flops_ratio))
    return torch.stack([confidence, entropy, margin, exit_column, flops_column], dim=1)


def choose_target_exit(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    flops_ratios: list[float],
    strategy: str,
    reward_lambda: float,
) -> torch.Tensor:
    correctness = [(logits.argmax(dim=1) == labels).float() for logits in logits_list]

    if strategy == "earliest_correct":
        chosen = torch.full((labels.size(0),), len(logits_list) - 1, dtype=torch.long, device=labels.device)
        for index, correct in enumerate(correctness[:-1]):
            replace_mask = (chosen == len(logits_list) - 1) & (correct > 0)
            chosen = torch.where(replace_mask, torch.full_like(chosen, index), chosen)
        return chosen

    if strategy == "best_reward":
        rewards = torch.stack(
            [correct - reward_lambda * flops_ratios[index] for index, correct in enumerate(correctness)],
            dim=1,
        )
        return rewards.argmax(dim=1)

    raise ValueError(f"Unsupported supervision strategy: {strategy}")


def build_controller_dataset(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    supervision_strategy: str,
    reward_lambda: float,
) -> TensorDataset:
    feature_batches = []
    label_batches = []
    flops_ratios = model.flops_ratios

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model(images)
            target_exits = choose_target_exit(
                logits_list=logits_list,
                labels=labels,
                flops_ratios=flops_ratios,
                strategy=supervision_strategy,
                reward_lambda=reward_lambda,
            )

            for exit_index, logits in enumerate(logits_list[:-1]):
                features = build_controller_features(logits, exit_index + 1, flops_ratios[exit_index])
                targets = (target_exits == exit_index).long()
                valid_mask = target_exits >= exit_index
                feature_batches.append(features[valid_mask].cpu())
                label_batches.append(targets[valid_mask].cpu())

    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return TensorDataset(features, labels)


def evaluate_controller(
    model: torch.nn.Module,
    controller: ExitController,
    dataloader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    controller.eval()
    total_correct = 0
    total_examples = 0
    exit_counts = [0, 0, 0, 0]
    flops_used = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model(images)

            final_logits = logits_list[-1]
            chosen_indices = torch.full((labels.size(0),), len(logits_list) - 1, dtype=torch.long, device=device)
            chosen_logits = final_logits

            for exit_index, logits in enumerate(logits_list[:-1]):
                undecided = chosen_indices == len(logits_list) - 1
                features = build_controller_features(logits, exit_index + 1, model.flops_ratios[exit_index]).to(device)
                decisions = controller(features).argmax(dim=1)
                exit_mask = undecided & (decisions == 1)
                chosen_indices = torch.where(exit_mask, torch.full_like(chosen_indices, exit_index), chosen_indices)
                chosen_logits = torch.where(exit_mask.unsqueeze(1), logits, chosen_logits)

            total_correct += (chosen_logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

            for exit_index in chosen_indices.tolist():
                exit_counts[exit_index] += 1
                flops_used += model.flops_ratios[exit_index]

    return {
        "accuracy": total_correct / max(total_examples, 1),
        "avg_exit": sum((index + 1) * count for index, count in enumerate(exit_counts)) / max(total_examples, 1),
        "average_flops_ratio": flops_used / max(total_examples, 1),
        "flops_reduction": 1.0 - (flops_used / max(total_examples, 1)),
        "exit_distribution": {
            model.exit_names[index]: count / max(total_examples, 1)
            for index, count in enumerate(exit_counts)
        },
    }
