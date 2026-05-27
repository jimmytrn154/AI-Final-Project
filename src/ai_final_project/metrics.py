from __future__ import annotations

from typing import Iterable

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def average_accuracy(logits_list: Iterable[torch.Tensor], labels: torch.Tensor) -> list[float]:
    return [accuracy_from_logits(logits, labels) for logits in logits_list]


def softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.max(dim=1).values


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    return -(probabilities * log_probabilities).sum(dim=1)


def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]
