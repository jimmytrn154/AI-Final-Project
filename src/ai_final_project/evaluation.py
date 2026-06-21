from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from .metrics import accuracy_from_logits, softmax_confidence
from .profiling import benchmark_dataloader_latency


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def _records_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    exit_indices: torch.Tensor,
    flops_ratios: list[float],
    method: str,
) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=1)
    confidences = softmax_confidence(logits)
    records = []
    for sample_index in range(labels.size(0)):
        exit_index = int(exit_indices[sample_index].item())
        prediction = int(predictions[sample_index].item())
        label = int(labels[sample_index].item())
        records.append(
            {
                "method": method,
                "label": label,
                "prediction": prediction,
                "exit_index": exit_index + 1,
                "confidence": float(confidences[sample_index].item()),
                "flops_ratio": float(flops_ratios[exit_index]),
                "correct": prediction == label,
            }
        )
    return records


def collect_classifier_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = "full_resnet18",
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            exit_indices = torch.zeros(labels.size(0), dtype=torch.long, device=device)
            records.extend(_records_from_logits(logits, labels, exit_indices, [1.0], method))
    return records


def collect_early_exit_final_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = "early_exit_final",
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    final_index = len(model.exit_names) - 1
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model.forward_all_exits(images)[-1]
            exit_indices = torch.full((labels.size(0),), final_index, dtype=torch.long, device=device)
            records.extend(_records_from_logits(logits, labels, exit_indices, model.flops_ratios, method))
    return records


def evaluate_early_exit_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_weights: list[float],
) -> dict[str, Any]:
    model.eval()
    total_examples = 0
    weighted_loss_sum = 0.0
    correct_counts = [0, 0, 0, 0]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model(images)
            losses = [
                weight * torch.nn.functional.cross_entropy(logits, labels)
                for weight, logits in zip(loss_weights, logits_list)
            ]
            weighted_loss_sum += sum(losses).item() * labels.size(0)
            total_examples += labels.size(0)

            for index, logits in enumerate(logits_list):
                correct_counts[index] += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": weighted_loss_sum / max(total_examples, 1),
        "per_exit_accuracy": [count / max(total_examples, 1) for count in correct_counts],
        "final_accuracy": correct_counts[-1] / max(total_examples, 1),
    }


def decide_exit(
    logits_list: list[torch.Tensor],
    flops_ratios: list[float],
    strategy: str,
    threshold: float,
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits_list[0].shape[0]
    chosen_indices = torch.full((batch_size,), len(logits_list) - 1, dtype=torch.long, device=logits_list[0].device)
    chosen_logits = logits_list[-1]

    for index, logits in enumerate(logits_list[:-1]):
        confidence = softmax_confidence(logits)
        flops_ratio = flops_ratios[index]
        if strategy == "fixed":
            current_threshold = threshold
        elif strategy == "accuracy_first":
            current_threshold = threshold + alpha * flops_ratio
        elif strategy == "budget_first":
            current_threshold = threshold - alpha * flops_ratio
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        undecided = chosen_indices == len(logits_list) - 1
        exit_mask = undecided & (confidence >= current_threshold)
        chosen_indices = torch.where(exit_mask, torch.full_like(chosen_indices, index), chosen_indices)
        expanded_mask = exit_mask.unsqueeze(1)
        chosen_logits = torch.where(expanded_mask, logits, chosen_logits)

    return chosen_logits, chosen_indices


def evaluate_exit_strategy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    strategy: str,
    threshold: float,
    alpha: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    total_correct = 0
    total_examples = 0
    exit_counter: Counter[int] = Counter()
    flops_used = 0.0
    flops_ratios = model.flops_ratios

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model(images)
            chosen_logits, exit_indices = decide_exit(
                logits_list=logits_list,
                flops_ratios=flops_ratios,
                strategy=strategy,
                threshold=threshold,
                alpha=alpha,
            )
            total_correct += (chosen_logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

            for exit_index in exit_indices.tolist():
                exit_counter[exit_index] += 1
                flops_used += flops_ratios[exit_index]

    avg_exit = sum((index + 1) * count for index, count in exit_counter.items()) / max(total_examples, 1)
    exit_distribution = {
        model.exit_names[index]: exit_counter.get(index, 0) / max(total_examples, 1)
        for index in range(len(model.exit_names))
    }
    average_flops_ratio = flops_used / max(total_examples, 1)

    return {
        "accuracy": total_correct / max(total_examples, 1),
        "avg_exit": avg_exit,
        "average_flops_ratio": average_flops_ratio,
        "flops_reduction": 1.0 - average_flops_ratio,
        "exit_distribution": exit_distribution,
        "threshold": threshold,
        "alpha": alpha,
        "strategy": strategy,
    }


def collect_exit_strategy_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    strategy: str,
    threshold: float,
    alpha: float = 0.0,
    method: str | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    method_name = method or f"{strategy}_threshold"

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model(images)
            chosen_logits, exit_indices = decide_exit(
                logits_list=logits_list,
                flops_ratios=model.flops_ratios,
                strategy=strategy,
                threshold=threshold,
                alpha=alpha,
            )
            records.extend(_records_from_logits(chosen_logits, labels, exit_indices, model.flops_ratios, method_name))
    return records


def collect_controller_records(
    model: torch.nn.Module,
    controller: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = "learned_controller",
) -> list[dict[str, Any]]:
    model.eval()
    controller.eval()
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            chosen_logits, exit_indices = model.forward_with_supervised_controller(images, controller)
            records.extend(_records_from_logits(chosen_logits, labels, exit_indices, model.flops_ratios, method))
    return records


def collect_reinforce_controller_records(
    model: torch.nn.Module,
    controller: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = "reinforce_controller",
) -> list[dict[str, Any]]:
    model.eval()
    controller.eval()
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            chosen_logits, exit_indices = model.forward_with_reinforce_controller(images, controller)
            records.extend(_records_from_logits(chosen_logits, labels, exit_indices, model.flops_ratios, method))
    return records


def benchmark_exit_strategy_latency(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    strategy: str,
    threshold: float,
    alpha: float = 0.0,
    warmup_batches: int = 3,
    timed_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    return benchmark_dataloader_latency(
        lambda images: model.forward_with_policy(images, strategy, threshold, alpha),
        dataloader,
        device,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )


def benchmark_controller_latency(
    model: torch.nn.Module,
    controller: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    timed_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    controller.eval()
    return benchmark_dataloader_latency(
        lambda images: model.forward_with_supervised_controller(images, controller),
        dataloader,
        device,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )


def benchmark_reinforce_controller_latency(
    model: torch.nn.Module,
    controller: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    timed_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    controller.eval()
    return benchmark_dataloader_latency(
        lambda images: model.forward_with_reinforce_controller(images, controller),
        dataloader,
        device,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )
