from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable


def confusion_matrix_from_records(records: list[dict[str, Any]], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for record in records:
        matrix[int(record["label"])][int(record["prediction"])] += 1
    return matrix


def classification_metrics_from_records(records: list[dict[str, Any]], num_classes: int) -> dict[str, Any]:
    matrix = confusion_matrix_from_records(records, num_classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(num_classes))

    per_class = []
    f1_values = []
    precision_values = []
    recall_values = []
    weighted_f1_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0

    for class_index in range(num_classes):
        true_positive = matrix[class_index][class_index]
        support = sum(matrix[class_index])
        predicted = sum(matrix[row_index][class_index] for row_index in range(num_classes))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = true_positive / support if support else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support
        per_class.append(
            {
                "class_index": class_index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": mean(precision_values) if precision_values else 0.0,
        "macro_recall": mean(recall_values) if recall_values else 0.0,
        "macro_f1": mean(f1_values) if f1_values else 0.0,
        "weighted_precision": weighted_precision_sum / total if total else 0.0,
        "weighted_recall": weighted_recall_sum / total if total else 0.0,
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def summarize_policy_records(records: list[dict[str, Any]], num_classes: int) -> dict[str, float]:
    classification = classification_metrics_from_records(records, num_classes)
    average_flops_ratio = mean(float(record["flops_ratio"]) for record in records) if records else 0.0
    avg_exit = mean(float(record["exit_index"]) for record in records) if records else 0.0
    latency_values = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    return {
        "accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "weighted_f1": classification["weighted_f1"],
        "flops_reduction": 1.0 - average_flops_ratio,
        "avg_exit": avg_exit,
        "latency_ms": mean(latency_values) if latency_values else math.nan,
    }


def bootstrap_statistics(
    records: list[dict[str, Any]],
    num_classes: int,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if not records:
        return {}

    rng = random.Random(seed)
    metric_values: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "flops_reduction": [],
        "avg_exit": [],
    }

    if any(record.get("latency_ms") is not None for record in records):
        metric_values["latency_ms"] = []

    for _ in range(num_bootstrap):
        sample = [records[rng.randrange(len(records))] for _ in range(len(records))]
        summary = summarize_policy_records(sample, num_classes)
        for key in metric_values:
            value = summary.get(key)
            if value is not None and not math.isnan(value):
                metric_values[key].append(value)

    return {key: summarize_distribution(values) for key, values in metric_values.items() if values}


def summarize_distribution(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    lower_index = int(0.025 * (len(sorted_values) - 1))
    upper_index = int(0.975 * (len(sorted_values) - 1))
    return {
        "mean": mean(sorted_values),
        "std": pstdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        "ci95_low": sorted_values[lower_index],
        "ci95_high": sorted_values[upper_index],
    }


def exit_distribution_from_records(records: list[dict[str, Any]], exit_names: list[str]) -> dict[str, float]:
    total = max(len(records), 1)
    distribution = {name: 0.0 for name in exit_names}
    for record in records:
        exit_index = int(record["exit_index"]) - 1
        distribution[exit_names[exit_index]] += 1.0 / total
    return distribution


def per_exit_analysis(records: list[dict[str, Any]], exit_names: list[str]) -> dict[str, dict[str, float]]:
    analysis = {}
    for exit_index, exit_name in enumerate(exit_names, start=1):
        exit_records = [record for record in records if int(record["exit_index"]) == exit_index]
        count = len(exit_records)
        correct = sum(1 for record in exit_records if record["correct"])
        confidences = [float(record["confidence"]) for record in exit_records]
        analysis[exit_name] = {
            "count": count,
            "fraction": count / max(len(records), 1),
            "accuracy": correct / count if count else 0.0,
            "average_confidence": mean(confidences) if confidences else 0.0,
        }
    return analysis


def add_latency_to_records(records: list[dict[str, Any]], latency_ms: float | None) -> list[dict[str, Any]]:
    if latency_ms is None:
        return records
    return [{**record, "latency_ms": latency_ms} for record in records]


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return destination


def append_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_exists = destination.exists()
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return destination


def select_best_result(results: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[float, float]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(results, key=key)
