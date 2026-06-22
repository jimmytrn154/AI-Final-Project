from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import BASE_DEFAULTS, load_json, merge_defaults, resolve_run_outputs

try:
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover
    raise SystemExit("matplotlib is required to generate ablation plots.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for deeper ablation experiments.")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--run-name", default="2026-06-20-deeper-experiments")
    return parser.parse_args()


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def save_accuracy_vs_threshold(metrics_dir: Path, plots_dir: Path) -> None:
    payload = maybe_load_json(metrics_dir / "threshold_ablation_results.json")
    if not payload:
        return

    thresholds = [item["threshold"] for item in payload["results"]]
    accuracies = [item["test"]["accuracy"] for item in payload["results"]]
    flops_reductions = [item["test"]["flops_reduction"] for item in payload["results"]]

    _, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, accuracies, marker="o", label="Accuracy")
    axis.plot(thresholds, flops_reductions, marker="s", label="FLOPs reduction")
    axis.set_title("Fixed Threshold Ablation")
    axis.set_xlabel("Threshold")
    axis.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_vs_threshold.png")
    plt.close()


def save_accuracy_flops_frontier(metrics_dir: Path, plots_dir: Path) -> None:
    points = []
    for filename, label in (
        ("threshold_ablation_results.json", "fixed"),
        ("dynamic_threshold_ablation_results.json", "dynamic"),
        ("controller_lambda_sweep_results.json", "controller"),
        ("reinforce_lambda_sweep_results.json", "reinforce"),
    ):
        payload = maybe_load_json(metrics_dir / filename)
        if not payload:
            continue
        for item in payload["results"]:
            points.append(
                {
                    "family": label,
                    "method": item["method"],
                    "accuracy": item["test"]["accuracy"],
                    "flops_reduction": item["test"]["flops_reduction"],
                }
            )

    if not points:
        return

    _, axis = plt.subplots(figsize=(8, 5))
    families = sorted({point["family"] for point in points})
    for family in families:
        family_points = [point for point in points if point["family"] == family]
        axis.scatter(
            [point["flops_reduction"] for point in family_points],
            [point["accuracy"] for point in family_points],
            label=family,
        )
    axis.set_title("Accuracy vs FLOPs Reduction Frontier")
    axis.set_xlabel("FLOPs reduction")
    axis.set_ylabel("Accuracy")
    axis.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_flops_frontier.png")
    plt.close()


def save_lambda_tradeoff(metrics_dir: Path, plots_dir: Path, filename: str, output_name: str, title: str) -> None:
    payload = maybe_load_json(metrics_dir / filename)
    if not payload:
        return

    lambdas = [item.get("lambda_cost", item.get("reward_lambda")) for item in payload["results"]]
    accuracies = [item["test"]["accuracy"] for item in payload["results"]]
    flops_reductions = [item["test"]["flops_reduction"] for item in payload["results"]]

    _, axis = plt.subplots(figsize=(8, 5))
    axis.plot(lambdas, accuracies, marker="o", label="Accuracy")
    axis.plot(lambdas, flops_reductions, marker="s", label="FLOPs reduction")
    axis.set_title(title)
    axis.set_xlabel("Lambda")
    axis.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / output_name)
    plt.close()


def save_exit_distribution(metrics_dir: Path, plots_dir: Path) -> None:
    payload = maybe_load_json(metrics_dir / "per_exit_analysis.json")
    if not payload:
        return

    methods = list(payload.keys())
    exit_names = ["exit_1", "exit_2", "exit_3", "final_exit"]
    bottoms = [0.0 for _ in methods]
    _, axis = plt.subplots(figsize=(10, 5))

    for exit_name in exit_names:
        values = [payload[method].get(exit_name, {}).get("fraction", 0.0) for method in methods]
        axis.bar(methods, values, bottom=bottoms, label=exit_name)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    axis.set_title("Exit Distribution Comparison")
    axis.set_ylabel("Fraction")
    axis.legend()
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "exit_distribution_comparison.png")
    plt.close()


def save_bootstrap_intervals(metrics_dir: Path, plots_dir: Path) -> None:
    payload = maybe_load_json(metrics_dir / "bootstrap_statistics.json")
    if not payload:
        return

    methods = list(payload.keys())
    means = [payload[method]["accuracy"]["mean"] for method in methods]
    lower = [mean - payload[method]["accuracy"]["ci95_low"] for mean, method in zip(means, methods)]
    upper = [payload[method]["accuracy"]["ci95_high"] - mean for mean, method in zip(means, methods)]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756", "#B279A2"][: len(methods)]

    _, axis = plt.subplots(figsize=(10, 5))
    axis.bar(methods, means, yerr=[lower, upper], capsize=4, color=colors)
    axis.set_title("Bootstrap Accuracy Confidence Intervals")
    axis.set_ylabel("Accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "bootstrap_confidence_intervals.png")
    plt.close()


def save_confusion_matrices(metrics_dir: Path, plots_dir: Path) -> None:
    payload = maybe_load_json(metrics_dir / "classification_metrics.json")
    if not payload:
        return

    methods = list(payload.keys())[:6]
    if not methods:
        return

    rows = 2 if len(methods) > 3 else 1
    columns = 3 if len(methods) > 1 else 1
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, method in zip(axes_list, methods):
        matrix = payload[method]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(method)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_list[len(methods):]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix_best_methods.png")
    plt.close()


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), BASE_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    metrics_dir = PROJECT_ROOT / config["outputs"]["metrics_dir"]
    plots_dir = PROJECT_ROOT / config["outputs"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_accuracy_vs_threshold(metrics_dir, plots_dir)
    save_accuracy_flops_frontier(metrics_dir, plots_dir)
    save_lambda_tradeoff(
        metrics_dir,
        plots_dir,
        "controller_lambda_sweep_results.json",
        "controller_lambda_tradeoff.png",
        "Learned Controller Lambda Tradeoff",
    )
    save_lambda_tradeoff(
        metrics_dir,
        plots_dir,
        "reinforce_lambda_sweep_results.json",
        "reinforce_lambda_tradeoff.png",
        "REINFORCE Lambda Tradeoff",
    )
    save_exit_distribution(metrics_dir, plots_dir)
    save_bootstrap_intervals(metrics_dir, plots_dir)
    save_confusion_matrices(metrics_dir, plots_dir)


if __name__ == "__main__":
    main()
