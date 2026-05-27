from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import BASE_DEFAULTS, load_json, merge_defaults

try:
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover
    raise SystemExit("matplotlib is required to generate plots.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots from experiment outputs.")
    parser.add_argument("--config", default="configs/baseline.json")
    return parser.parse_args()


def load_rows(comparison_path: Path) -> list[dict[str, str]]:
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def save_scatter(rows: list[dict[str, str]], x_key: str, y_key: str, output_path: Path, title: str, x_label: str) -> None:
    xs = []
    ys = []
    labels = []
    for row in rows:
        x_value = numeric(row.get(x_key))
        y_value = numeric(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
        labels.append(row["method"])

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys)
    for x_value, y_value, label in zip(xs, ys, labels):
        plt.annotate(label, (x_value, y_value), fontsize=8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_bar(rows: list[dict[str, str]], metric_key: str, output_path: Path, title: str, y_label: str) -> None:
    labels = []
    values = []
    for row in rows:
        value = numeric(row.get(metric_key))
        if value is None:
            continue
        labels.append(row["method"])
        values.append(value)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), BASE_DEFAULTS)
    metrics_dir = PROJECT_ROOT / config["outputs"]["metrics_dir"]
    plots_dir = PROJECT_ROOT / config["outputs"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = load_rows(metrics_dir / "comparison.csv")
    save_scatter(
        comparison_rows,
        "flops_reduction",
        "accuracy",
        plots_dir / "accuracy_vs_flops.png",
        "Accuracy vs FLOPs Reduction",
        "FLOPs Reduction",
    )
    save_scatter(
        comparison_rows,
        "latency_reduction",
        "accuracy",
        plots_dir / "accuracy_vs_latency.png",
        "Accuracy vs Latency Reduction",
        "Latency Reduction",
    )
    save_bar(comparison_rows, "co2_reduction", plots_dir / "co2_reduction.png", "CO2 Reduction", "CO2 Reduction")
    save_bar(comparison_rows, "accuracy", plots_dir / "accuracy_comparison.png", "Accuracy Comparison", "Accuracy")


if __name__ == "__main__":
    main()
