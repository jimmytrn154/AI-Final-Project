from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from .config import save_json


def write_metrics_json(metrics_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    destination = Path(metrics_dir) / filename
    save_json(destination, payload)
    return destination


def bootstrap_run_metrics(
    project_root: str | Path,
    outputs_config: dict[str, Any],
) -> None:
    run_name = outputs_config.get("run_name")
    if not run_name:
        return

    project_root = Path(project_root)
    metrics_dir = project_root / outputs_config["metrics_dir"]
    reference_metrics_dir = project_root / outputs_config["reference_metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("baseline_metrics.json", "early_exit_metrics.json"):
        source = reference_metrics_dir / filename
        destination = metrics_dir / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

    comparison_path = metrics_dir / "comparison.csv"
    if comparison_path.exists():
        return

    reference_comparison_path = reference_metrics_dir / "comparison.csv"
    if not reference_comparison_path.exists():
        return

    with reference_comparison_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        reference_rows = list(reader)

    if not fieldnames:
        return

    kept_methods = {"full_resnet18", "early_exit_resnet18"}
    seeded_rows = [row for row in reference_rows if row.get("method") in kept_methods]
    if not seeded_rows:
        return

    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seeded_rows)


def append_comparison_row(metrics_dir: str | Path, row: dict[str, Any]) -> Path:
    destination = Path(metrics_dir) / "comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_exists = destination.exists()
    fieldnames = [
        "method",
        "accuracy",
        "flops_per_sample",
        "flops_reduction",
        "latency_ms",
        "latency_reduction",
        "co2_kg",
        "co2_reduction",
        "avg_exit",
        "notes",
    ]
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})
    return destination


def upsert_comparison_row(metrics_dir: str | Path, row: dict[str, Any]) -> Path:
    destination = Path(metrics_dir) / "comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "accuracy",
        "flops_per_sample",
        "flops_reduction",
        "latency_ms",
        "latency_reduction",
        "co2_kg",
        "co2_reduction",
        "avg_exit",
        "notes",
    ]

    rows: list[dict[str, Any]] = []
    if destination.exists():
        with destination.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    filtered_rows = [existing for existing in rows if existing.get("method") != row.get("method")]
    filtered_rows.append({key: row.get(key) for key in fieldnames})

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    return destination
