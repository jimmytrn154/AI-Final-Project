from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import save_json


def write_metrics_json(metrics_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    destination = Path(metrics_dir) / filename
    save_json(destination, payload)
    return destination


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
