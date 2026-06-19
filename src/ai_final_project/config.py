from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def merge_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


BASE_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "num_workers": 2,
    "data": {
        "root": "data",
        "batch_size": 128,
        "val_split": 0.1,
        "normalize_mean": [0.4914, 0.4822, 0.4465],
        "normalize_std": [0.2023, 0.1994, 0.2010],
    },
    "training": {
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.0,
    },
    "outputs": {
        "checkpoint_dir": "checkpoints",
        "metrics_dir": "outputs/metrics",
        "plots_dir": "outputs/plots",
        "run_name": None,
        "runs_root": "outputs/runs",
        "reference_metrics_dir": "outputs/metrics",
    },
}


def resolve_run_outputs(
    config: dict[str, Any],
    project_root: str | Path,
    run_name_override: str | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(config)
    outputs = resolved.setdefault("outputs", {})
    outputs.setdefault("run_name", None)
    outputs.setdefault("runs_root", "outputs/runs")
    outputs.setdefault("reference_metrics_dir", "outputs/metrics")

    run_name = run_name_override if run_name_override is not None else outputs.get("run_name")
    if not run_name:
        return resolved

    legacy_checkpoint_dir = Path(outputs["checkpoint_dir"])
    outputs["run_name"] = run_name
    run_root = Path(outputs["runs_root"]) / run_name
    checkpoint_dir = run_root / "checkpoints"
    metrics_dir = run_root / "metrics"
    plots_dir = run_root / "plots"

    outputs["checkpoint_dir"] = str(checkpoint_dir)
    outputs["metrics_dir"] = str(metrics_dir)
    outputs["plots_dir"] = str(plots_dir)

    managed_checkpoint_keys = (
        "controller_checkpoint_path",
        "warm_start_checkpoint",
        "reinforce_checkpoint_path",
        "reinforce_last_checkpoint_path",
    )
    for key in managed_checkpoint_keys:
        if key in resolved:
            current_path = Path(resolved[key])
            if current_path.parent == legacy_checkpoint_dir:
                resolved[key] = str(checkpoint_dir / current_path.name)

    manifest_path = Path(project_root) / run_root / "run_manifest.json"
    if not manifest_path.exists():
        save_json(
            manifest_path,
            {
                "run_name": run_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reference_metrics_dir": outputs["reference_metrics_dir"],
                "checkpoint_dir": outputs["checkpoint_dir"],
                "metrics_dir": outputs["metrics_dir"],
                "plots_dir": outputs["plots_dir"],
            },
        )

    return resolved


EARLY_EXIT_DEFAULTS: dict[str, Any] = merge_defaults(
    {
        "training": {
            "epochs": 25,
            "loss_weights": [0.3, 0.5, 0.7, 1.0],
        }
    },
    BASE_DEFAULTS,
)


THRESHOLD_SWEEP_DEFAULTS: dict[str, Any] = merge_defaults(
    {
        "thresholds": [0.6, 0.7, 0.8, 0.9, 0.95],
        "checkpoint_path": "checkpoints/early_exit_best.pt",
        "experiment_name": "fixed_threshold_sweep",
    },
    EARLY_EXIT_DEFAULTS,
)


DYNAMIC_THRESHOLD_DEFAULTS: dict[str, Any] = merge_defaults(
    {
        "checkpoint_path": "checkpoints/early_exit_best.pt",
        "experiment_name": "dynamic_threshold",
        "base_threshold": 0.85,
        "alpha_values": [0.1, 0.2, 0.3],
        "modes": ["accuracy_first", "budget_first"],
    },
    EARLY_EXIT_DEFAULTS,
)


CONTROLLER_DEFAULTS: dict[str, Any] = merge_defaults(
    {
        "checkpoint_path": "checkpoints/early_exit_best.pt",
        "controller_checkpoint_path": "checkpoints/controller_best.pt",
        "supervision_strategy": "earliest_correct",
        "reward_lambda": 0.1,
        "training": {
            "epochs": 15,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
        },
    },
    EARLY_EXIT_DEFAULTS,
)


REINFORCE_CONTROLLER_DEFAULTS: dict[str, Any] = merge_defaults(
    {
        "checkpoint_path": "checkpoints/early_exit_best.pt",
        "warm_start_checkpoint": "checkpoints/controller_best.pt",
        "reinforce_checkpoint_path": "checkpoints/reinforce_controller_best.pt",
        "reinforce_last_checkpoint_path": "checkpoints/reinforce_controller_last.pt",
        "supervised_warm_start": True,
        "lambda_cost": 0.20,
        "wrong_penalty": 0.0,
        "entropy_beta": 0.005,
        "baseline_momentum": 0.90,
        "gradient_clip_norm": 1.0,
        "controller": {
            "input_dim": 5,
            "hidden_dim": 32,
        },
        "training": {
            "epochs": 30,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
        },
    },
    EARLY_EXIT_DEFAULTS,
)
