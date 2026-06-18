from __future__ import annotations

import json
from copy import deepcopy
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
    },
}


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
