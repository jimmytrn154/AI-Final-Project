from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import EARLY_EXIT_DEFAULTS, load_json, merge_defaults
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import evaluate_early_exit_model
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.profiling import estimate_flops, measure_latency, track_emissions
from ai_final_project.results import append_comparison_row, write_metrics_json
from ai_final_project.training import train_early_exit_model
from ai_final_project.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the early-exit ResNet-18 model.")
    parser.add_argument("--config", default="configs/early_exit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), EARLY_EXIT_DEFAULTS)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])
    model = EarlyExitResNet18(num_classes=10)
    checkpoint_path = PROJECT_ROOT / config["outputs"]["checkpoint_dir"] / "early_exit_best.pt"

    train_result, emissions = track_emissions(
        lambda: train_early_exit_model(model, dataloaders["train"], dataloaders["val"], device, config, checkpoint_path),
        project_name="early-exit-training",
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    test_metrics = evaluate_early_exit_model(model, dataloaders["test"], device, config["training"]["loss_weights"])
    sample_batch = next(iter(dataloaders["test"]))[0][:1]
    flops = estimate_flops(model, sample_batch)
    latency_ms = measure_latency(model, sample_batch, device)

    payload = {
        "config": config,
        "train_result": train_result,
        "test_metrics": test_metrics,
        "flops_per_sample": flops,
        "latency_ms": latency_ms,
        "co2_kg": emissions,
        "flops_ratios": model.flops_ratios,
        "exit_names": model.exit_names,
    }
    write_metrics_json(config["outputs"]["metrics_dir"], "early_exit_metrics.json", payload)
    append_comparison_row(
        config["outputs"]["metrics_dir"],
        {
            "method": "early_exit_resnet18",
            "accuracy": test_metrics["final_accuracy"],
            "flops_per_sample": flops,
            "flops_reduction": 0.0,
            "latency_ms": latency_ms,
            "latency_reduction": 0.0,
            "co2_kg": emissions,
            "co2_reduction": 0.0,
            "avg_exit": "n/a",
            "notes": "Backbone with auxiliary exits trained jointly; no early-exit policy applied.",
        },
    )


if __name__ == "__main__":
    main()
