from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import DYNAMIC_THRESHOLD_DEFAULTS, load_json, merge_defaults, resolve_run_outputs
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import benchmark_exit_strategy_latency, evaluate_exit_strategy
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.results import append_comparison_row, bootstrap_run_metrics, write_metrics_json
from ai_final_project.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic threshold early-exit experiments.")
    parser.add_argument("--config", default="configs/dynamic_threshold.json")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), DYNAMIC_THRESHOLD_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])

    model = EarlyExitResNet18(num_classes=10)
    checkpoint_path = PROJECT_ROOT / config["checkpoint_path"]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    bootstrap_run_metrics(PROJECT_ROOT, config["outputs"])
    baseline = load_json(PROJECT_ROOT / config["outputs"]["metrics_dir"] / "baseline_metrics.json")
    baseline_flops = baseline.get("flops_per_sample")
    baseline_latency = baseline.get("latency_ms")
    baseline_co2 = baseline.get("co2_kg")

    results = []
    for mode in config["modes"]:
        for alpha in config["alpha_values"]:
            result = evaluate_exit_strategy(
                model,
                dataloaders["test"],
                device,
                strategy=mode,
                threshold=config["base_threshold"],
                alpha=alpha,
            )
            latency_metrics = benchmark_exit_strategy_latency(
                model,
                dataloaders["test"],
                device,
                strategy=mode,
                threshold=config["base_threshold"],
                alpha=alpha,
            )
            result["estimated_flops_per_sample"] = (
                baseline_flops * result["average_flops_ratio"] if baseline_flops is not None else None
            )
            result["latency_ms"] = latency_metrics["latency_ms"]
            result["latency_std_ms"] = latency_metrics["latency_std_ms"]
            result["latency_reduction"] = (
                1.0 - (result["latency_ms"] / baseline_latency)
                if baseline_latency not in (None, 0)
                else None
            )
            result["estimated_co2_kg"] = (
                baseline_co2 * result["average_flops_ratio"] if baseline_co2 is not None else None
            )
            results.append(result)

            append_comparison_row(
                config["outputs"]["metrics_dir"],
                {
                    "method": f"{mode}_alpha_{alpha:.2f}",
                    "accuracy": result["accuracy"],
                    "flops_per_sample": result["estimated_flops_per_sample"],
                    "flops_reduction": result["flops_reduction"],
                    "latency_ms": result["latency_ms"],
                    "latency_reduction": result["latency_reduction"],
                    "co2_kg": result["estimated_co2_kg"],
                    "co2_reduction": result["flops_reduction"],
                    "avg_exit": result["avg_exit"],
                    "notes": "Latency is measured directly with conditional execution; CO2 is scaled from the baseline using average FLOPs ratio.",
                },
            )

    best_result = max(results, key=lambda item: (item["accuracy"], item["flops_reduction"]))
    write_metrics_json(
        config["outputs"]["metrics_dir"],
        "dynamic_threshold_results.json",
        {"config": config, "results": results, "best_result": best_result},
    )


if __name__ == "__main__":
    main()
