from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.analysis import write_csv_rows
from ai_final_project.config import THRESHOLD_SWEEP_DEFAULTS, load_json, merge_defaults, resolve_run_outputs
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import benchmark_exit_strategy_latency, evaluate_exit_strategy
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.results import bootstrap_run_metrics, write_metrics_json
from ai_final_project.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense threshold ablations for early-exit policies.")
    parser.add_argument("--config", default="configs/fixed_threshold.json")
    parser.add_argument("--run-name", default="2026-06-19-deeper-experiments")
    parser.add_argument("--thresholds", default="0.50:0.99:0.05")
    parser.add_argument("--base-thresholds", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--alphas", default="0.05,0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--modes", default="accuracy_first,budget_first")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--timed-batches", type=int, default=None)
    return parser.parse_args()


def parse_float_values(value: str) -> list[float]:
    if ":" not in value:
        return [float(item) for item in value.split(",") if item]

    start_text, stop_text, step_text = value.split(":")
    start = float(start_text)
    stop = float(stop_text)
    step = float(step_text)
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    if values[-1] < stop:
        values.append(round(stop, 4))
    elif abs(values[-1] - stop) > 1e-9 and stop not in values:
        values.append(round(stop, 4))
    return sorted(set(values))


def load_baseline(metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / "baseline_metrics.json"
    return load_json(path) if path.exists() else {}


def enrich_result(
    result: dict[str, Any],
    baseline: dict[str, Any],
    latency_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    baseline_flops = baseline.get("flops_per_sample")
    baseline_latency = baseline.get("latency_ms")
    baseline_co2 = baseline.get("co2_kg")

    result["flops_per_sample"] = (
        baseline_flops * result["average_flops_ratio"] if baseline_flops is not None else None
    )
    result["latency_ms"] = latency_metrics["latency_ms"] if latency_metrics else None
    result["latency_std_ms"] = latency_metrics["latency_std_ms"] if latency_metrics else None
    result["latency_reduction"] = (
        1.0 - (result["latency_ms"] / baseline_latency)
        if baseline_latency not in (None, 0) and result["latency_ms"] is not None
        else None
    )
    result["co2_kg"] = baseline_co2 * result["average_flops_ratio"] if baseline_co2 is not None else None
    result["co2_reduction"] = result["flops_reduction"]
    return result


def comparison_row(method: str, split: str, result: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "method": method,
        "split": split,
        "accuracy": result["accuracy"],
        "flops_per_sample": result.get("flops_per_sample"),
        "flops_reduction": result["flops_reduction"],
        "latency_ms": result.get("latency_ms"),
        "latency_reduction": result.get("latency_reduction"),
        "co2_kg": result.get("co2_kg"),
        "co2_reduction": result.get("co2_reduction"),
        "avg_exit": result["avg_exit"],
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), THRESHOLD_SWEEP_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])

    model = EarlyExitResNet18(num_classes=10)
    model.load_state_dict(torch.load(PROJECT_ROOT / config["checkpoint_path"], map_location=device, weights_only=True))
    model.to(device)

    bootstrap_run_metrics(PROJECT_ROOT, config["outputs"])
    metrics_dir = PROJECT_ROOT / config["outputs"]["metrics_dir"]
    baseline = load_baseline(metrics_dir)

    thresholds = parse_float_values(args.thresholds)
    base_thresholds = parse_float_values(args.base_thresholds)
    alphas = parse_float_values(args.alphas)
    modes = [mode for mode in args.modes.split(",") if mode]

    fixed_results = []
    dynamic_results = []
    comparison_rows = []

    for threshold in thresholds:
        val_result = evaluate_exit_strategy(model, dataloaders["val"], device, "fixed", threshold)
        test_result = evaluate_exit_strategy(model, dataloaders["test"], device, "fixed", threshold)
        latency_metrics = None
        if not args.skip_latency:
            latency_metrics = benchmark_exit_strategy_latency(
                model,
                dataloaders["test"],
                device,
                "fixed",
                threshold,
                timed_batches=args.timed_batches,
            )
        test_result = enrich_result(test_result, baseline, latency_metrics)
        entry = {
            "method": f"fixed_threshold_{threshold:.2f}",
            "threshold": threshold,
            "val": val_result,
            "test": test_result,
        }
        fixed_results.append(entry)
        comparison_rows.append(
            comparison_row(
                entry["method"],
                "test",
                test_result,
                "Threshold ablation; latency measured directly unless skip-latency was used; CO2 is FLOPs-scaled.",
            )
        )

    for mode in modes:
        for base_threshold in base_thresholds:
            for alpha in alphas:
                val_result = evaluate_exit_strategy(
                    model,
                    dataloaders["val"],
                    device,
                    mode,
                    base_threshold,
                    alpha=alpha,
                )
                test_result = evaluate_exit_strategy(
                    model,
                    dataloaders["test"],
                    device,
                    mode,
                    base_threshold,
                    alpha=alpha,
                )
                latency_metrics = None
                if not args.skip_latency:
                    latency_metrics = benchmark_exit_strategy_latency(
                        model,
                        dataloaders["test"],
                        device,
                        mode,
                        base_threshold,
                        alpha=alpha,
                        timed_batches=args.timed_batches,
                    )
                test_result = enrich_result(test_result, baseline, latency_metrics)
                entry = {
                    "method": f"{mode}_base_{base_threshold:.2f}_alpha_{alpha:.2f}",
                    "mode": mode,
                    "base_threshold": base_threshold,
                    "alpha": alpha,
                    "val": val_result,
                    "test": test_result,
                }
                dynamic_results.append(entry)
                comparison_rows.append(
                    comparison_row(
                        entry["method"],
                        "test",
                        test_result,
                        "Dynamic-threshold ablation; latency measured directly unless skip-latency was used; CO2 is FLOPs-scaled.",
                    )
                )

    fixed_best = max(fixed_results, key=lambda item: (item["test"]["accuracy"], item["test"]["flops_reduction"]))
    dynamic_best = max(dynamic_results, key=lambda item: (item["test"]["accuracy"], item["test"]["flops_reduction"]))

    write_metrics_json(
        config["outputs"]["metrics_dir"],
        "threshold_ablation_results.json",
        {"config": config, "results": fixed_results, "best_result": fixed_best},
    )
    write_metrics_json(
        config["outputs"]["metrics_dir"],
        "dynamic_threshold_ablation_results.json",
        {"config": config, "results": dynamic_results, "best_result": dynamic_best},
    )
    write_csv_rows(
        metrics_dir / "ablation_comparison.csv",
        comparison_rows,
        [
            "method",
            "split",
            "accuracy",
            "flops_per_sample",
            "flops_reduction",
            "latency_ms",
            "latency_reduction",
            "co2_kg",
            "co2_reduction",
            "avg_exit",
            "notes",
        ],
    )


if __name__ == "__main__":
    main()
