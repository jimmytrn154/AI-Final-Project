from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.analysis import append_csv_rows
from ai_final_project.config import REINFORCE_CONTROLLER_DEFAULTS, load_json, merge_defaults, resolve_run_outputs
from ai_final_project.controller import ExitController, evaluate_reinforce_controller
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import benchmark_reinforce_controller_latency
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.results import bootstrap_run_metrics, write_metrics_json
from ai_final_project.training import train_reinforce_controller
from ai_final_project.utils import resolve_device, set_seed


ABLATION_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REINFORCE controller lambda sweep.")
    parser.add_argument("--config", default="configs/reinforce_controller.json")
    parser.add_argument("--run-name", default="2026-06-19-deeper-experiments")
    parser.add_argument("--lambdas", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wrong-penalty", type=float, default=0.0)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--timed-batches", type=int, default=None)
    return parser.parse_args()


def parse_float_values(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def load_baseline(metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / "baseline_metrics.json"
    return load_json(path) if path.exists() else {}


def resolve_warm_start(path: Path) -> Path:
    if path.exists():
        return path
    fallback = PROJECT_ROOT / "checkpoints/controller_best.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No supervised controller warm-start checkpoint found at {path} or {fallback}")


def enrich_metrics(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    latency_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    baseline_flops = baseline.get("flops_per_sample")
    baseline_latency = baseline.get("latency_ms")
    baseline_co2 = baseline.get("co2_kg")
    enriched = dict(metrics)
    enriched["flops_per_sample"] = (
        baseline_flops * metrics["average_flops_ratio"] if baseline_flops is not None else None
    )
    enriched["latency_ms"] = latency_metrics["latency_ms"] if latency_metrics else None
    enriched["latency_std_ms"] = latency_metrics["latency_std_ms"] if latency_metrics else None
    enriched["latency_reduction"] = (
        1.0 - (enriched["latency_ms"] / baseline_latency)
        if baseline_latency not in (None, 0) and enriched["latency_ms"] is not None
        else None
    )
    enriched["co2_kg"] = baseline_co2 * metrics["average_flops_ratio"] if baseline_co2 is not None else None
    enriched["co2_reduction"] = metrics["flops_reduction"]
    return enriched


def comparison_row(method: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "method": method,
        "split": "test",
        "accuracy": metrics["accuracy"],
        "flops_per_sample": metrics.get("flops_per_sample"),
        "flops_reduction": metrics["flops_reduction"],
        "latency_ms": metrics.get("latency_ms"),
        "latency_reduction": metrics.get("latency_reduction"),
        "co2_kg": metrics.get("co2_kg"),
        "co2_reduction": metrics.get("co2_reduction"),
        "avg_exit": metrics["avg_exit"],
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), REINFORCE_CONTROLLER_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    config["wrong_penalty"] = args.wrong_penalty

    set_seed(config["seed"])
    device = resolve_device(config["device"])
    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])

    early_exit_model = EarlyExitResNet18(num_classes=10)
    early_exit_model.load_state_dict(
        torch.load(PROJECT_ROOT / config["checkpoint_path"], map_location=device, weights_only=True)
    )
    early_exit_model.to(device)

    bootstrap_run_metrics(PROJECT_ROOT, config["outputs"])
    metrics_dir = PROJECT_ROOT / config["outputs"]["metrics_dir"]
    checkpoint_dir = PROJECT_ROOT / config["outputs"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(metrics_dir)
    warm_start_path = resolve_warm_start(PROJECT_ROOT / config["warm_start_checkpoint"])

    results = []
    comparison_rows = []
    for lambda_cost in parse_float_values(args.lambdas):
        run_config = deepcopy(config)
        run_config["lambda_cost"] = lambda_cost
        checkpoint_path = checkpoint_dir / f"reinforce_controller_lambda_{lambda_cost:.2f}_best.pt"
        last_checkpoint_path = checkpoint_dir / f"reinforce_controller_lambda_{lambda_cost:.2f}_last.pt"

        controller = ExitController(
            input_dim=run_config["controller"]["input_dim"],
            hidden_dim=run_config["controller"]["hidden_dim"],
        )
        if run_config.get("supervised_warm_start", True):
            controller.load_state_dict(torch.load(warm_start_path, map_location=device, weights_only=True))
        controller.to(device)

        train_result = train_reinforce_controller(
            controller=controller,
            early_exit_model=early_exit_model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=device,
            config=run_config,
            checkpoint_path=checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
        )

        controller.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        controller.to(device)
        val_metrics = evaluate_reinforce_controller(
            early_exit_model,
            controller,
            dataloaders["val"],
            device,
            lambda_cost=lambda_cost,
            wrong_penalty=args.wrong_penalty,
        )
        test_metrics = evaluate_reinforce_controller(
            early_exit_model,
            controller,
            dataloaders["test"],
            device,
            lambda_cost=lambda_cost,
            wrong_penalty=args.wrong_penalty,
        )
        latency_metrics = None
        if not args.skip_latency:
            latency_metrics = benchmark_reinforce_controller_latency(
                early_exit_model,
                controller,
                dataloaders["test"],
                device,
                timed_batches=args.timed_batches,
            )

        enriched_test = enrich_metrics(test_metrics, baseline, latency_metrics)
        method = f"reinforce_controller_lambda_{lambda_cost:.2f}"
        entry = {
            "method": method,
            "lambda_cost": lambda_cost,
            "wrong_penalty": args.wrong_penalty,
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "last_checkpoint_path": str(last_checkpoint_path.relative_to(PROJECT_ROOT)),
            "warm_start_checkpoint": str(warm_start_path.relative_to(PROJECT_ROOT)),
            "train_result": train_result,
            "val": val_metrics,
            "test": enriched_test,
        }
        results.append(entry)
        comparison_rows.append(
            comparison_row(
                method,
                enriched_test,
                "REINFORCE lambda sweep; reward is correct - lambda * FLOPs ratio unless wrong_penalty is set; CO2 is FLOPs-scaled.",
            )
        )

    best_result = max(results, key=lambda item: (item["test"]["accuracy"], item["test"]["flops_reduction"]))
    payload = {
        "config": config,
        "lambda_values": parse_float_values(args.lambdas),
        "wrong_penalty": args.wrong_penalty,
        "results": results,
        "best_result": best_result,
    }
    write_metrics_json(config["outputs"]["metrics_dir"], "reinforce_lambda_sweep_results.json", payload)
    append_csv_rows(metrics_dir / "ablation_comparison.csv", comparison_rows, ABLATION_FIELDS)


if __name__ == "__main__":
    main()
