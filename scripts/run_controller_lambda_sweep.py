from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.analysis import append_csv_rows
from ai_final_project.config import CONTROLLER_DEFAULTS, load_json, merge_defaults, resolve_run_outputs
from ai_final_project.controller import ExitController, build_controller_dataset, evaluate_controller
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import benchmark_controller_latency
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.results import bootstrap_run_metrics, write_metrics_json
from ai_final_project.training import train_controller
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
    parser = argparse.ArgumentParser(description="Run supervised controller reward-label lambda sweep.")
    parser.add_argument("--config", default="configs/controller.json")
    parser.add_argument("--run-name", default="2026-06-19-deeper-experiments")
    parser.add_argument("--lambdas", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--timed-batches", type=int, default=None)
    return parser.parse_args()


def parse_float_values(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def load_baseline(metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / "baseline_metrics.json"
    return load_json(path) if path.exists() else {}


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


def comparison_row(method: str, split: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "method": method,
        "split": split,
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
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), CONTROLLER_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    config["supervision_strategy"] = "best_reward"
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

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

    results = []
    comparison_rows = []
    for lambda_cost in parse_float_values(args.lambdas):
        run_config = deepcopy(config)
        run_config["reward_lambda"] = lambda_cost
        checkpoint_path = checkpoint_dir / f"controller_best_reward_lambda_{lambda_cost:.2f}.pt"

        controller_dataset = build_controller_dataset(
            early_exit_model,
            dataloaders["train"],
            device,
            run_config["supervision_strategy"],
            run_config["reward_lambda"],
        )
        train_size = int(0.9 * len(controller_dataset))
        val_size = len(controller_dataset) - train_size
        train_dataset, val_dataset = random_split(
            controller_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(run_config["seed"]),
        )
        train_loader = DataLoader(train_dataset, batch_size=run_config["data"]["batch_size"], shuffle=True)
        controller_val_loader = DataLoader(val_dataset, batch_size=run_config["data"]["batch_size"], shuffle=False)

        controller = ExitController()
        train_result = train_controller(controller, train_loader, controller_val_loader, device, run_config, checkpoint_path)
        controller.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        controller.to(device)

        val_metrics = evaluate_controller(early_exit_model, controller, dataloaders["val"], device)
        test_metrics = evaluate_controller(early_exit_model, controller, dataloaders["test"], device)
        latency_metrics = None
        if not args.skip_latency:
            latency_metrics = benchmark_controller_latency(
                early_exit_model,
                controller,
                dataloaders["test"],
                device,
                timed_batches=args.timed_batches,
            )

        enriched_test = enrich_metrics(test_metrics, baseline, latency_metrics)
        method = f"learned_controller_best_reward_lambda_{lambda_cost:.2f}"
        entry = {
            "method": method,
            "lambda_cost": lambda_cost,
            "supervision_strategy": "best_reward",
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "train_result": train_result,
            "val": val_metrics,
            "test": enriched_test,
        }
        results.append(entry)
        comparison_rows.append(
            comparison_row(
                method,
                "test",
                enriched_test,
                "Supervised best_reward label sweep; reward_lambda changes target exits; CO2 is FLOPs-scaled.",
            )
        )

    best_result = max(results, key=lambda item: (item["test"]["accuracy"], item["test"]["flops_reduction"]))
    payload = {
        "config": config,
        "lambda_values": parse_float_values(args.lambdas),
        "reference_note": "reward_lambda has no effect for earliest_correct; this sweep uses best_reward labels.",
        "results": results,
        "best_result": best_result,
    }
    write_metrics_json(config["outputs"]["metrics_dir"], "controller_lambda_sweep_results.json", payload)
    append_csv_rows(metrics_dir / "ablation_comparison.csv", comparison_rows, ABLATION_FIELDS)


if __name__ == "__main__":
    main()
