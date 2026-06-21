from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.analysis import (
    add_latency_to_records,
    bootstrap_statistics,
    classification_metrics_from_records,
    per_exit_analysis,
)
from ai_final_project.config import BASE_DEFAULTS, load_json, merge_defaults, resolve_run_outputs, save_json
from ai_final_project.controller import ExitController
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.evaluation import (
    collect_classifier_records,
    collect_controller_records,
    collect_early_exit_final_records,
    collect_exit_strategy_records,
    collect_reinforce_controller_records,
)
from ai_final_project.models import EarlyExitResNet18, build_cifar_resnet18
from ai_final_project.results import bootstrap_run_metrics
from ai_final_project.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze statistical robustness for selected early-exit methods.")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--run-name", default="2026-06-20-deeper-experiments")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--num-classes", type=int, default=10)
    return parser.parse_args()


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def load_early_exit_model(device: torch.device) -> EarlyExitResNet18:
    model = EarlyExitResNet18(num_classes=10)
    model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/early_exit_best.pt", map_location=device, weights_only=True))
    model.to(device)
    return model


def selected_policy_specs(metrics_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    fixed_payload = maybe_load_json(metrics_dir / "threshold_ablation_results.json")
    if fixed_payload:
        best = fixed_payload["best_result"]
        specs.append(
            {
                "kind": "fixed",
                "method": best["method"],
                "threshold": best["threshold"],
                "latency_ms": best["test"].get("latency_ms"),
            }
        )

    dynamic_payload = maybe_load_json(metrics_dir / "dynamic_threshold_ablation_results.json")
    if dynamic_payload:
        best = dynamic_payload["best_result"]
        specs.append(
            {
                "kind": "dynamic",
                "method": best["method"],
                "strategy": best["mode"],
                "threshold": best["base_threshold"],
                "alpha": best["alpha"],
                "latency_ms": best["test"].get("latency_ms"),
            }
        )

    controller_payload = maybe_load_json(metrics_dir / "controller_lambda_sweep_results.json")
    if controller_payload:
        best = controller_payload["best_result"]
        specs.append(
            {
                "kind": "controller",
                "method": best["method"],
                "checkpoint_path": best["checkpoint_path"],
                "latency_ms": best["test"].get("latency_ms"),
            }
        )

    reinforce_payload = maybe_load_json(metrics_dir / "reinforce_lambda_sweep_results.json")
    if reinforce_payload:
        best = reinforce_payload["best_result"]
        specs.append(
            {
                "kind": "reinforce",
                "method": best["method"],
                "checkpoint_path": best["checkpoint_path"],
                "latency_ms": best["test"].get("latency_ms"),
            }
        )

    return specs


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), BASE_DEFAULTS)
    config = resolve_run_outputs(config, PROJECT_ROOT, args.run_name)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])

    bootstrap_run_metrics(PROJECT_ROOT, config["outputs"])
    metrics_dir = PROJECT_ROOT / config["outputs"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = build_cifar_resnet18(num_classes=args.num_classes)
    baseline_model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/baseline_best.pt", map_location=device, weights_only=True))
    baseline_model.to(device)
    early_exit_model = load_early_exit_model(device)

    method_records: dict[str, list[dict[str, Any]]] = {}
    method_exit_names: dict[str, list[str]] = {}
    method_records["full_resnet18"] = collect_classifier_records(
        baseline_model,
        dataloaders["test"],
        device,
        method="full_resnet18",
    )
    method_exit_names["full_resnet18"] = ["final"]

    method_records["early_exit_final"] = collect_early_exit_final_records(
        early_exit_model,
        dataloaders["test"],
        device,
        method="early_exit_final",
    )
    method_exit_names["early_exit_final"] = early_exit_model.exit_names

    for spec in selected_policy_specs(metrics_dir):
        if spec["kind"] == "fixed":
            records = collect_exit_strategy_records(
                early_exit_model,
                dataloaders["test"],
                device,
                "fixed",
                spec["threshold"],
                method=spec["method"],
            )
        elif spec["kind"] == "dynamic":
            records = collect_exit_strategy_records(
                early_exit_model,
                dataloaders["test"],
                device,
                spec["strategy"],
                spec["threshold"],
                alpha=spec["alpha"],
                method=spec["method"],
            )
        elif spec["kind"] == "controller":
            controller = ExitController()
            controller.load_state_dict(torch.load(PROJECT_ROOT / spec["checkpoint_path"], map_location=device, weights_only=True))
            controller.to(device)
            records = collect_controller_records(
                early_exit_model,
                controller,
                dataloaders["test"],
                device,
                method=spec["method"],
            )
        elif spec["kind"] == "reinforce":
            controller = ExitController()
            controller.load_state_dict(torch.load(PROJECT_ROOT / spec["checkpoint_path"], map_location=device, weights_only=True))
            controller.to(device)
            records = collect_reinforce_controller_records(
                early_exit_model,
                controller,
                dataloaders["test"],
                device,
                method=spec["method"],
            )
        else:
            continue

        method_records[spec["method"]] = add_latency_to_records(records, spec.get("latency_ms"))
        method_exit_names[spec["method"]] = early_exit_model.exit_names

    classification_payload = {
        method: classification_metrics_from_records(records, args.num_classes)
        for method, records in method_records.items()
    }
    bootstrap_payload = {
        method: bootstrap_statistics(
            records,
            args.num_classes,
            num_bootstrap=args.bootstrap_samples,
            seed=config["seed"],
        )
        for method, records in method_records.items()
    }
    per_exit_payload = {
        method: per_exit_analysis(records, method_exit_names[method])
        for method, records in method_records.items()
    }

    save_json(metrics_dir / "classification_metrics.json", classification_payload)
    save_json(metrics_dir / "bootstrap_statistics.json", bootstrap_payload)
    save_json(metrics_dir / "per_exit_analysis.json", per_exit_payload)


if __name__ == "__main__":
    main()
