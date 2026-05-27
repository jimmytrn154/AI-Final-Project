from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import CONTROLLER_DEFAULTS, load_json, merge_defaults
from ai_final_project.controller import ExitController, build_controller_dataset, evaluate_controller
from ai_final_project.data import create_cifar10_dataloaders
from ai_final_project.models import EarlyExitResNet18
from ai_final_project.results import append_comparison_row, write_metrics_json
from ai_final_project.training import train_controller
from ai_final_project.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the learned exit controller.")
    parser.add_argument("--config", default="configs/controller.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_defaults(load_json(PROJECT_ROOT / args.config), CONTROLLER_DEFAULTS)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    dataloaders = create_cifar10_dataloaders(config["data"], config["num_workers"], config["seed"])
    early_exit_model = EarlyExitResNet18(num_classes=10)
    early_exit_model.load_state_dict(torch.load(PROJECT_ROOT / config["checkpoint_path"], map_location=device))
    early_exit_model.to(device)

    controller_dataset = build_controller_dataset(
        early_exit_model,
        dataloaders["train"],
        device,
        config["supervision_strategy"],
        config["reward_lambda"],
    )
    train_size = int(0.9 * len(controller_dataset))
    val_size = len(controller_dataset) - train_size
    train_dataset, val_dataset = random_split(
        controller_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    controller = ExitController()
    checkpoint_path = PROJECT_ROOT / config["controller_checkpoint_path"]

    train_result = train_controller(controller, train_loader, val_loader, device, config, checkpoint_path)
    controller.load_state_dict(torch.load(checkpoint_path, map_location=device))
    controller.to(device)
    test_metrics = evaluate_controller(early_exit_model, controller, dataloaders["test"], device)

    payload = {
        "config": config,
        "train_result": train_result,
        "test_metrics": test_metrics,
    }
    write_metrics_json(config["outputs"]["metrics_dir"], "controller_results.json", payload)
    append_comparison_row(
        config["outputs"]["metrics_dir"],
        {
            "method": f"learned_controller_{config['supervision_strategy']}",
            "accuracy": test_metrics["accuracy"],
            "flops_per_sample": None,
            "flops_reduction": test_metrics["flops_reduction"],
            "latency_ms": None,
            "latency_reduction": None,
            "co2_kg": None,
            "co2_reduction": None,
            "avg_exit": test_metrics["avg_exit"],
            "notes": "Controller metrics use learned exit decisions; direct latency/emissions estimation still pending.",
        },
    )


if __name__ == "__main__":
    main()
