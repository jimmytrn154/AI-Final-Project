from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.config import BASE_DEFAULTS, resolve_run_outputs
from ai_final_project.results import bootstrap_run_metrics


class OutputVersioningTests(unittest.TestCase):
    def test_resolve_run_outputs_preserves_legacy_paths_without_run_name(self) -> None:
        config = {
            "outputs": {
                "checkpoint_dir": "checkpoints",
                "metrics_dir": "outputs/metrics",
                "plots_dir": "outputs/plots",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_run_outputs(config, temp_dir)

        self.assertEqual(resolved["outputs"]["checkpoint_dir"], "checkpoints")
        self.assertEqual(resolved["outputs"]["metrics_dir"], "outputs/metrics")
        self.assertEqual(resolved["outputs"]["plots_dir"], "outputs/plots")

    def test_resolve_run_outputs_rewrites_run_specific_paths_and_manifest(self) -> None:
        config = {
            "outputs": dict(BASE_DEFAULTS["outputs"]),
            "controller_checkpoint_path": "checkpoints/controller_best.pt",
            "warm_start_checkpoint": "checkpoints/controller_best.pt",
            "reinforce_checkpoint_path": "checkpoints/reinforce_controller_best.pt",
            "reinforce_last_checkpoint_path": "checkpoints/reinforce_controller_last.pt",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_run_outputs(config, temp_dir, "2026-06-19-direct-latency")
            run_root = Path(temp_dir) / "outputs/runs/2026-06-19-direct-latency"

            self.assertEqual(
                resolved["outputs"]["checkpoint_dir"],
                "outputs/runs/2026-06-19-direct-latency/checkpoints",
            )
            self.assertEqual(
                resolved["outputs"]["metrics_dir"],
                "outputs/runs/2026-06-19-direct-latency/metrics",
            )
            self.assertEqual(
                resolved["outputs"]["plots_dir"],
                "outputs/runs/2026-06-19-direct-latency/plots",
            )
            self.assertEqual(
                resolved["controller_checkpoint_path"],
                "outputs/runs/2026-06-19-direct-latency/checkpoints/controller_best.pt",
            )
            self.assertEqual(
                resolved["warm_start_checkpoint"],
                "outputs/runs/2026-06-19-direct-latency/checkpoints/controller_best.pt",
            )
            self.assertTrue((run_root / "run_manifest.json").exists())

    def test_bootstrap_run_metrics_copies_reference_files_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            reference_dir = project_root / "outputs/metrics"
            reference_dir.mkdir(parents=True, exist_ok=True)

            (reference_dir / "baseline_metrics.json").write_text('{"accuracy": 0.88}\n', encoding="utf-8")
            (reference_dir / "early_exit_metrics.json").write_text('{"accuracy": 0.87}\n', encoding="utf-8")
            with (reference_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"method": "full_resnet18", "accuracy": "0.88"},
                        {"method": "early_exit_resnet18", "accuracy": "0.87"},
                        {"method": "fixed_threshold_0.95", "accuracy": "0.86"},
                    ]
                )

            outputs_config = {
                "run_name": "2026-06-19-direct-latency",
                "metrics_dir": "outputs/runs/2026-06-19-direct-latency/metrics",
                "reference_metrics_dir": "outputs/metrics",
            }

            bootstrap_run_metrics(project_root, outputs_config)
            bootstrap_run_metrics(project_root, outputs_config)

            run_metrics_dir = project_root / outputs_config["metrics_dir"]
            self.assertTrue((run_metrics_dir / "baseline_metrics.json").exists())
            self.assertTrue((run_metrics_dir / "early_exit_metrics.json").exists())

            with (run_metrics_dir / "comparison.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["method"] for row in rows], ["full_resnet18", "early_exit_resnet18"])


if __name__ == "__main__":
    unittest.main()
