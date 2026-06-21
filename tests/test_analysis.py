from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.analysis import (
    bootstrap_statistics,
    classification_metrics_from_records,
    exit_distribution_from_records,
    per_exit_analysis,
)
from ai_final_project.controller import choose_target_exit


class AnalysisMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"label": 0, "prediction": 0, "exit_index": 1, "confidence": 0.90, "flops_ratio": 0.2, "correct": True},
            {"label": 0, "prediction": 1, "exit_index": 2, "confidence": 0.60, "flops_ratio": 0.4, "correct": False},
            {"label": 1, "prediction": 1, "exit_index": 2, "confidence": 0.80, "flops_ratio": 0.4, "correct": True},
            {"label": 1, "prediction": 1, "exit_index": 4, "confidence": 0.95, "flops_ratio": 1.0, "correct": True},
        ]

    def test_classification_metrics_known_values(self) -> None:
        metrics = classification_metrics_from_records(self.records, num_classes=2)

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_precision"], (1.0 + (2.0 / 3.0)) / 2.0)
        self.assertAlmostEqual(metrics["macro_recall"], (0.5 + 1.0) / 2.0)

    def test_bootstrap_statistics_contains_expected_metrics(self) -> None:
        stats = bootstrap_statistics(self.records, num_classes=2, num_bootstrap=20, seed=7)

        self.assertIn("accuracy", stats)
        self.assertIn("macro_f1", stats)
        self.assertIn("flops_reduction", stats)
        self.assertLessEqual(stats["accuracy"]["ci95_low"], stats["accuracy"]["mean"])
        self.assertGreaterEqual(stats["accuracy"]["ci95_high"], stats["accuracy"]["mean"])

    def test_exit_distribution_and_per_exit_analysis(self) -> None:
        distribution = exit_distribution_from_records(self.records, ["exit_1", "exit_2", "exit_3", "final_exit"])
        analysis = per_exit_analysis(self.records, ["exit_1", "exit_2", "exit_3", "final_exit"])

        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertAlmostEqual(distribution["exit_2"], 0.5)
        self.assertEqual(analysis["exit_2"]["count"], 2)
        self.assertAlmostEqual(analysis["exit_2"]["accuracy"], 0.5)


class ControllerLabelAblationTests(unittest.TestCase):
    def test_earliest_correct_labels_do_not_change_with_lambda(self) -> None:
        logits_list = [
            torch.tensor([[4.0, 0.1], [0.1, 4.0]], dtype=torch.float32),
            torch.tensor([[0.1, 4.0], [4.0, 0.1]], dtype=torch.float32),
            torch.tensor([[4.0, 0.1], [4.0, 0.1]], dtype=torch.float32),
            torch.tensor([[4.0, 0.1], [0.1, 4.0]], dtype=torch.float32),
        ]
        labels = torch.tensor([0, 1])
        flops_ratios = [0.2, 0.4, 0.7, 1.0]

        low_lambda = choose_target_exit(logits_list, labels, flops_ratios, "earliest_correct", 0.05)
        high_lambda = choose_target_exit(logits_list, labels, flops_ratios, "earliest_correct", 0.50)

        self.assertEqual(low_lambda.tolist(), high_lambda.tolist())

    def test_best_reward_labels_can_change_with_lambda(self) -> None:
        logits_list = [
            torch.tensor([[0.1, 4.0]], dtype=torch.float32),
            torch.tensor([[4.0, 0.1]], dtype=torch.float32),
            torch.tensor([[4.0, 0.1]], dtype=torch.float32),
            torch.tensor([[4.0, 0.1]], dtype=torch.float32),
        ]
        labels = torch.tensor([0])
        flops_ratios = [0.2, 0.4, 0.7, 1.0]

        low_lambda = choose_target_exit(logits_list, labels, flops_ratios, "best_reward", 0.05)
        high_lambda = choose_target_exit(logits_list, labels, flops_ratios, "best_reward", 6.00)

        self.assertEqual(low_lambda.tolist(), [1])
        self.assertEqual(high_lambda.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
