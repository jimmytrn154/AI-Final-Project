from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.evaluation import collect_exit_strategy_records, decide_exit


class DummyEarlyExitModel(torch.nn.Module):
    @property
    def flops_ratios(self) -> list[float]:
        return [0.2, 0.4, 0.7, 1.0]

    @property
    def exit_names(self) -> list[str]:
        return ["exit_1", "exit_2", "exit_3", "final_exit"]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        batch_size = images.size(0)
        return [
            torch.tensor([[6.0, 0.1]], dtype=torch.float32, device=images.device).expand(batch_size, -1),
            torch.tensor([[0.1, 6.0]], dtype=torch.float32, device=images.device).expand(batch_size, -1),
            torch.tensor([[0.1, 6.0]], dtype=torch.float32, device=images.device).expand(batch_size, -1),
            torch.tensor([[0.1, 6.0]], dtype=torch.float32, device=images.device).expand(batch_size, -1),
        ]


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits_list = [
            torch.tensor([[6.0, 0.1], [1.0, 0.9]], dtype=torch.float32),
            torch.tensor([[2.0, 0.5], [4.0, 0.1]], dtype=torch.float32),
            torch.tensor([[1.0, 0.5], [0.1, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.2], [0.3, 0.1]], dtype=torch.float32),
        ]
        self.flops_ratios = [0.2, 0.4, 0.7, 1.0]

    def test_fixed_threshold_exits_early(self) -> None:
        _, exit_indices = decide_exit(self.logits_list, self.flops_ratios, "fixed", threshold=0.8)
        self.assertEqual(exit_indices.tolist(), [0, 1])

    def test_budget_first_becomes_more_permissive(self) -> None:
        _, exit_indices = decide_exit(
            self.logits_list,
            self.flops_ratios,
            "budget_first",
            threshold=0.95,
            alpha=0.5,
        )
        self.assertEqual(exit_indices.tolist(), [0, 1])

    def test_collect_exit_strategy_records_returns_one_record_per_example(self) -> None:
        dataloader = DataLoader(TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 0, 0])), batch_size=2)
        records = collect_exit_strategy_records(
            DummyEarlyExitModel(),
            dataloader,
            torch.device("cpu"),
            strategy="fixed",
            threshold=0.8,
            method="dummy_fixed",
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["method"], "dummy_fixed")
        self.assertEqual(records[0]["exit_index"], 1)
        self.assertIn("confidence", records[0])
        self.assertIn("flops_ratio", records[0])


if __name__ == "__main__":
    unittest.main()
