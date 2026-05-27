from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.evaluation import decide_exit


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


if __name__ == "__main__":
    unittest.main()
