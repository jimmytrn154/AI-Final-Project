from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.models import EarlyExitResNet18, build_cifar_resnet18


class ModelShapeTests(unittest.TestCase):
    def test_baseline_output_shape(self) -> None:
        model = build_cifar_resnet18(num_classes=10)
        logits = model(torch.randn(4, 3, 32, 32))
        self.assertEqual(logits.shape, (4, 10))

    def test_early_exit_output_shapes(self) -> None:
        model = EarlyExitResNet18(num_classes=10)
        logits_list = model(torch.randn(4, 3, 32, 32))
        self.assertEqual(len(logits_list), 4)
        for logits in logits_list:
            self.assertEqual(logits.shape, (4, 10))


if __name__ == "__main__":
    unittest.main()
