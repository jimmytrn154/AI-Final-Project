from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.models import EarlyExitResNet18, build_cifar_resnet18


class AlwaysExitController(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-10.0, 10.0]], dtype=features.dtype, device=features.device).expand(features.size(0), -1)


class AlwaysContinueController(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[10.0, -10.0]], dtype=features.dtype, device=features.device).expand(features.size(0), -1)

    def policy(self, features: torch.Tensor):
        return torch.distributions.Categorical(logits=self(features))


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

    def test_forward_until_matches_forward_all_exits(self) -> None:
        model = EarlyExitResNet18(num_classes=10)
        images = torch.randn(2, 3, 32, 32)
        logits_list = model.forward_all_exits(images)

        for exit_index, expected_logits in enumerate(logits_list):
            actual_logits = model.forward_until(images, exit_index)
            self.assertTrue(torch.allclose(actual_logits, expected_logits))

    def test_forward_with_policy_respects_extreme_thresholds(self) -> None:
        model = EarlyExitResNet18(num_classes=10)
        images = torch.randn(3, 3, 32, 32)

        _, early_indices = model.forward_with_policy(images, "fixed", threshold=0.0)
        _, final_indices = model.forward_with_policy(images, "fixed", threshold=1.1)

        self.assertEqual(early_indices.tolist(), [0, 0, 0])
        self.assertEqual(final_indices.tolist(), [3, 3, 3])

    def test_forward_with_supervised_controller_routes_deterministically(self) -> None:
        model = EarlyExitResNet18(num_classes=10)
        images = torch.randn(2, 3, 32, 32)

        _, exit_indices = model.forward_with_supervised_controller(images, AlwaysExitController())
        self.assertEqual(exit_indices.tolist(), [0, 0])

    def test_forward_with_reinforce_controller_can_force_final_exit(self) -> None:
        model = EarlyExitResNet18(num_classes=10)
        images = torch.randn(2, 3, 32, 32)

        _, exit_indices = model.forward_with_reinforce_controller(images, AlwaysContinueController())
        self.assertEqual(exit_indices.tolist(), [3, 3])


if __name__ == "__main__":
    unittest.main()
