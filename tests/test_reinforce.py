from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_final_project.controller import (
    ExitController,
    build_reinforce_state,
    compute_reinforce_reward,
    evaluate_reinforce_controller,
)


class DummyEarlyExitModel(torch.nn.Module):
    def __init__(self, logits_list: list[torch.Tensor]) -> None:
        super().__init__()
        self._logits_list = logits_list
        self._flops_ratios = [0.2, 0.4, 0.7, 1.0]
        self._exit_names = ["exit_1", "exit_2", "exit_3", "final_exit"]

    @property
    def flops_ratios(self) -> list[float]:
        return self._flops_ratios

    @property
    def exit_names(self) -> list[str]:
        return self._exit_names

    def forward_all_exits(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [logits.to(images.device) for logits in self._logits_list]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_all_exits(images)


class AlwaysContinueController(torch.nn.Module):
    def eval(self):
        return self

    def policy(self, state: torch.Tensor):
        return torch.distributions.Categorical(logits=torch.tensor([[10.0, -10.0]], device=state.device).expand(state.size(0), -1))


class AlwaysExitController(torch.nn.Module):
    def eval(self):
        return self

    def policy(self, state: torch.Tensor):
        return torch.distributions.Categorical(logits=torch.tensor([[-10.0, 10.0]], device=state.device).expand(state.size(0), -1))


class ReinforceHelperTests(unittest.TestCase):
    def test_build_reinforce_state_shape_and_normalization(self) -> None:
        logits = torch.tensor([[4.0, 1.0, 0.5], [1.0, 2.0, 0.5]], dtype=torch.float32)
        state = build_reinforce_state(logits, exit_index=1, flops_ratio=0.42, num_exits=4)

        self.assertEqual(state.shape, (2, 5))
        self.assertTrue(torch.all(state[:, 1] >= 0.0))
        self.assertTrue(torch.all(state[:, 1] <= 1.0))
        self.assertTrue(torch.allclose(state[:, 3], torch.full((2,), 1.0 / 3.0)))
        self.assertTrue(torch.allclose(state[:, 4], torch.full((2,), 0.42)))

    def test_compute_reinforce_reward(self) -> None:
        logits = torch.tensor([[4.0, 0.1], [4.0, 0.1]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        rewards = compute_reinforce_reward(logits, labels, flops_ratio=0.25, lambda_cost=0.2)
        penalized_rewards = compute_reinforce_reward(logits, labels, flops_ratio=0.25, lambda_cost=0.2, wrong_penalty=1.0)

        self.assertTrue(torch.allclose(rewards, torch.tensor([0.95, -0.05])))
        self.assertTrue(torch.allclose(penalized_rewards, torch.tensor([0.95, -1.05])))

    def test_evaluate_reinforce_controller_forces_final_exit(self) -> None:
        logits_list = [
            torch.tensor([[3.0, 0.2]], dtype=torch.float32),
            torch.tensor([[0.5, 0.4]], dtype=torch.float32),
            torch.tensor([[0.4, 0.3]], dtype=torch.float32),
            torch.tensor([[6.0, 0.1]], dtype=torch.float32),
        ]
        model = DummyEarlyExitModel(logits_list)
        dataloader = DataLoader(TensorDataset(torch.randn(1, 3, 32, 32), torch.tensor([0])), batch_size=1)

        metrics = evaluate_reinforce_controller(
            model,
            AlwaysContinueController(),
            dataloader,
            torch.device("cpu"),
            lambda_cost=0.2,
        )

        self.assertEqual(metrics["avg_exit"], 4.0)
        self.assertEqual(metrics["exit_distribution"]["final_exit"], 1.0)

    def test_evaluate_reinforce_controller_uses_deterministic_argmax(self) -> None:
        logits_list = [
            torch.tensor([[6.0, 0.1]], dtype=torch.float32),
            torch.tensor([[0.2, 0.5]], dtype=torch.float32),
            torch.tensor([[0.2, 0.4]], dtype=torch.float32),
            torch.tensor([[0.1, 4.0]], dtype=torch.float32),
        ]
        model = DummyEarlyExitModel(logits_list)
        dataloader = DataLoader(TensorDataset(torch.randn(1, 3, 32, 32), torch.tensor([0])), batch_size=1)

        metrics = evaluate_reinforce_controller(
            model,
            AlwaysExitController(),
            dataloader,
            torch.device("cpu"),
            lambda_cost=0.2,
        )

        self.assertEqual(metrics["avg_exit"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_exit_controller_policy_distribution(self) -> None:
        controller = ExitController()
        state = torch.randn(2, 5)
        distribution = controller.policy(state)

        self.assertEqual(distribution.logits.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
