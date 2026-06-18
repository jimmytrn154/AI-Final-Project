from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .controller import build_reinforce_state, compute_reinforce_reward, evaluate_reinforce_controller
from .evaluation import evaluate_classifier, evaluate_early_exit_model


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    model.to(device)
    training_config = config["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.get("label_smoothing", 0.0))
    best_accuracy = 0.0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config["epochs"]):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for images, labels in tqdm(train_loader, desc=f"baseline epoch {epoch + 1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        train_loss = running_loss / max(total_examples, 1)
        val_metrics = evaluate_classifier(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        if val_metrics["accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    return {"best_val_accuracy": best_accuracy, "history": history}


def train_early_exit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    model.to(device)
    training_config = config["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    loss_weights = training_config["loss_weights"]
    best_accuracy = 0.0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config["epochs"]):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for images, labels in tqdm(train_loader, desc=f"early-exit epoch {epoch + 1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits_list = model(images)
            losses = [
                weight * nn.functional.cross_entropy(logits, labels)
                for weight, logits in zip(loss_weights, logits_list)
            ]
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        train_loss = running_loss / max(total_examples, 1)
        val_metrics = evaluate_early_exit_model(model, val_loader, device, loss_weights)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "per_exit_accuracy": val_metrics["per_exit_accuracy"],
                "final_accuracy": val_metrics["final_accuracy"],
            }
        )

        if val_metrics["final_accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["final_accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    return {"best_val_accuracy": best_accuracy, "history": history}


def train_controller(
    controller: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    controller.to(device)
    training_config = config["training"]
    optimizer = optim.Adam(
        controller.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config["epochs"]):
        controller.train()
        running_loss = 0.0
        total_examples = 0

        for features, labels in tqdm(train_loader, desc=f"controller epoch {epoch + 1}", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = controller(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        controller.eval()
        correct = 0
        val_examples = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = controller(features).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                val_examples += labels.size(0)

        val_accuracy = correct / max(val_examples, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(total_examples, 1),
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(controller.state_dict(), checkpoint_path)

    return {"best_val_accuracy": best_accuracy, "history": history}


def train_reinforce_controller(
    controller: torch.nn.Module,
    early_exit_model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    last_checkpoint_path: str | Path,
) -> dict[str, Any]:
    controller.to(device)
    early_exit_model.to(device)
    early_exit_model.eval()
    for parameter in early_exit_model.parameters():
        parameter.requires_grad = False

    training_config = config["training"]
    optimizer = optim.Adam(
        controller.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    best_reward = float("-inf")
    history = []
    reward_baseline = 0.0
    entropy_beta = config["entropy_beta"]
    baseline_momentum = config["baseline_momentum"]
    gradient_clip_norm = config["gradient_clip_norm"]
    lambda_cost = config["lambda_cost"]
    wrong_penalty = config.get("wrong_penalty", 0.0)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = Path(last_checkpoint_path)
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config["epochs"]):
        controller.train()
        policy_loss_sum = 0.0
        reward_sum = 0.0
        entropy_sum = 0.0
        batch_count = 0

        for images, labels in tqdm(train_loader, desc=f"reinforce epoch {epoch + 1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits_list = early_exit_model.forward_all_exits(images)

            sample_losses = []
            sample_rewards = []
            sample_entropies = []

            for sample_index in range(images.size(0)):
                trajectory_log_probs = []
                trajectory_entropies = []
                selected_logits = logits_list[-1][sample_index : sample_index + 1]
                selected_flops_ratio = early_exit_model.flops_ratios[-1]

                for exit_index, batch_logits in enumerate(logits_list):
                    logits = batch_logits[sample_index : sample_index + 1]
                    is_final_exit = exit_index == len(logits_list) - 1

                    if is_final_exit:
                        selected_logits = logits
                        selected_flops_ratio = early_exit_model.flops_ratios[exit_index]
                        break

                    state = build_reinforce_state(
                        logits=logits,
                        exit_index=exit_index,
                        flops_ratio=early_exit_model.flops_ratios[exit_index],
                        num_exits=len(logits_list),
                    )
                    distribution = controller.policy(state)
                    action = distribution.sample()
                    trajectory_log_probs.append(distribution.log_prob(action).squeeze(0))
                    trajectory_entropies.append(distribution.entropy().squeeze(0))

                    if action.item() == 1:
                        selected_logits = logits
                        selected_flops_ratio = early_exit_model.flops_ratios[exit_index]
                        break

                reward = compute_reinforce_reward(
                    logits=selected_logits,
                    labels=labels[sample_index : sample_index + 1],
                    flops_ratio=selected_flops_ratio,
                    lambda_cost=lambda_cost,
                    wrong_penalty=wrong_penalty,
                ).squeeze(0)
                sample_rewards.append(reward)

                if trajectory_log_probs:
                    log_prob_sum = torch.stack(trajectory_log_probs).sum()
                    entropy_value = torch.stack(trajectory_entropies).sum()
                else:
                    log_prob_sum = torch.zeros((), device=device)
                    entropy_value = torch.zeros((), device=device)

                advantage = (reward - reward_baseline).detach()
                sample_losses.append(-advantage * log_prob_sum - entropy_beta * entropy_value)
                sample_entropies.append(entropy_value.detach())

            rewards_tensor = torch.stack(sample_rewards)
            entropies_tensor = torch.stack(sample_entropies) if sample_entropies else torch.zeros(1, device=device)
            policy_loss = torch.stack(sample_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

            batch_mean_reward = rewards_tensor.mean().item()
            reward_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * batch_mean_reward

            policy_loss_sum += policy_loss.item()
            reward_sum += batch_mean_reward
            entropy_sum += entropies_tensor.mean().item()
            batch_count += 1

        val_metrics = evaluate_reinforce_controller(
            early_exit_model,
            controller,
            val_loader,
            device,
            lambda_cost=lambda_cost,
            wrong_penalty=wrong_penalty,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_reward": reward_sum / max(batch_count, 1),
                "val_reward": val_metrics["average_reward"],
                "policy_loss": policy_loss_sum / max(batch_count, 1),
                "policy_entropy": entropy_sum / max(batch_count, 1),
                "val_accuracy": val_metrics["accuracy"],
                "average_flops_ratio": val_metrics["average_flops_ratio"],
                "exit_1_rate": val_metrics["exit_distribution"]["exit_1"],
                "exit_2_rate": val_metrics["exit_distribution"]["exit_2"],
                "exit_3_rate": val_metrics["exit_distribution"]["exit_3"],
                "final_exit_rate": val_metrics["exit_distribution"]["final_exit"],
            }
        )

        torch.save(controller.state_dict(), last_checkpoint_path)
        if val_metrics["average_reward"] >= best_reward:
            best_reward = val_metrics["average_reward"]
            torch.save(controller.state_dict(), checkpoint_path)

    return {"best_val_reward": best_reward, "history": history}
