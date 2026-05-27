from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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
