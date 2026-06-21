#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "outputs" / "runs" / "2026-06-20-deeper-experiments" / "metrics"
FIGURES_DIR = ROOT / "workspace" / "figures"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

METHOD_LABELS = {
    "accuracy_first_base_0.90_alpha_0.50": "Dynamic\n(acc-first)",
    "early_exit_final": "Early-exit\nfinal",
    "fixed_threshold_0.99": "Fixed\n$\\tau=0.99$",
    "full_resnet18": "Baseline",
    "learned_controller_best_reward_lambda_0.30": "Supervised\n$\\lambda=0.30$",
    "reinforce_controller_lambda_0.05": "REINFORCE\n$\\lambda=0.05$",
}

PANEL_TITLES = {
    "accuracy_first_base_0.90_alpha_0.50": "Dynamic (acc-first)",
    "early_exit_final": "Early-exit final",
    "fixed_threshold_0.99": "Fixed $\\tau=0.99$",
    "full_resnet18": "Baseline",
    "learned_controller_best_reward_lambda_0.30": "Supervised $\\lambda=0.30$",
    "reinforce_controller_lambda_0.05": "REINFORCE $\\lambda=0.05$",
}


def hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_json(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text())


def render_exit_distribution() -> None:
    data = load_json("per_exit_analysis.json")
    method_order = [
        "full_resnet18",
        "early_exit_final",
        "accuracy_first_base_0.90_alpha_0.50",
        "fixed_threshold_0.99",
        "learned_controller_best_reward_lambda_0.30",
        "reinforce_controller_lambda_0.05",
    ]
    exits = ["exit_1", "exit_2", "exit_3", "final_exit"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(method_order))
    x = np.arange(len(method_order))

    for exit_name, color in zip(exits, colors):
        values = []
        for method in method_order:
            method_data = data[method]
            if method == "full_resnet18" and exit_name == "final_exit":
                values.append(method_data["final"]["fraction"])
            else:
                values.append(method_data.get(exit_name, {}).get("fraction", 0.0))
        ax.bar(x, values, bottom=bottom, label=exit_name, color=color, edgecolor="white", linewidth=0.5)
        bottom += np.array(values)

    ax.set_title("Exit Distribution Comparison")
    ax.set_ylabel("Fraction of test samples")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in method_order])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exit_distribution_comparison.png", bbox_inches="tight")
    plt.close(fig)


def render_bootstrap_cis() -> None:
    data = load_json("bootstrap_statistics.json")
    method_order = [
        "full_resnet18",
        "early_exit_final",
        "fixed_threshold_0.99",
        "accuracy_first_base_0.90_alpha_0.50",
        "learned_controller_best_reward_lambda_0.30",
        "reinforce_controller_lambda_0.05",
    ]

    means = [data[m]["accuracy"]["mean"] for m in method_order]
    low = [data[m]["accuracy"]["mean"] - data[m]["accuracy"]["ci95_low"] for m in method_order]
    high = [data[m]["accuracy"]["ci95_high"] - data[m]["accuracy"]["mean"] for m in method_order]
    errors = np.array([low, high])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(method_order))
    ax.bar(x, means, color="#4C78A8", width=0.7)
    ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="black", elinewidth=1, capsize=4)
    ax.set_title("Bootstrap Accuracy Confidence Intervals")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in method_order])
    ax.set_ylim(0.68, 0.90)
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bootstrap_confidence_intervals.png", bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrices() -> None:
    data = load_json("classification_metrics.json")
    method_order = [
        "full_resnet18",
        "early_exit_final",
        "fixed_threshold_0.99",
        "accuracy_first_base_0.90_alpha_0.50",
        "learned_controller_best_reward_lambda_0.30",
        "reinforce_controller_lambda_0.05",
    ]

    matrices = [np.array(data[m]["confusion_matrix"]) for m in method_order]
    vmax = max(mat.max() for mat in matrices)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, method, matrix in zip(axes.flat, method_order, matrices):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(PANEL_TITLES[method])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(0, 10, 2))
        ax.set_yticks(range(0, 10, 2))
        hide_spines(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_best_methods.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render_exit_distribution()
    render_bootstrap_cis()
    render_confusion_matrices()


if __name__ == "__main__":
    main()
