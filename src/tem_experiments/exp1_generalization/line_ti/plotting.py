"""
Plotting for Line TI experiment.

Generates:
    training_loss.png
    zero_shot_accuracy.png
    summary_figure.png / .pdf
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_line_ti_results(
    train_history: List[Dict[str, float]],
    eval_history: List[Dict[str, float]],
    output_dir: Path,
) -> List[Path]:
    """Generate all Line TI diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # ---- Training loss ----
    p = output_dir / "training_loss.png"
    _plot_training_loss(train_history, p)
    paths.append(p)

    # ---- Zero-shot accuracy ----
    if len(eval_history) > 0:
        p = output_dir / "zero_shot_accuracy.png"
        _plot_zero_shot_accuracy(eval_history, p)
        paths.append(p)

    # ---- Summary figure ----
    for ext in ["png", "pdf"]:
        p = output_dir / f"summary_figure.{ext}"
        _plot_summary(train_history, eval_history, p)
        paths.append(p)

    return paths


def _plot_training_loss(history: List[Dict], path: Path) -> None:
    steps = [h["step"] for h in history]
    keys = ["loss_total", "loss_x_p", "loss_x_g", "loss_x_gt"]
    plt.figure(figsize=(8, 5))
    for k in keys:
        if k in history[0]:
            plt.plot(steps, [h[k] for h in history], label=k)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Line TI — Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_zero_shot_accuracy(eval_history: List[Dict], path: Path) -> None:
    steps = [h["step"] for h in eval_history]
    zs = [h["zero_shot_accuracy"] for h in eval_history]
    seen = [h["seen_accuracy"] for h in eval_history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, zs, "o-", label="Zero-shot accuracy")
    plt.plot(steps, seen, "s-", label="Seen-edge accuracy")

    # chance line
    chance = 1.0 / 45  # roughly 1/N_x
    plt.axhline(chance, color="gray", ls="--", label=f"Chance ({chance:.3f})")

    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("Line TI — Zero-shot Structural Generalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_summary(
    train_history: List[Dict],
    eval_history: List[Dict],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # A: training loss
    ax = axes[0]
    steps = [h["step"] for h in train_history]
    for k in ["loss_total", "loss_x_g", "loss_x_gt"]:
        if k in train_history[0]:
            ax.plot(steps, [h[k] for h in train_history], label=k)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("A. Training Losses")
    ax.legend(fontsize=8)

    # B: zero-shot accuracy
    ax = axes[1]
    if len(eval_history) > 0:
        es = [h["step"] for h in eval_history]
        ax.plot(es, [h["zero_shot_accuracy"] for h in eval_history], "o-", label="Zero-shot")
        ax.plot(es, [h["seen_accuracy"] for h in eval_history], "s-", label="Seen")
        ax.axhline(1.0 / 45, color="gray", ls="--", label="Chance")
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("B. Zero-shot Generalization")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 1A: Transitive Inference (Line Graph)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=250)
    plt.close(fig)
