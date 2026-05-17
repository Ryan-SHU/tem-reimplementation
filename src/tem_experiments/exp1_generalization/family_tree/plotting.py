"""Plotting for Family Tree experiment. Same structure as Line TI."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_family_tree_results(
    train_history: List[Dict[str, float]],
    eval_history: List[Dict[str, float]],
    output_dir: Path,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Training loss
    p = output_dir / "training_loss.png"
    steps = [h["step"] for h in train_history]
    plt.figure(figsize=(8, 5))
    for k in ["loss_total", "loss_x_p", "loss_x_g", "loss_x_gt"]:
        if k in train_history[0]:
            plt.plot(steps, [h[k] for h in train_history], label=k)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Family Tree — Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)

    # Zero-shot accuracy
    if len(eval_history) > 0:
        p = output_dir / "zero_shot_accuracy.png"
        es = [h["step"] for h in eval_history]
        plt.figure(figsize=(8, 5))
        plt.plot(es, [h["zero_shot_accuracy"] for h in eval_history], "o-", label="Zero-shot")
        plt.plot(es, [h["seen_accuracy"] for h in eval_history], "s-", label="Seen")
        plt.axhline(1.0 / 45, color="gray", ls="--", label="Chance")
        plt.ylim(-0.02, 1.02)
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        plt.title("Family Tree — Zero-shot Generalization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        paths.append(p)

    # Summary
    for ext in ["png", "pdf"]:
        p = output_dir / f"summary_figure.{ext}"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for k in ["loss_total", "loss_x_g"]:
            if k in train_history[0]:
                ax.plot(steps, [h[k] for h in train_history], label=k)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("A. Training Losses")
        ax.legend(fontsize=8)

        ax = axes[1]
        if len(eval_history) > 0:
            ax.plot(es, [h["zero_shot_accuracy"] for h in eval_history], "o-", label="Zero-shot")
            ax.plot(es, [h["seen_accuracy"] for h in eval_history], "s-", label="Seen")
            ax.axhline(1.0 / 45, color="gray", ls="--", label="Chance")
            ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("B. Zero-shot Generalization")
        ax.legend(fontsize=8)

        fig.suptitle("Experiment 1B: Family Tree", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(p, dpi=250)
        plt.close(fig)
        paths.append(p)

    return paths
