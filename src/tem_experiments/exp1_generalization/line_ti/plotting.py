"""
Offline plotting for Experiment 1A: Line TI.

This file reads saved CSV files and generates plots.

Plot categories
---------------
1. Diagnostic plots
   - training_losses.png
   - diagnostic_accuracy_vs_checkpoint.png

2. Paper-style plots
   - figure3_linked_observations.png
   - figure3_nodes_visited_count.png
   - figure3_nodes_visited_fraction.png
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_train_metrics_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _read_eval_summary_csv(path: Path) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float | str] = {}
            for k, v in row.items():
                if k == "event_file":
                    parsed[k] = v
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _read_event_rows_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _load_events_by_checkpoint(event_dir: Path) -> Dict[int, List[Dict[str, float]]]:
    event_dir = Path(event_dir)
    files = sorted(event_dir.glob("eval_events_step_*.csv"))

    by_step: Dict[int, List[Dict[str, float]]] = {}
    for file in files:
        rows = _read_event_rows_csv(file)
        if len(rows) == 0:
            continue
        step = int(rows[0]["checkpoint_step"])
        by_step[step] = rows
    return by_step


def _group_steps_into_quantiles(
    steps: List[int],
    num_quantiles: int,
) -> Dict[int, List[int]]:
    if len(steps) == 0:
        return {}

    steps = sorted(steps)
    n = len(steps)

    groups: Dict[int, List[int]] = {q: [] for q in range(num_quantiles)}
    for idx, step in enumerate(steps):
        q = min(num_quantiles - 1, idx * num_quantiles // n)
        groups[q].append(step)

    return {q: s for q, s in groups.items() if len(s) > 0}


def _aggregate_accuracy_by_x(
    rows: List[Dict[str, float]],
    x_key: str,
    event_filter_key: str = "inferable_unseen_link",
) -> Tuple[List[float], List[float], List[int]]:
    """
    Aggregate accuracy by exact x-values.

    For Line TI this works well because coverage values are discrete.
    """
    bucket: Dict[float, List[int]] = {}

    for row in rows:
        if int(row[event_filter_key]) != 1:
            continue

        x = float(row[x_key])
        correct = int(row["correct"])

        if x not in bucket:
            bucket[x] = [0, 0]  # correct_sum, total_count

        bucket[x][0] += correct
        bucket[x][1] += 1

    xs = sorted(bucket.keys())
    ys = [bucket[x][0] / max(bucket[x][1], 1) for x in xs]
    ns = [bucket[x][1] for x in xs]

    return xs, ys, ns


def _format_quantile_label(q: int, steps: List[int]) -> str:
    return f"Q{q + 1} ({min(steps)}–{max(steps)})"


def plot_training_losses_from_csv(
    train_metrics_csv: Path,
    output_path: Path,
) -> None:
    rows = _read_train_metrics_csv(train_metrics_csv)
    if len(rows) == 0:
        return

    steps = [r["step"] for r in rows]

    plt.figure(figsize=(9, 5))
    for key in ["loss_total", "loss_x_p", "loss_x_g", "loss_x_gt"]:
        if key in rows[0]:
            plt.plot(steps, [r[key] for r in rows], label=key)

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Line TI — Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_diagnostic_accuracy_from_summary(
    eval_summary_csv: Path,
    output_path: Path,
    chance: float = 1.0 / 45.0,
) -> None:
    rows = _read_eval_summary_csv(eval_summary_csv)
    if len(rows) == 0:
        return

    rows.sort(key=lambda r: float(r["checkpoint_step"]))
    steps = [float(r["checkpoint_step"]) for r in rows]
    inferable = [float(r["inferable_unseen_accuracy"]) for r in rows]
    seen = [float(r["seen_accuracy"]) for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, inferable, "o-", label="Inferable unseen link accuracy")
    plt.plot(steps, seen, "s-", label="Seen-edge accuracy")
    plt.axhline(chance, color="gray", ls="--", label=f"Chance ({chance:.3f})")
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("Line TI — Diagnostic Accuracy vs. Checkpoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_figure3_style_quantiles(
    event_dir: Path,
    output_path: Path,
    x_key: str,
    x_label: str,
    title: str,
    num_training_quantiles: int = 5,
    chance: float = 1.0 / 45.0,
) -> None:
    """
    Paper-style plot:
        x = exploration coverage
        y = correct inference of link
        different lines = different training quantiles
    """
    events_by_step = _load_events_by_checkpoint(event_dir)
    steps = sorted(events_by_step.keys())
    if len(steps) == 0:
        return

    quantile_groups = _group_steps_into_quantiles(steps, num_training_quantiles)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy vs coverage
    ax = axes[0]
    for q, q_steps in quantile_groups.items():
        merged_rows: List[Dict[str, float]] = []
        for step in q_steps:
            merged_rows.extend(events_by_step[step])

        xs, ys, ns = _aggregate_accuracy_by_x(
            merged_rows,
            x_key=x_key,
            event_filter_key="inferable_unseen_link",
        )
        if len(xs) == 0:
            continue

        ax.plot(
            xs, ys,
            marker="o",
            linewidth=2,
            label=_format_quantile_label(q, q_steps),
        )

    ax.axhline(chance, color="gray", ls="--", label=f"Chance ({chance:.3f})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Correct inference of link")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Correct inference of link")
    ax.legend(fontsize=8)

    # Right: counts
    ax = axes[1]
    for q, q_steps in quantile_groups.items():
        merged_rows: List[Dict[str, float]] = []
        for step in q_steps:
            merged_rows.extend(events_by_step[step])

        xs, ys, ns = _aggregate_accuracy_by_x(
            merged_rows,
            x_key=x_key,
            event_filter_key="inferable_unseen_link",
        )
        if len(xs) == 0:
            continue

        ax.plot(
            xs, ns,
            marker="o",
            linewidth=2,
            label=_format_quantile_label(q, q_steps),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of inferable test events")
    ax.set_title("B. Sample count")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def make_all_line_ti_plots(
    run_dir: Path,
    num_training_quantiles: int = 5,
) -> List[Path]:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    train_metrics_csv = run_dir / "train_metrics.csv"
    eval_summary_csv = run_dir / "eval" / "eval_summary.csv"
    event_dir = run_dir / "eval" / "events"

    if train_metrics_csv.exists():
        p = plot_dir / "training_losses.png"
        plot_training_losses_from_csv(train_metrics_csv, p)
        paths.append(p)

    if eval_summary_csv.exists():
        p = plot_dir / "diagnostic_accuracy_vs_checkpoint.png"
        plot_diagnostic_accuracy_from_summary(eval_summary_csv, p)
        paths.append(p)

    if event_dir.exists():
        # 1) number of linked observations
        p = plot_dir / "figure3_linked_observations.png"
        plot_figure3_style_quantiles(
            event_dir=event_dir,
            output_path=p,
            x_key="unique_edges_seen_count_before_t",
            x_label="Number of linked observations",
            title="Exp 1A: Line TI — Correct inference vs. linked observations",
            num_training_quantiles=num_training_quantiles,
        )
        paths.append(p)

        # 2) number of nodes visited
        p = plot_dir / "figure3_nodes_visited_count.png"
        plot_figure3_style_quantiles(
            event_dir=event_dir,
            output_path=p,
            x_key="nodes_visited_count_before_t",
            x_label="Number of nodes visited",
            title="Exp 1A: Line TI — Correct inference vs. nodes visited",
            num_training_quantiles=num_training_quantiles,
        )
        paths.append(p)

        # 3) proportion of nodes visited
        p = plot_dir / "figure3_nodes_visited_fraction.png"
        plot_figure3_style_quantiles(
            event_dir=event_dir,
            output_path=p,
            x_key="nodes_visited_fraction_before_t",
            x_label="Proportion of nodes visited",
            title="Exp 1A: Line TI — Correct inference vs. proportion of nodes visited",
            num_training_quantiles=num_training_quantiles,
        )
        paths.append(p)

    return paths
