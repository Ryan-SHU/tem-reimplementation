"""
Offline plotting for Experiment 1A: Line TI.

This file only reads saved CSV files and generates plots.

Main paper-style Figure 3A logic
--------------------------------
The uploaded paper panel is:

    y = correct inference of link
    x = # times node visited
    color = training quantile

The crucial detail:

    filter = first_time_edge

not:

    filter = inferable_unseen_link

For Figure 3A, x = number of times the target node had been visited BEFORE
the current transition. Therefore x starts at 0.

A row contributes to the Figure 3A curve when:
    edge_visit_count_before_t == 0

Then it is binned by:
    target_visit_count_before_t

This file is backward-compatible with old eval event CSVs. If the CSV does
not contain target_visit_count_before_t or edge_visit_count_before_t, they
are reconstructed from episode_id, t, state_prev, action_id, and state_cur.
"""

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

from tem_experiments.exp1_generalization.line_ti.definitions import (
    NUM_NODES,
    NUM_OBSERVATIONS,
)


# These are the legend bins shown in the paper-style panel.
# They are fractions of training progress: checkpoint_step / max_checkpoint_step.
PAPER_TRAINING_FRACTION_RANGES: List[Tuple[float, float]] = [
    (0.0, 0.1),
    (0.1, 0.14),
    (0.15, 0.19),
    (0.2, 0.29),
    (0.4, 0.5),
    (0.9, 1.0),
]

PAPER_COLORS = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#CCB974",  # yellow/brown
    "#64B5CD",  # cyan
]


def _read_train_metrics_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)

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
                elif v is None or v == "":
                    parsed[k] = 0.0
                else:
                    parsed[k] = float(v)
            rows.append(parsed)

    return rows


def _read_event_rows_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = 0.0
                else:
                    parsed[k] = float(v)
            rows.append(parsed)

    return _reconstruct_event_fields(rows)


def _reconstruct_event_fields(
    rows: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """
    Reconstruct paper-analysis fields from saved event rows.

    The saved rows already contain:
        checkpoint_step, episode_id, t, state_prev, action_id, state_cur, correct

    We reconstruct, for each episode:
        target_visit_count_before_t
        edge_visit_count_before_t
        first_time_edge
        inferable_unseen_link
        nodes_visited_count_before_t
        nodes_visited_fraction_before_t
        unique_edges_seen_count_before_t

    Important:
        For Figure 3A, the main filter is first_time_edge.
        The x-axis is target_visit_count_before_t.
    """
    if len(rows) == 0:
        return rows

    groups: Dict[Tuple[int, int], List[Dict[str, float]]] = {}

    for row in rows:
        checkpoint_step = int(row["checkpoint_step"])
        episode_id = int(row["episode_id"])
        groups.setdefault((checkpoint_step, episode_id), []).append(row)

    rebuilt: List[Dict[str, float]] = []

    for _, episode_rows in groups.items():
        episode_rows.sort(key=lambda r: int(r["t"]))

        if len(episode_rows) == 0:
            continue

        # The first row represents transition s_0 -> s_1.
        # Therefore state_prev of the first row is s_0 and has already been visited once.
        initial_state = int(episode_rows[0]["state_prev"])

        state_visit_counts: Dict[int, int] = {initial_state: 1}

        # Match the original behavioural analysis:
        # edge identity is directed state transition (first, second).
        # For Line TI, action is redundant given state_prev and state_cur.
        edge_visit_counts: Dict[Tuple[int, int], int] = {}

        for row in episode_rows:
            s_prev = int(row["state_prev"])
            s_cur = int(row["state_cur"])

            edge = (s_prev, s_cur)

            source_visit_count_before_t = state_visit_counts.get(s_prev, 0)
            target_visit_count_before_t = state_visit_counts.get(s_cur, 0)

            edge_visit_count_before_t = edge_visit_counts.get(edge, 0)

            first_time_edge = int(edge_visit_count_before_t == 0)
            edge_seen_before_t = int(edge_visit_count_before_t > 0)
            target_seen_before_t = int(target_visit_count_before_t > 0)

            inferable_unseen_link = int(
                first_time_edge == 1 and target_seen_before_t == 1
            )

            nodes_visited_count_before_t = len(state_visit_counts)
            nodes_visited_fraction_before_t = nodes_visited_count_before_t / NUM_NODES
            unique_edges_seen_count_before_t = len(edge_visit_counts)

            row["source_visit_count_before_t"] = float(source_visit_count_before_t)
            row["target_visit_count_before_t"] = float(target_visit_count_before_t)
            row["edge_visit_count_before_t"] = float(edge_visit_count_before_t)
            row["first_time_edge"] = float(first_time_edge)
            row["edge_seen_before_t"] = float(edge_seen_before_t)
            row["target_seen_before_t"] = float(target_seen_before_t)
            row["inferable_unseen_link"] = float(inferable_unseen_link)
            row["nodes_visited_count_before_t"] = float(nodes_visited_count_before_t)
            row["nodes_visited_fraction_before_t"] = float(
                nodes_visited_fraction_before_t
            )
            row["unique_edges_seen_count_before_t"] = float(
                unique_edges_seen_count_before_t
            )

            rebuilt.append(row)

            edge_visit_counts[edge] = edge_visit_count_before_t + 1
            state_visit_counts[s_cur] = target_visit_count_before_t + 1

    rebuilt.sort(
        key=lambda r: (
            int(r["checkpoint_step"]),
            int(r["episode_id"]),
            int(r["t"]),
        )
    )
    return rebuilt


def _load_events_by_checkpoint(event_dir: Path) -> Dict[int, List[Dict[str, float]]]:
    event_dir = Path(event_dir)
    files = sorted(event_dir.glob("eval_events_step_*.csv"))

    by_step: Dict[int, List[Dict[str, float]]] = {}

    for file in files:
        rows = _read_event_rows_csv(file)
        if len(rows) == 0:
            continue

        checkpoint_step = int(rows[0]["checkpoint_step"])
        by_step[checkpoint_step] = rows

    return by_step


def _group_steps_by_paper_training_ranges(
    steps: List[int],
    ranges: List[Tuple[float, float]] = PAPER_TRAINING_FRACTION_RANGES,
) -> List[Tuple[str, List[int]]]:
    """
    Group checkpoints into the same style of training-fraction ranges as the paper.

    step_fraction = checkpoint_step / max_checkpoint_step

    Ranges are half-open [lo, hi), except the last range includes hi.
    """
    steps = sorted(steps)
    if len(steps) == 0:
        return []

    max_step = max(steps)
    groups: List[Tuple[str, List[int]]] = []

    for idx, (lo, hi) in enumerate(ranges):
        selected: List[int] = []

        for step in steps:
            frac = step / max_step

            if idx == len(ranges) - 1:
                in_range = lo <= frac <= hi
            else:
                in_range = lo <= frac < hi

            if in_range:
                selected.append(step)

        if len(selected) > 0:
            label = f"{lo:g}-{hi:g}"
            groups.append((label, selected))

    return groups


def _aggregate_accuracy_by_x(
    rows: Iterable[Dict[str, float]],
    x_key: str,
    filter_key: str,
    max_x: Optional[int] = None,
    min_count_per_x: int = 1,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    Aggregate binary correctness by exact x value.

    Returns:
        xs
        mean accuracy
        standard error
        counts
    """
    buckets: Dict[float, List[int]] = {}

    for row in rows:
        if int(row.get(filter_key, 0.0)) != 1:
            continue

        x = float(row[x_key])

        if max_x is not None and x > max_x:
            continue

        correct = int(row["correct"])

        if x not in buckets:
            buckets[x] = [0, 0]

        buckets[x][0] += correct
        buckets[x][1] += 1

    xs: List[float] = []
    ys: List[float] = []
    sems: List[float] = []
    ns: List[int] = []

    for x in sorted(buckets.keys()):
        correct_sum, total = buckets[x]
        if total < min_count_per_x:
            continue

        p = correct_sum / total
        sem = math.sqrt(max(p * (1.0 - p), 0.0) / total)

        xs.append(x)
        ys.append(p)
        sems.append(sem)
        ns.append(total)

    return xs, ys, sems, ns


def plot_training_losses_from_csv(
    train_metrics_csv: Path,
    output_path: Path,
) -> None:
    rows = _read_train_metrics_csv(train_metrics_csv)
    if len(rows) == 0:
        return

    steps = [row["step"] for row in rows]

    plt.figure(figsize=(9, 5))

    for key in ["loss_total", "loss_x_p", "loss_x_g", "loss_x_gt"]:
        if key in rows[0]:
            plt.plot(steps, [row[key] for row in rows], label=key)

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Line TI — Training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_diagnostic_accuracy_from_summary(
    eval_summary_csv: Path,
    output_path: Path,
    chance: float = 1.0 / NUM_OBSERVATIONS,
) -> None:
    rows = _read_eval_summary_csv(eval_summary_csv)
    if len(rows) == 0:
        return

    rows.sort(key=lambda row: float(row["checkpoint_step"]))

    steps = [float(row["checkpoint_step"]) for row in rows]
    inferable = [float(row["inferable_unseen_accuracy"]) for row in rows]
    seen = [float(row["seen_accuracy"]) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, inferable, "o-", label="Inferable unseen link")
    plt.plot(steps, seen, "s-", label="Seen edge")
    plt.axhline(chance, color="black", linestyle="--", label=f"Chance ({chance:.3f})")

    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("Line TI — Diagnostic accuracy vs. checkpoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_figure3a_times_node_visited(
    event_dir: Path,
    output_path: Path,
    max_times_node_visited: int = 9,
    min_count_per_x: int = 1,
    chance: float = 1.0 / NUM_OBSERVATIONS,
) -> None:
    """
    Reproduce the structure of paper Figure 3A.

    y:
        correct inference of link

    x:
        # times node visited

    event filter:
        first_time_edge == 1

    This is intentionally NOT filtering to inferable_unseen_link, because the
    original panel includes x=0, where the target node has not been visited before.
    """
    events_by_step = _load_events_by_checkpoint(event_dir)
    steps = sorted(events_by_step.keys())

    if len(steps) == 0:
        return

    training_groups = _group_steps_by_paper_training_ranges(steps)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for idx, (label, group_steps) in enumerate(training_groups):
        merged_rows: List[Dict[str, float]] = []

        for step in group_steps:
            merged_rows.extend(events_by_step[step])

        xs, ys, sems, ns = _aggregate_accuracy_by_x(
            rows=merged_rows,
            x_key="target_visit_count_before_t",
            filter_key="first_time_edge",
            max_x=max_times_node_visited,
            min_count_per_x=min_count_per_x,
        )

        if len(xs) == 0:
            continue

        color = PAPER_COLORS[idx % len(PAPER_COLORS)]

        lower = [max(0.0, y - sem) for y, sem in zip(ys, sems)]
        upper = [min(1.0, y + sem) for y, sem in zip(ys, sems)]

        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            color=color,
            label=label,
        )
        ax.fill_between(
            xs,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.axhline(chance, color="black", linestyle="--", linewidth=2)

    ax.set_xlabel("# times node visited")
    ax.set_ylabel("Correct inference of link")
    ax.set_xlim(0, max_times_node_visited)
    ax.set_ylim(0.0, 1.02)

    ax.legend(
        title="Training\nquantile",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_figure3_proportion_nodes_visited(
    event_dir: Path,
    output_path: Path,
    min_count_per_x: int = 1,
    chance: float = 1.0 / NUM_OBSERVATIONS,
) -> None:
    """
    Secondary plot:
        y = correct inference of link
        x = proportion of nodes visited

    For this plot, the relevant event filter is inferable_unseen_link:
        first_time_edge == 1 and target_seen_before_t == 1
    """
    events_by_step = _load_events_by_checkpoint(event_dir)
    steps = sorted(events_by_step.keys())

    if len(steps) == 0:
        return

    training_groups = _group_steps_by_paper_training_ranges(steps)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for idx, (label, group_steps) in enumerate(training_groups):
        merged_rows: List[Dict[str, float]] = []

        for step in group_steps:
            merged_rows.extend(events_by_step[step])

        xs, ys, sems, ns = _aggregate_accuracy_by_x(
            rows=merged_rows,
            x_key="nodes_visited_fraction_before_t",
            filter_key="inferable_unseen_link",
            max_x=None,
            min_count_per_x=min_count_per_x,
        )

        if len(xs) == 0:
            continue

        color = PAPER_COLORS[idx % len(PAPER_COLORS)]

        lower = [max(0.0, y - sem) for y, sem in zip(ys, sems)]
        upper = [min(1.0, y + sem) for y, sem in zip(ys, sems)]

        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            color=color,
            label=label,
        )
        ax.fill_between(
            xs,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.axhline(chance, color="black", linestyle="--", linewidth=2)

    ax.set_xlabel("Proportion of nodes visited")
    ax.set_ylabel("Correct inference of link")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    ax.legend(
        title="Training\nquantile",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_binned_accuracy(
    bins: List[Dict],
    output_path: Path,
    title: str = "",
    chance: float = 1.0 / 45.0,
) -> None:
    """
    Generic binned accuracy plot used by family_tree and spatial_graph.

    Each bin is a dict with keys: target_visit_count_before_t, correct, total, accuracy.
    """
    if len(bins) == 0:
        return

    xs = [b["target_visit_count_before_t"] for b in bins]
    ys = [b["accuracy"] for b in bins]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "o-", linewidth=2, color="#4C72B0")
    plt.axhline(chance, color="black", linestyle="--", linewidth=1.5, label=f"Chance ({chance:.3f})")
    plt.xlabel("# times node visited before transition")
    plt.ylabel("Correct inference of link")
    if title:
        plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_all_line_ti_plots(
    run_dir: Path,
    num_training_quantiles: int = 5,
) -> List[Path]:
    """
    Generate all Line TI plots.

    num_training_quantiles is kept for API compatibility, but the paper-style
    Figure 3A uses the fixed paper-like training fraction ranges:
        0.0-0.1
        0.1-0.14
        0.15-0.19
        0.2-0.29
        0.4-0.5
        0.9-1.0
    """
    del num_training_quantiles

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
        p = plot_dir / "figure3a_correct_inference_times_node_visited.png"
        plot_figure3a_times_node_visited(
            event_dir=event_dir,
            output_path=p,
            max_times_node_visited=9,
            min_count_per_x=1,
        )
        paths.append(p)

        p = plot_dir / "figure3_correct_inference_proportion_nodes_visited.png"
        plot_figure3_proportion_nodes_visited(
            event_dir=event_dir,
            output_path=p,
            min_count_per_x=1,
        )
        paths.append(p)

    return paths
