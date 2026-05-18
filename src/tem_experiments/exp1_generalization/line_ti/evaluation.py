"""
Offline checkpoint evaluation for Experiment 1A: Line TI.

This file is evaluation-only.

Responsibilities
----------------
- load saved checkpoints
- run fresh-world continuous-walk evaluation
- save raw event rows
- save checkpoint-level summary CSV
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tem.config import TEMConfig
from tem.models.tem import TEM
from tem_data.envs.line import LineEnvironment
from tem_training.checkpointing import load_checkpoint

from tem_experiments.exp1_generalization.line_ti.definitions import (
    EVAL_EPISODES,
    EVAL_WALK_LEN,
    NUM_NODES,
)
from tem_experiments.exp1_generalization.line_ti.helpers import (
    collect_evaluation_events,
    summarize_events,
    write_event_rows_csv,
)


SUMMARY_FIELDNAMES = [
    "checkpoint_step",
    "inferable_unseen_accuracy",
    "inferable_unseen_correct",
    "inferable_unseen_total",
    "seen_accuracy",
    "seen_correct",
    "seen_total",
    "first_time_edge_accuracy",
    "first_time_edge_correct",
    "first_time_edge_total",
    "walk_len",
    "eval_episodes",
    "event_file",
]


@dataclass
class LineTIEvalResult:
    summary_rows: List[Dict[str, float | int | str]]
    summary_csv_path: Path
    event_files: List[Path]


def _discover_checkpoint_paths(
    checkpoint_dir: Path,
    selector: str = "all",
    checkpoint_steps: Optional[List[int]] = None,
) -> List[Path]:
    checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_steps is not None and len(checkpoint_steps) > 0:
        paths = [
            checkpoint_dir / f"step_{step:07d}.pt"
            for step in checkpoint_steps
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        return paths

    if selector == "latest":
        path = checkpoint_dir / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return [path]

    if selector != "all":
        raise ValueError(
            f"selector must be 'all' or 'latest', got {selector!r}."
        )

    paths = sorted(checkpoint_dir.glob("step_*.pt"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {checkpoint_dir}")
    return paths


def _write_summary_csv(
    path: Path,
    rows: List[Dict[str, float | int | str]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_line_ti_checkpoints(
    config: TEMConfig,
    run_dir: Path,
    device: torch.device,
    selector: str = "all",
    checkpoint_steps: Optional[List[int]] = None,
    eval_episodes: int = EVAL_EPISODES,
    walk_len: int = EVAL_WALK_LEN,
    seed: int = 0,
) -> LineTIEvalResult:
    """
    Offline evaluation of one or more saved checkpoints.
    """
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"

    eval_dir = run_dir / "eval"
    events_dir = eval_dir / "events"
    summary_csv_path = eval_dir / "eval_summary.csv"

    events_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = _discover_checkpoint_paths(
        checkpoint_dir=checkpoint_dir,
        selector=selector,
        checkpoint_steps=checkpoint_steps,
    )

    summary_rows: List[Dict[str, float | int | str]] = []
    event_files: List[Path] = []

    print(f"Evaluating {len(ckpt_paths)} checkpoint(s) from {checkpoint_dir}")

    for ckpt_path in ckpt_paths:
        model = TEM(config).to(device)
        ckpt = load_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=None,
            map_location=device,
        )
        checkpoint_step = int(ckpt["step"])

        env = LineEnvironment(
            num_nodes=NUM_NODES,
            num_observations=config.data.num_observations,
        )

        eval_gen = torch.Generator()
        eval_gen.manual_seed(seed * 100000 + checkpoint_step)

        rows = collect_evaluation_events(
            model=model,
            env=env,
            walk_len=walk_len,
            num_episodes=eval_episodes,
            device=device,
            generator=eval_gen,
            checkpoint_step=checkpoint_step,
        )

        event_file = events_dir / f"eval_events_step_{checkpoint_step:07d}.csv"
        write_event_rows_csv(event_file, rows)
        event_files.append(event_file)

        summary = summarize_events(rows)
        summary_row: Dict[str, float | int | str] = {
            "checkpoint_step": checkpoint_step,
            "inferable_unseen_accuracy": summary["inferable_unseen_accuracy"],
            "inferable_unseen_correct": summary["inferable_unseen_correct"],
            "inferable_unseen_total": summary["inferable_unseen_total"],
            "seen_accuracy": summary["seen_accuracy"],
            "seen_correct": summary["seen_correct"],
            "seen_total": summary["seen_total"],
            "first_time_edge_accuracy": summary["first_time_edge_accuracy"],
            "first_time_edge_correct": summary["first_time_edge_correct"],
            "first_time_edge_total": summary["first_time_edge_total"],
            "walk_len": walk_len,
            "eval_episodes": eval_episodes,
            "event_file": event_file.name,
        }
        summary_rows.append(summary_row)

        print(
            f"  step={checkpoint_step:7d} | "
            f"inferable_unseen={summary_row['inferable_unseen_accuracy']:.4f} "
            f"({summary_row['inferable_unseen_correct']}/{summary_row['inferable_unseen_total']}) | "
            f"seen={summary_row['seen_accuracy']:.4f} "
            f"({summary_row['seen_correct']}/{summary_row['seen_total']})"
        )

    summary_rows.sort(key=lambda row: int(row["checkpoint_step"]))
    _write_summary_csv(summary_csv_path, summary_rows)

    print(f"Saved eval summary to: {summary_csv_path}")

    return LineTIEvalResult(
        summary_rows=summary_rows,
        summary_csv_path=summary_csv_path,
        event_files=event_files,
    )
