"""
Helper functions for Experiment 1A (Line TI).

This file implements event-level continuous-walk evaluation.

For each checkpoint and each fresh world:
    1. resample sensory observations on the same graph
    2. run one continuous random walk
    3. record one event row for every transition t >= 1

The paper-style metric is:

    correct inference of link

Operationally, we define an inferable unseen link event as:

    edge_visit_count_before_t == 0
    and target_visit_count_before_t > 0

That is, the current transition/link has not been traversed before, but the
target node has already been observed in the current world.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from tem_data.base import GraphEnvironment
from tem_data.sampling.random_walk import generate_random_walk


EVENT_FIELDNAMES = [
    "checkpoint_step",
    "episode_id",
    "t",
    "state_prev",
    "action_id",
    "state_cur",
    "source_visit_count_before_t",
    "target_visit_count_before_t",
    "current_target_visit_index",
    "edge_visit_count_before_t",
    "current_edge_visit_index",
    "edge_seen_before_t",
    "first_time_edge",
    "target_seen_before_t",
    "inferable_unseen_link",
    "nodes_visited_count_before_t",
    "nodes_visited_fraction_before_t",
    "unique_edges_seen_count_before_t",
    "pred_obs_id",
    "true_obs_id",
    "correct",
]


def evaluate_continuous_single(
    model: torch.nn.Module,
    env: GraphEnvironment,
    walk_len: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    checkpoint_step: Optional[int] = None,
    episode_id: int = 0,
) -> List[Dict[str, int | float]]:
    """
    Evaluate one continuous walk in one fresh world.

    Returns
    -------
    List of event rows, one row per transition t >= 1.
    """
    model.eval()

    env.resample_observations(generator)

    walk = generate_random_walk(
        env=env,
        batch_size=1,
        seq_len=walk_len,
        generator=generator,
    )

    pos = walk["position"]
    act = walk["action"]
    vis = walk["visited"]
    obs = env.get_observation_ids(pos)

    x = F.one_hot(obs, env.num_observations).float().to(device)
    a = F.one_hot(act, env.num_actions).float().to(device)
    v = vis.to(device)

    with torch.no_grad():
        output = model(x=x, a=a, visited=v)

    pred_ids = output.x_gt[0].argmax(dim=-1).cpu()
    true_obs = obs[0].cpu()
    pos_seq = pos[0].cpu()
    act_seq = act[0].cpu()

    state_visit_counts: Dict[int, int] = {}
    edge_visit_counts: Dict[Tuple[int, int, int], int] = {}

    initial_state = int(pos_seq[0].item())
    state_visit_counts[initial_state] = 1

    rows: List[Dict[str, int | float]] = []

    for t in range(1, walk_len):
        s_prev = int(pos_seq[t - 1].item())
        a_t = int(act_seq[t].item())
        s_cur = int(pos_seq[t].item())

        edge = (s_prev, a_t, s_cur)

        source_visit_count_before_t = state_visit_counts.get(s_prev, 0)
        target_visit_count_before_t = state_visit_counts.get(s_cur, 0)
        current_target_visit_index = target_visit_count_before_t + 1

        edge_visit_count_before_t = edge_visit_counts.get(edge, 0)
        current_edge_visit_index = edge_visit_count_before_t + 1

        edge_seen_before_t = edge_visit_count_before_t > 0
        first_time_edge = edge_visit_count_before_t == 0
        target_seen_before_t = target_visit_count_before_t > 0

        inferable_unseen_link = first_time_edge and target_seen_before_t

        nodes_visited_count_before_t = len(state_visit_counts)
        nodes_visited_fraction_before_t = (
            nodes_visited_count_before_t / env.num_states
        )
        unique_edges_seen_count_before_t = len(edge_visit_counts)

        pred_obs_id = int(pred_ids[t].item())
        true_obs_id = int(true_obs[t].item())
        correct = int(pred_obs_id == true_obs_id)

        row = {
            "checkpoint_step": int(checkpoint_step or -1),
            "episode_id": int(episode_id),
            "t": int(t),
            "state_prev": s_prev,
            "action_id": a_t,
            "state_cur": s_cur,
            "source_visit_count_before_t": source_visit_count_before_t,
            "target_visit_count_before_t": target_visit_count_before_t,
            "current_target_visit_index": current_target_visit_index,
            "edge_visit_count_before_t": edge_visit_count_before_t,
            "current_edge_visit_index": current_edge_visit_index,
            "edge_seen_before_t": int(edge_seen_before_t),
            "first_time_edge": int(first_time_edge),
            "target_seen_before_t": int(target_seen_before_t),
            "inferable_unseen_link": int(inferable_unseen_link),
            "nodes_visited_count_before_t": nodes_visited_count_before_t,
            "nodes_visited_fraction_before_t": nodes_visited_fraction_before_t,
            "unique_edges_seen_count_before_t": unique_edges_seen_count_before_t,
            "pred_obs_id": pred_obs_id,
            "true_obs_id": true_obs_id,
            "correct": correct,
        }
        rows.append(row)

        edge_visit_counts[edge] = edge_visit_count_before_t + 1
        state_visit_counts[s_cur] = target_visit_count_before_t + 1

    return rows


def evaluate_continuous(
    model: torch.nn.Module,
    env: GraphEnvironment,
    walk_len: int,
    num_episodes: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    checkpoint_step: Optional[int] = None,
) -> Dict[str, object]:
    """
    Generic continuous-walk zero-shot evaluation for any graph environment.

    Used by family_tree and spatial_graph experiments.

    Returns a dict with:
        overall_unseen_accuracy, overall_unseen_correct, overall_unseen_total,
        overall_seen_accuracy, overall_seen_correct, overall_seen_total,
        bins (list of per-checkpoint bin dicts).
    """
    rows = collect_evaluation_events(
        model=model,
        env=env,
        walk_len=walk_len,
        num_episodes=num_episodes,
        device=device,
        generator=generator,
        checkpoint_step=checkpoint_step,
    )

    summary = summarize_events(rows)

    # Build per-step bins for backward compatibility
    inferable = [r for r in rows if int(r["inferable_unseen_link"]) == 1]

    bin_data: Dict[int, Dict[str, List[int]]] = {}
    for r in inferable:
        key = int(r["target_visit_count_before_t"])
        if key not in bin_data:
            bin_data[key] = {"correct": [], "total": []}
        bin_data[key]["correct"].append(int(r["correct"]))
        bin_data[key]["total"].append(1)

    bins: List[Dict] = []
    for key in sorted(bin_data.keys()):
        c = sum(bin_data[key]["correct"])
        t = len(bin_data[key]["total"])
        bins.append(
            {
                "target_visit_count_before_t": key,
                "correct": c,
                "total": t,
                "accuracy": c / max(t, 1),
            }
        )

    return {
        "overall_unseen_accuracy": summary["inferable_unseen_accuracy"],
        "overall_unseen_correct": summary["inferable_unseen_correct"],
        "overall_unseen_total": summary["inferable_unseen_total"],
        "overall_seen_accuracy": summary["seen_accuracy"],
        "overall_seen_correct": summary["seen_correct"],
        "overall_seen_total": summary["seen_total"],
        "bins": bins,
    }


def collect_evaluation_events(
    model: torch.nn.Module,
    env: GraphEnvironment,
    walk_len: int,
    num_episodes: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    checkpoint_step: Optional[int] = None,
) -> List[Dict[str, int | float]]:
    """
    Collect event-level evaluation rows across many fresh worlds.
    """
    rows: List[Dict[str, int | float]] = []

    for episode_id in range(num_episodes):
        rows.extend(
            evaluate_continuous_single(
                model=model,
                env=env,
                walk_len=walk_len,
                device=device,
                generator=generator,
                checkpoint_step=checkpoint_step,
                episode_id=episode_id,
            )
        )

    return rows


def summarize_events(
    rows: List[Dict[str, int | float]],
) -> Dict[str, float | int]:
    """
    Compute checkpoint-level summary metrics from raw event rows.
    """
    inferable = [r for r in rows if int(r["inferable_unseen_link"]) == 1]
    seen = [r for r in rows if int(r["edge_seen_before_t"]) == 1]
    first_time = [r for r in rows if int(r["first_time_edge"]) == 1]

    inferable_correct = sum(int(r["correct"]) for r in inferable)
    inferable_total = len(inferable)

    seen_correct = sum(int(r["correct"]) for r in seen)
    seen_total = len(seen)

    first_time_correct = sum(int(r["correct"]) for r in first_time)
    first_time_total = len(first_time)

    return {
        "inferable_unseen_accuracy": inferable_correct / max(inferable_total, 1),
        "inferable_unseen_correct": inferable_correct,
        "inferable_unseen_total": inferable_total,
        "seen_accuracy": seen_correct / max(seen_total, 1),
        "seen_correct": seen_correct,
        "seen_total": seen_total,
        "first_time_edge_accuracy": first_time_correct / max(first_time_total, 1),
        "first_time_edge_correct": first_time_correct,
        "first_time_edge_total": first_time_total,
    }


def write_event_rows_csv(
    path: Path,
    rows: List[Dict[str, int | float]],
) -> None:
    """
    Save raw event rows to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
