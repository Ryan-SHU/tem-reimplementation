"""
Helper functions for Experiment 1A (Line TI).

This file implements the correct continuous-walk evaluation protocol.

Core idea
---------
For a fresh world:
    1. resample sensory observations on the same graph
    2. run one continuous walk
    3. at each step t >= 1:
         - use x_gt[t] as the model prediction for the current target observation
         - classify whether the current edge was seen before
         - classify whether the current edge is an inferable unseen link
         - record exploration coverage before step t

Raw event rows are saved and used later for offline plotting.

Why x_gt?
---------
x_gt is the transition-only predictive branch used against the current
observation target during training. It is the most appropriate branch for
evaluating link inference before the current observation is incorporated.
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
    A list of event rows, one row per step t >= 1.
    """
    model.eval()

    # fresh world
    env.resample_observations(generator)

    walk = generate_random_walk(
        env=env,
        batch_size=1,
        seq_len=walk_len,
        generator=generator,
    )

    pos = walk["position"]   # [1, T]
    act = walk["action"]     # [1, T]
    vis = walk["visited"]    # [1, T]
    obs = env.get_observation_ids(pos)

    x = F.one_hot(obs, env.num_observations).float().to(device)
    a = F.one_hot(act, env.num_actions).float().to(device)
    v = vis.to(device)

    with torch.no_grad():
        output = model(x=x, a=a, visited=v)

    pred_ids = output.x_gt[0].argmax(dim=-1).cpu()   # [T]
    true_obs = obs[0].cpu()
    pos_seq = pos[0].cpu()
    act_seq = act[0].cpu()

    seen_edges: Set[Tuple[int, int, int]] = set()
    visited_states: Set[int] = set()

    visited_states.add(int(pos_seq[0].item()))

    rows: List[Dict[str, int | float]] = []

    for t in range(1, walk_len):
        s_prev = int(pos_seq[t - 1].item())
        a_t = int(act_seq[t].item())
        s_cur = int(pos_seq[t].item())

        edge = (s_prev, a_t, s_cur)

        edge_seen_before = edge in seen_edges
        first_time_edge = not edge_seen_before
        target_seen_before = s_cur in visited_states

        # This is the paper-relevant event type:
        # the link itself is new, but the target node has already been visited
        inferable_unseen_link = first_time_edge and target_seen_before

        correct = int(pred_ids[t].item()) == int(true_obs[t].item())

        row = {
            "checkpoint_step": int(checkpoint_step or -1),
            "episode_id": int(episode_id),
            "t": int(t),
            "state_prev": s_prev,
            "action_id": a_t,
            "state_cur": s_cur,
            "edge_seen_before_t": int(edge_seen_before),
            "first_time_edge": int(first_time_edge),
            "target_seen_before_t": int(target_seen_before),
            "inferable_unseen_link": int(inferable_unseen_link),
            "nodes_visited_count_before_t": int(len(visited_states)),
            "nodes_visited_fraction_before_t": float(len(visited_states) / env.num_states),
            "unique_edges_seen_count_before_t": int(len(seen_edges)),
            "pred_obs_id": int(pred_ids[t].item()),
            "true_obs_id": int(true_obs[t].item()),
            "correct": int(correct),
        }
        rows.append(row)

        # update AFTER logging the current event
        seen_edges.add(edge)
        visited_states.add(s_cur)

    return rows


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
