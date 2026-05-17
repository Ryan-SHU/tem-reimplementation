"""
Helper functions for Line TI experiment.

Contains:
    - single-world evaluation (exploration + zero-shot query)
    - edge-tracking utilities
    - accuracy computation
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from tem_data.base import GraphEnvironment
from tem_data.sampling.random_walk import generate_random_walk, traversed_edges


def evaluate_zero_shot_single(
    model: torch.nn.Module,
    env: GraphEnvironment,
    explore_len: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """
    Run one evaluation episode in a *fresh* world.

    Steps:
        1. Resample observations (new world).
        2. Random walk of length explore_len (batch=1).
        3. Feed exploration walk to model.
        4. Identify un-traversed edges.
        5. For each un-traversed edge, teleport + query.
        6. Return accuracy dict.

    Returns:
        {
            "zero_shot_correct": int,
            "zero_shot_total":   int,
            "seen_correct":      int,
            "seen_total":        int,
            "nodes_visited":     int,
        }
    """
    model.eval()

    # 1. new world
    env.resample_observations(generator)

    # 2. exploration walk  [1, T_exp]
    walk = generate_random_walk(env, batch_size=1, seq_len=explore_len, generator=generator)
    pos = walk["position"]  # [1, T_exp]
    act = walk["action"]    # [1, T_exp]
    vis = walk["visited"]   # [1, T_exp]
    obs = env.get_observation_ids(pos)

    x_exp = F.one_hot(obs, env.num_observations).float().to(device)  # [1,T,Nx]
    a_exp = F.one_hot(act, env.num_actions).float().to(device)       # [1,T,Na]
    vis_exp = vis.to(device)                                          # [1,T]

    # 3. feed exploration walk
    with torch.no_grad():
        output_exp = model(x=x_exp, a=a_exp, visited=vis_exp)

    # 4. identify traversed and un-traversed edges
    seen = traversed_edges(pos[0], act[0])
    all_edges = env.all_valid_edges()
    unseen = [e for e in all_edges if e not in seen]

    visited_states = set(pos[0].tolist())
    nodes_visited = len(visited_states)

    # 5. query each edge
    # Strategy: build a short continuation sequence for each query edge.
    # For edge (s, a, s'):
    #   step 0: provide x_s with dummy action  -> model "teleports" to s
    #   step 1: provide action a -> model predicts x from transition
    # We collect the model prediction at step 1 and compare with true obs.

    zero_shot_correct = 0
    zero_shot_total = 0
    seen_correct = 0
    seen_total = 0

    for (s, a_int, s_next) in all_edges:
        # We only query edges whose *source* state was visited during exploration
        if s not in visited_states:
            continue

        # Build a 2-step mini-sequence
        obs_s = env.get_observation_ids(torch.tensor([s])).item()
        obs_s_next = env.get_observation_ids(torch.tensor([s_next])).item()

        # step 0: teleport to s
        x_query = torch.zeros(1, 2, env.num_observations, device=device)
        a_query = torch.zeros(1, 2, env.num_actions, device=device)
        v_query = torch.ones(1, 2, device=device)

        x_query[0, 0, obs_s] = 1.0         # observation at s
        a_query[0, 0, 0] = 1.0             # dummy action for step 0
        x_query[0, 1, obs_s_next] = 1.0    # true obs at s' (used only for loss, not prediction)
        a_query[0, 1, a_int] = 1.0         # the queried action

        with torch.no_grad():
            output_q = model(x=x_query, a=a_query, visited=v_query)

        # The model's generative prediction at step 1 (before seeing x)
        # is the key. We use x_gen from the generative pathway.
        # output_q.x_gen is [1, T, Nx] — take step 1
        pred = output_q.x_gt[0, 1]         # [Nx]
        pred_id = int(pred.argmax().item())
        correct = int(pred_id == obs_s_next)

        edge_tuple = (s, a_int, s_next)
        if edge_tuple in seen:
            seen_correct += correct
            seen_total += 1
        else:
            zero_shot_correct += correct
            zero_shot_total += 1

    return {
        "zero_shot_correct": zero_shot_correct,
        "zero_shot_total": zero_shot_total,
        "seen_correct": seen_correct,
        "seen_total": seen_total,
        "nodes_visited": nodes_visited,
    }


def evaluate_zero_shot(
    model: torch.nn.Module,
    env: GraphEnvironment,
    explore_len: int,
    num_episodes: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """
    Average zero-shot evaluation over many new-world episodes.

    Returns:
        {
            "zero_shot_accuracy": float,
            "seen_accuracy":      float,
            "avg_nodes_visited":  float,
        }
    """
    total_zs_correct = 0
    total_zs_count = 0
    total_seen_correct = 0
    total_seen_count = 0
    total_nodes = 0

    for _ in range(num_episodes):
        result = evaluate_zero_shot_single(
            model=model,
            env=env,
            explore_len=explore_len,
            device=device,
            generator=generator,
        )
        total_zs_correct += result["zero_shot_correct"]
        total_zs_count += result["zero_shot_total"]
        total_seen_correct += result["seen_correct"]
        total_seen_count += result["seen_total"]
        total_nodes += result["nodes_visited"]

    zs_acc = total_zs_correct / max(total_zs_count, 1)
    seen_acc = total_seen_correct / max(total_seen_count, 1)
    avg_nodes = total_nodes / max(num_episodes, 1)

    return {
        "zero_shot_accuracy": zs_acc,
        "seen_accuracy": seen_acc,
        "avg_nodes_visited": avg_nodes,
        "zero_shot_correct": total_zs_correct,
        "zero_shot_total": total_zs_count,
        "seen_correct": total_seen_correct,
        "seen_total": total_seen_count,
    }
