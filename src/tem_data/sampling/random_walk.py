"""
Generic random walk on a GraphEnvironment.

Convention:
    position[:, 0]  = start state (uniform random)
    action[:, 0]    = dummy = 0
    action[:, t]    = sampled at position[:, t-1], then
    position[:, t]  = tau(position[:, t-1], action[:, t])
"""

from typing import Dict, Optional, Set, Tuple

import torch

from tem_data.base import GraphEnvironment


def generate_random_walk(
    env: GraphEnvironment,
    batch_size: int,
    seq_len: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
        position:  [B, T] long
        action:    [B, T] long
        visited:   [B, T] float  (all 1.0)
    """
    position = torch.zeros(batch_size, seq_len, dtype=torch.long)
    action = torch.zeros(batch_size, seq_len, dtype=torch.long)
    visited = torch.ones(batch_size, seq_len, dtype=torch.float32)

    cur = env.sample_start_states(batch_size, generator)
    position[:, 0] = cur

    for t in range(1, seq_len):
        a = env.sample_valid_actions(cur, generator)
        nxt = env.next_state(cur, a)
        action[:, t] = a
        position[:, t] = nxt
        cur = nxt

    return {"position": position, "action": action, "visited": visited}


def traversed_edges(
    position: torch.Tensor,
    action: torch.Tensor,
) -> Set[Tuple[int, int, int]]:
    """
    Extract set of (state, action, next_state) actually traversed.
    Works on a single sequence: position [T], action [T].
    """
    edges = set()
    T = position.shape[0]
    for t in range(1, T):
        s = int(position[t - 1].item())
        a = int(action[t].item())
        s_next = int(position[t].item())
        edges.add((s, a, s_next))
    return edges
