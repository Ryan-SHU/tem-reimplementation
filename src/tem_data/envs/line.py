"""
Line graph environment for Transitive Inference.

Graph:  0 -- 1 -- 2 -- ... -- (N-1)

Actions:
    0 = left  (toward smaller index)
    1 = right (toward larger index)

Boundary: stay in place.

Paper reference: Figure 3 A, D  —  7-node TI hierarchy.
"""

from typing import Optional

import torch

from tem_data.base import GraphEnvironment


class LineEnvironment(GraphEnvironment):

    name = "line"

    def __init__(
        self,
        num_nodes: int,
        num_observations: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if num_nodes < 2:
            raise ValueError("num_nodes must be >= 2.")

        self.num_nodes = num_nodes
        self.num_states = num_nodes
        self.num_actions = 2
        self.num_observations = num_observations

        self.transition_table = self._build_transitions()
        self.valid_action_mask = self._build_valid_mask()
        self.resample_observations(generator)

    def _build_transitions(self) -> torch.Tensor:
        # [S, 2]
        table = torch.zeros(self.num_states, 2, dtype=torch.long)
        for s in range(self.num_states):
            table[s, 0] = max(s - 1, 0)             # left
            table[s, 1] = min(s + 1, self.num_states - 1)  # right
        return table

    def _build_valid_mask(self) -> torch.Tensor:
        state_ids = torch.arange(self.num_states).unsqueeze(1)  # [S, 1]
        mask = self.transition_table != state_ids                # [S, 2]
        # endpoints: at least one action is valid
        for s in range(self.num_states):
            if not mask[s].any():
                mask[s] = True
        return mask
