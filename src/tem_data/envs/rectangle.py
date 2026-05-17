"""
Rectangle grid environment.

Grid:  height x width
State: row * width + col

Actions:
    0 = up
    1 = down
    2 = left
    3 = right

Boundary: stay in place.

Paper reference: Figure 3 C, F  —  2D spatial graph.
                 Figure 4       —  grid / place representations.
"""

from typing import Dict, List, Optional, Tuple

import torch

from tem_data.base import GraphEnvironment


class RectangleEnvironment(GraphEnvironment):

    name = "rectangle"

    def __init__(
        self,
        height: int,
        width: int,
        num_observations: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1.")

        self.height = height
        self.width = width
        self.num_states = height * width
        self.num_actions = 4
        self.num_observations = num_observations

        self.transition_table = self._build_transitions()
        self.valid_action_mask = self._build_valid_mask()
        self.resample_observations(generator)

    def _build_transitions(self) -> torch.Tensor:
        S, A = self.num_states, 4
        table = torch.zeros(S, A, dtype=torch.long)
        for s in range(S):
            r, c = divmod(s, self.width)
            table[s, 0] = max(r - 1, 0) * self.width + c                    # up
            table[s, 1] = min(r + 1, self.height - 1) * self.width + c      # down
            table[s, 2] = r * self.width + max(c - 1, 0)                    # left
            table[s, 3] = r * self.width + min(c + 1, self.width - 1)       # right
        return table

    def _build_valid_mask(self) -> torch.Tensor:
        state_ids = torch.arange(self.num_states).unsqueeze(1)
        mask = self.transition_table != state_ids
        for s in range(self.num_states):
            if not mask[s].any():
                mask[s] = True
        return mask

    def state_to_coord(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return state // self.width, state % self.width

    def state_metadata(self) -> List[Dict[str, object]]:
        meta = []
        for s in range(self.num_states):
            r, c = divmod(s, self.width)
            meta.append({"state": s, "row": r, "col": c})
        return meta
