from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tem_data.base import DiscreteEnvironment


@dataclass
class RectangleEnvironment(DiscreteEnvironment):
    """
    A 2D rectangular grid environment.

    States:
        state_id = row * width + col

    Actions:
        0 = up
        1 = down
        2 = left
        3 = right

    Boundary behavior:
        If an action would move outside the grid, the state remains unchanged.

    Observations:
        Each state is assigned one sensory observation id.
    """

    height: int
    width: int
    num_observations: int
    seed: int = 0

    name = "rectangle"

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive.")

        if self.width <= 0:
            raise ValueError("width must be positive.")

        if self.num_observations <= 0:
            raise ValueError("num_observations must be positive.")

        self.num_states = self.height * self.width
        self.num_actions = 4
        self.action_names = ["up", "down", "left", "right"]

        self.transition_table = self._build_transition_table()
        self.valid_action_mask = self._build_valid_action_mask()
        self.observation_ids = self._build_observation_ids()

    def _build_transition_table(self) -> torch.Tensor:
        table = torch.zeros(
            self.num_states,
            self.num_actions,
            dtype=torch.long,
        )

        for state in range(self.num_states):
            row = state // self.width
            col = state % self.width

            table[state, 0] = self.coord_to_state_int(max(row - 1, 0), col)
            table[state, 1] = self.coord_to_state_int(min(row + 1, self.height - 1), col)
            table[state, 2] = self.coord_to_state_int(row, max(col - 1, 0))
            table[state, 3] = self.coord_to_state_int(row, min(col + 1, self.width - 1))

        return table

    def _build_valid_action_mask(self) -> torch.Tensor:
        state_ids = torch.arange(self.num_states).unsqueeze(1)
        moved = self.transition_table != state_ids

        for state in range(self.num_states):
            if not moved[state].any():
                moved[state] = True

        return moved

    def _build_observation_ids(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.num_observations >= self.num_states:
            observation_ids = torch.randperm(
                self.num_observations,
                generator=generator,
            )[: self.num_states]
        else:
            observation_ids = torch.randint(
                low=0,
                high=self.num_observations,
                size=(self.num_states,),
                generator=generator,
            )

        return observation_ids.long()

    def coord_to_state_int(self, row: int, col: int) -> int:
        return row * self.width + col

    def coord_to_state(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        return row * self.width + col

    def state_to_coord(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = state // self.width
        col = state % self.width
        return row, col

    def next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape != action.shape:
            raise ValueError(
                "state and action must have the same shape. "
                f"Got state {tuple(state.shape)} and action {tuple(action.shape)}."
            )

        table = self.transition_table.to(state.device)
        return table[state, action]

    def get_observation_ids(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        observation_ids = self.observation_ids.to(state.device)
        return observation_ids[state]

    def sample_start_states(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        return torch.randint(
            low=0,
            high=self.num_states,
            size=(batch_size,),
            generator=generator,
            dtype=torch.long,
        )

    def sample_valid_actions(
        self,
        state: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if state.dim() != 1:
            raise ValueError(
                f"state must have shape [B], but got {tuple(state.shape)}."
            )

        valid_probs = self.valid_action_mask[state].float()

        action = torch.multinomial(
            valid_probs,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)

        return action.long()

    def valid_actions_for_state(self, state: int) -> List[int]:
        if state < 0 or state >= self.num_states:
            raise ValueError(f"state must be in [0, {self.num_states}), got {state}.")

        mask = self.valid_action_mask[state]
        return torch.where(mask)[0].tolist()

    def state_metadata(self) -> List[Dict[str, object]]:
        metadata = []

        for state in range(self.num_states):
            row = state // self.width
            col = state % self.width

            metadata.append(
                {
                    "state": state,
                    "row": row,
                    "col": col,
                }
            )

        return metadata
