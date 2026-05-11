from dataclasses import dataclass

import torch


@dataclass
class RectangleEnvironment:
    """
    A 2D rectangular grid environment.

    This environment is intentionally simple and deterministic in its structure.

    States:
        Each grid cell is one discrete state.

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

        If num_observations >= num_states:
            Each state receives a unique observation id.

        If num_observations < num_states:
            Observation ids are sampled with replacement.
    """

    height: int
    width: int
    num_observations: int
    seed: int = 0

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
        """
        Build deterministic transition table.

        Shape:
            transition_table: [num_states, num_actions]

        Meaning:
            transition_table[s, a] = next state after taking action a at state s.
        """
        table = torch.zeros(
            self.num_states,
            self.num_actions,
            dtype=torch.long,
        )

        for state in range(self.num_states):
            row = state // self.width
            col = state % self.width

            # Action 0: up
            next_row = max(row - 1, 0)
            next_col = col
            table[state, 0] = self.coord_to_state_int(next_row, next_col)

            # Action 1: down
            next_row = min(row + 1, self.height - 1)
            next_col = col
            table[state, 1] = self.coord_to_state_int(next_row, next_col)

            # Action 2: left
            next_row = row
            next_col = max(col - 1, 0)
            table[state, 2] = self.coord_to_state_int(next_row, next_col)

            # Action 3: right
            next_row = row
            next_col = min(col + 1, self.width - 1)
            table[state, 3] = self.coord_to_state_int(next_row, next_col)

        return table

    def _build_valid_action_mask(self) -> torch.Tensor:
        """
        Build a mask indicating which actions actually move the agent.

        Shape:
            valid_action_mask: [num_states, num_actions]

        Meaning:
            valid_action_mask[s, a] = True
                if action a changes the state from s.

            valid_action_mask[s, a] = False
                if action a hits a wall and leaves the state unchanged.

        Special case:
            In a 1x1 grid, no action can move the agent.
            In that case, all actions are allowed and they all keep the state fixed.
        """
        state_ids = torch.arange(self.num_states).unsqueeze(1)
        moved = self.transition_table != state_ids

        for state in range(self.num_states):
            if not moved[state].any():
                moved[state] = True

        return moved

    def _build_observation_ids(self) -> torch.Tensor:
        """
        Assign one observation id to each state.

        Shape:
            observation_ids: [num_states]
        """
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
        """
        Convert a single integer coordinate to a state id.
        """
        return row * self.width + col

    def coord_to_state(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert tensor coordinates to state ids.

        Inputs:
            row: any shape
            col: same shape as row

        Output:
            state: same shape as row
        """
        return row * self.width + col

    def state_to_coord(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert state ids to row and column coordinates.

        Input:
            state: any shape

        Outputs:
            row: same shape as state
            col: same shape as state
        """
        row = state // self.width
        col = state % self.width
        return row, col

    def next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply actions to states.

        Inputs:
            state:  any shape, dtype long
            action: same shape as state, dtype long

        Output:
            next_state: same shape as state, dtype long
        """
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
        """
        Map states to sensory observation ids.

        Input:
            state: any shape, dtype long

        Output:
            observation_id: same shape as state, dtype long
        """
        observation_ids = self.observation_ids.to(state.device)
        return observation_ids[state]

    def sample_start_states(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample random initial states.

        Output:
            start_states: [batch_size]
        """
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
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample a valid action for each state.

        Input:
            state: [B]

        Output:
            action: [B]

        Notes:
            A valid action is one that moves the agent, except in degenerate
            states where no action can move the agent. In those states all
            actions are treated as valid.
        """
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

    def valid_actions_for_state(self, state: int) -> list[int]:
        """
        Return valid actions for one Python integer state.

        This is mostly useful for debugging and tests.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"state must be in [0, {self.num_states}), got {state}.")

        mask = self.valid_action_mask[state]
        return torch.where(mask)[0].tolist()
