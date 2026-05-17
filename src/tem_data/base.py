"""
Abstract graph environment for TEM.

Every TEM environment is a discrete graph:

    G = (S, A, tau, phi)

    S     = finite state set, indexed 0 .. num_states - 1
    A     = finite action set, indexed 0 .. num_actions - 1
    tau   = deterministic transition: S x A -> S
    phi   = sensory assignment: S -> {0 .. num_observations - 1}

Different 'worlds' share (S, A, tau) but differ in phi.
"""

from typing import Dict, List, Optional, Set, Tuple

import torch


class GraphEnvironment:
    """Base class for all TEM graph environments."""

    name: str = "graph"
    num_states: int
    num_actions: int
    num_observations: int

    transition_table: torch.Tensor     # [S, A] long
    valid_action_mask: torch.Tensor    # [S, A] bool
    observation_ids: torch.Tensor      # [S]    long

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #

    def next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transition. state, action: same shape. Returns next state."""
        return self.transition_table.to(state.device)[state, action]

    def get_observation_ids(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.observation_ids.to(state.device)[state]

    def sample_start_states(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.randint(
            0, self.num_states, (batch_size,),
            generator=generator, dtype=torch.long,
        )

    def sample_valid_actions(
        self,
        state: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        probs = self.valid_action_mask.to(state.device)[state].float()
        return torch.multinomial(
            probs, 1, replacement=True, generator=generator,
        ).squeeze(1)

    # ------------------------------------------------------------------ #
    # Sensory re-assignment (creates a 'new world' on same graph)
    # ------------------------------------------------------------------ #

    def resample_observations(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Randomly re-assign sensory observations to states."""
        if self.num_observations >= self.num_states:
            ids = torch.randperm(self.num_observations, generator=generator)[
                : self.num_states
            ]
        else:
            ids = torch.randint(
                0, self.num_observations, (self.num_states,),
                generator=generator,
            )
        self.observation_ids = ids.long()

    # ------------------------------------------------------------------ #
    # Edge tracking (for zero-shot evaluation)
    # ------------------------------------------------------------------ #

    def all_valid_edges(self) -> List[Tuple[int, int, int]]:
        """Return list of (state, action, next_state) for every valid edge."""
        edges = []
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if self.valid_action_mask[s, a]:
                    s_next = int(self.transition_table[s, a].item())
                    edges.append((s, a, s_next))
        return edges

    # ------------------------------------------------------------------ #
    # Metadata for analysis
    # ------------------------------------------------------------------ #

    def state_metadata(self) -> List[Dict[str, object]]:
        return [{"state": s} for s in range(self.num_states)]
