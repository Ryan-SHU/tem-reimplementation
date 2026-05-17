from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class DiscreteEnvironment(ABC):
    """
    Base interface for all TEM experimental environments.

    An environment defines:

        state space:
            s in {0, ..., num_states - 1}

        action / relation space:
            a in {0, ..., num_actions - 1}

        transition:
            next_state = tau(s, a)

        sensory mapping:
            observation_id = omega(s)

    TEM itself should never depend on concrete environment classes.
    It should only see x, a, visited tensors generated from this interface.
    """

    num_states: int
    num_actions: int
    num_observations: int
    transition_table: torch.Tensor
    valid_action_mask: torch.Tensor
    observation_ids: torch.Tensor

    @abstractmethod
    def next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_observation_ids(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

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

    def state_to_coord(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "state_to_coord is only available for spatial environments."
        )
