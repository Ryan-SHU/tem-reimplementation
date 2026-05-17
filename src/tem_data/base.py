from typing import Dict, List, Optional

import torch


class DiscreteEnvironment:
    """
    Base interface for discrete TEM environments.

    An environment defines:

        states:
            hidden graph nodes

        actions:
            graph relations or movement directions

        transition:
            state, action -> next state

        observation:
            state -> sensory observation id

    The TEM model never receives state ids directly.
    State ids are used only for data generation and analysis.
    """

    name = "discrete_environment"

    num_states: int
    num_actions: int
    num_observations: int
    action_names: List[str]

    def next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply actions to states.

        Args:
            state:
                Tensor of integer state ids.

            action:
                Tensor of integer action ids with same shape as state.

        Returns:
            Tensor of next state ids with same shape as state.
        """
        raise NotImplementedError

    def get_observation_ids(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert state ids to sensory observation ids.
        """
        raise NotImplementedError

    def sample_start_states(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample initial states.
        """
        raise NotImplementedError

    def sample_valid_actions(
        self,
        state: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample one valid action for each state in a batch.
        """
        raise NotImplementedError

    def state_metadata(self) -> List[Dict[str, object]]:
        """
        Return metadata for every state.

        This is used by analysis and plotting code.

        Non-spatial environments can return only:

            {"state": state_id}

        Spatial environments can additionally return:

            {"state": state_id, "row": row, "col": col}
        """
        metadata = []

        for state in range(self.num_states):
            metadata.append({"state": state})

        return metadata
