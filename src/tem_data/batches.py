from typing import Dict

import torch
import torch.nn.functional as F

from tem_data.base import DiscreteEnvironment
from tem_data.walks import generate_random_walk


class RandomWalkBatcher:
    """
    Convert random walks into TEM-ready training batches.

    Output batch:

        x:
            [B, T, N_x]
            One-hot sensory observations.

        a:
            [B, T, N_a]
            One-hot actions.

        visited:
            [B, T]
            Memory update mask.

        position:
            [B, T]
            Hidden environment states. Analysis only.

        observation_id:
            [B, T]
            Integer sensory ids. Analysis only.

        action_id:
            [B, T]
            Integer action ids. Analysis only.
    """

    def __init__(
        self,
        environment: DiscreteEnvironment,
        batch_size: int,
        sequence_length: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        self.environment = environment
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = torch.device(device)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample(self) -> Dict[str, torch.Tensor]:
        walk = generate_random_walk(
            environment=self.environment,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            generator=self.generator,
        )

        position = walk["position"]
        action_id = walk["action"]
        visited = walk["visited"]

        observation_id = self.environment.get_observation_ids(position)

        x = F.one_hot(
            observation_id,
            num_classes=self.environment.num_observations,
        ).float()

        a = F.one_hot(
            action_id,
            num_classes=self.environment.num_actions,
        ).float()

        batch = {
            "x": x,
            "a": a,
            "visited": visited,
            "position": position,
            "observation_id": observation_id,
            "action_id": action_id,
        }

        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        return self.sample()
