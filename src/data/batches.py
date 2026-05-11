import torch
import torch.nn.functional as F

from data.environments import RectangleEnvironment
from data.walks import generate_random_walk


class RandomWalkBatcher:
    """
    Turn random walks into TEM-ready training batches.

    Main output format:

        batch["x"]:
            [B, T, N_x]
            One-hot sensory observations.

        batch["a"]:
            [B, T, N_a]
            One-hot actions.

        batch["visited"]:
            [B, T]
            Memory-update mask.

        batch["position"]:
            [B, T]
            Integer environment state ids.
            This is not passed into the model.
            It is saved for representation analysis.

        batch["observation_id"]:
            [B, T]
            Integer sensory observation ids.

        batch["action_id"]:
            [B, T]
            Integer action ids.

    The TEM model only needs:

        x, a, visited

    Analysis code can use:

        position, observation_id, action_id
    """

    def __init__(
        self,
        environment: RectangleEnvironment,
        batch_size: int,
        sequence_length: int,
        device: str | torch.device = "cpu",
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

    def sample(self) -> dict[str, torch.Tensor]:
        """
        Sample one TEM training batch.
        """
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

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """
        Alias for sample().

        Some training code reads more naturally with:

            batch = batcher.sample_batch()

        while quick experiments often use:

            batch = batcher.sample()
        """
        return self.sample()
