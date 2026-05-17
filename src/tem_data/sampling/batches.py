"""
Convert integer random walks into TEM-ready one-hot tensors.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tem_data.base import GraphEnvironment
from tem_data.sampling.random_walk import generate_random_walk


class WalkBatcher:
    """
    Generates training batches for TEM.

    Each call to sample() creates a fresh random walk
    in the (possibly re-sampled) environment.
    """

    def __init__(
        self,
        env: GraphEnvironment,
        batch_size: int,
        seq_len: int,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.env = env
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.generator = generator

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        walk = generate_random_walk(
            self.env, self.batch_size, self.seq_len, self.generator,
        )
        pos = walk["position"]
        act = walk["action"]
        vis = walk["visited"]

        obs = self.env.get_observation_ids(pos)

        x = F.one_hot(obs, self.env.num_observations).float()
        a = F.one_hot(act, self.env.num_actions).float()

        return {
            "x": x.to(self.device),
            "a": a.to(self.device),
            "visited": vis.to(self.device),
            "position": pos.to(self.device),
            "observation_id": obs.to(self.device),
            "action_id": act.to(self.device),
        }
