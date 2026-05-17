from typing import Dict, Protocol

import torch


class BatchProvider(Protocol):
    """
    Minimal interface required by TEMTrainer.

    Any experiment-specific batcher can be used as long as it returns a
    dictionary containing:

        x:       [B, T, N_x]
        a:       [B, T, N_a]
        visited: [B, T]
    """

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
