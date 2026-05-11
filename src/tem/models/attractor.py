import torch
from torch import nn

from tem.models.activations import phi_p

class Attractor(nn.Module):
    """
    Attractor retrieval module.

    This module implements:

        h_0 = phi_p(query)

        for k in range(K):
            memory_drive = h_k @ memory
            h_{k+1} = phi_p(kappa * h_k + memory_drive)

    Input:
        query:  [B, P_f]
        memory: [B, P_f, P_f]
        kappa:  scalar float

    Output:
        h: [B, P_f]
    """

    def __init__(self, num_iterations: int) -> None:
        super().__init__()

        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

        self.num_iterations = num_iterations

    def forward(self, query: torch.Tensor, memory: torch.Tensor, kappa: float) -> torch.Tensor:
        
        if query.dim() != 2:
            raise ValueError(
                f"query must have shape [B, P], but got {tuple(query.shape)}."
            )
        
        if memory.dim() != 3:
            raise ValueError(
                f"memory must have shape [B, P, P], but got {tuple(memory.shape)}."
            )

        batch_size, p_dim = query.shape
        
        if memory.shape != (batch_size, p_dim, p_dim):
            raise ValueError(
                "memory shape must match query shape. "
                f"Expected {(batch_size, p_dim, p_dim)}, "
                f"but got {tuple(memory.shape)}."
            )

        h = phi_p(query)

        for _ in range(self.num_iterations):
            # bmm(h_k[:, None, :], memory)[:, 0, :]
            memory_drive = torch.bmm(h.unsqueeze(1), memory).squeeze(1)
            h = phi_p(kappa * h + memory_drive)

            # h.unsqueeze(1): [B, 1, P_f]
            # memory:         [B, P_f, P_f]
            # output:         [B, 1, P_f]
            # squeeze:        [B, P_f]

        return h