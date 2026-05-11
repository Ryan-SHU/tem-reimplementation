# src/models/tensor_ops.py
'''
num_phases       = Phi_f
compressed_x_dim = C
P_f              = Phi_f * C
'''

import torch

from tem.models.activations import EPS


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor along the last dimension.

    Formula:
        x / (||x||_2 + eps)

    Shape:
        input:  [B, C]
        output: [B, C]
    """
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (norm + EPS)


def tile_phase(x: torch.Tensor, num_phases: int) -> torch.Tensor:
    """
    Repeat a sensory vector across phase dimension.

    Input:
        x: [B, C]

    Output:
        y: [B, num_phases * C]

    Example:
        x shape [B, C]
        -> [B, 1, C]
        -> [B, Phi, C]
        -> [B, Phi * C]
    """
    if x.dim() != 2:
        raise ValueError(f"x must have shape [B, C], but got {tuple(x.shape)}.")

    batch_size, compressed_x_dim = x.shape

    y = x.unsqueeze(1)
    y = y.repeat(1, num_phases, 1)
    y = y.reshape(batch_size, num_phases * compressed_x_dim)

    return y


def repeat_sensory(x: torch.Tensor, compressed_x_dim: int) -> torch.Tensor:
    """
    Repeat each phase value across sensory dimension.

    Input:
        x: [B, Phi]

    Output:
        y: [B, Phi * C]

    Example:
        x shape [B, Phi]
        -> [B, Phi, 1]
        -> [B, Phi, C]
        -> [B, Phi * C]
    """
    if x.dim() != 2:
        raise ValueError(f"x must have shape [B, Phi], but got {tuple(x.shape)}.")

    batch_size, num_phases = x.shape

    y = x.unsqueeze(-1)
    y = y.repeat(1, 1, compressed_x_dim)
    y = y.reshape(batch_size, num_phases * compressed_x_dim)

    return y


def phase_sum(x: torch.Tensor, num_phases: int, compressed_x_dim: int) -> torch.Tensor:
    """
    Sum a p-space vector over phase dimension.

    Input:
        x: [B, Phi * C]

    Output:
        y: [B, C]
    """
    if x.dim() != 2:
        raise ValueError(f"x must have shape [B, Phi * C], but got {tuple(x.shape)}.")

    expected_dim = num_phases * compressed_x_dim

    if x.shape[-1] != expected_dim:
        raise ValueError(
            f"x last dimension must be {expected_dim}, "
            f"but got {x.shape[-1]}."
        )

    batch_size = x.shape[0]

    x_view = x.reshape(batch_size, num_phases, compressed_x_dim)
    y = x_view.sum(dim=1)

    return y


def sense_sum(x: torch.Tensor, num_phases: int, compressed_x_dim: int) -> torch.Tensor:
    """
    Sum a p-space vector over sensory dimension.

    Input:
        x: [B, Phi * C]

    Output:
        y: [B, Phi]
    """
    if x.dim() != 2:
        raise ValueError(f"x must have shape [B, Phi * C], but got {tuple(x.shape)}.")

    expected_dim = num_phases * compressed_x_dim

    if x.shape[-1] != expected_dim:
        raise ValueError(
            f"x last dimension must be {expected_dim}, "
            f"but got {x.shape[-1]}."
        )

    batch_size = x.shape[0]

    x_view = x.reshape(batch_size, num_phases, compressed_x_dim)
    y = x_view.sum(dim=2)

    return y
