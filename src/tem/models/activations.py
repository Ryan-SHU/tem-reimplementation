# src/models/activations.py
"""
phi_g is used in g-space.
phi_p is used in p-space and for compressed x encoding.
positive_softplus is used for sigma_path and sigma_mem.
"""

import torch 
import torch.nn.functional as F 

# epsilon
# Small constant for numerical stability.
# Used to avoid division by zero and exactly-zero scales.
EPS = 1e-8



def phi_g(x: torch.Tensor) -> torch.Tensor :
    """
    Grid-state activation.

    Formula:
        phi_g(x) = clip(tanh(x), -1, 1)

    Shape:
        input:  any shape
        output: same shape
    """
    return torch.clamp(torch.tanh(x), min=-1.0, max=1.0)


def phi_p(x: torch.Tensor) -> torch.Tensor :
    """
    Place-state activation.

    Formula:
        phi_p(x) = clip(leaky_relu(x), -1, 1)

    Shape:
        input:  any shape
        output: same shape
    """
    return torch.clamp(F.leaky_relu(x), min=-1.0, max=1.0)


def positive_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically safe positive scale function.

    Formula:
        softplus(x) + eps

    Shape:
        input:  any shape
        output: same shape
    """
    return F.softplus(x) + EPS