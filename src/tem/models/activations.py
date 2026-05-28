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

    Matches the original TEM paper (Whittington et al.):
        f_g_freq(g, _) = tf.minimum(tf.maximum(g, -1), 1)

    This is a hard clip to [-1, 1] with no tanh.

    Shape:
        input:  any shape
        output: same shape
    """
    return torch.clamp(x, min=-1.0, max=1.0)


def phi_p(x: torch.Tensor) -> torch.Tensor :
    """
    Place-state activation.

    Matches the original TEM paper (Whittington et al.):
        f_p_freq(p, _) = tf.nn.leaky_relu(tf.minimum(tf.maximum(p, -1), 1))

    This is clamp first, then leaky_relu (with alpha=0.01, PyTorch default).

    Shape:
        input:  any shape
        output: same shape
    """
    x = torch.clamp(x, min=-1.0, max=1.0)
    return F.leaky_relu(x)


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