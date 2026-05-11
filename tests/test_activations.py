import torch

from tem.models.activations import phi_g, phi_p, positive_softplus


def test_phi_g_shape() -> None:
    x = torch.randn(2, 3, 4)
    y = phi_g(x)

    assert y.shape == x.shape


def test_phi_g_range() -> None:
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = phi_g(x)

    assert torch.all(y >= -1.0)
    assert torch.all(y <= 1.0)


def test_phi_p_shape() -> None:
    x = torch.randn(2, 3, 4)
    y = phi_p(x)

    assert y.shape == x.shape


def test_phi_p_range() -> None:
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = phi_p(x)

    assert torch.all(y >= -1.0)
    assert torch.all(y <= 1.0)


def test_positive_softplus_is_positive() -> None:
    x = torch.randn(10)
    y = positive_softplus(x)

    assert torch.all(y > 0.0)
