import torch

from tem.models.tensor_ops import (
    l2_normalize,
    phase_sum,
    repeat_sensory,
    sense_sum,
    tile_phase,
)


def test_l2_normalize_shape() -> None:
    x = torch.randn(4, 8)
    y = l2_normalize(x)

    assert y.shape == x.shape


def test_l2_normalize_norm_is_close_to_one() -> None:
    x = torch.randn(4, 8)
    y = l2_normalize(x)

    norms = torch.linalg.norm(y, dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tile_phase_shape() -> None:
    batch_size = 2
    compressed_x_dim = 3
    num_phases = 4

    x = torch.randn(batch_size, compressed_x_dim)
    y = tile_phase(x, num_phases)

    assert y.shape == (batch_size, num_phases * compressed_x_dim)


def test_tile_phase_values() -> None:
    x = torch.tensor([[1.0, 2.0]])
    y = tile_phase(x, num_phases=3)

    expected = torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]])

    assert torch.allclose(y, expected)


def test_repeat_sensory_shape() -> None:
    batch_size = 2
    num_phases = 4
    compressed_x_dim = 3

    x = torch.randn(batch_size, num_phases)
    y = repeat_sensory(x, compressed_x_dim)

    assert y.shape == (batch_size, num_phases * compressed_x_dim)


def test_repeat_sensory_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = repeat_sensory(x, compressed_x_dim=2)

    expected = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])

    assert torch.allclose(y, expected)


def test_phase_sum_shape() -> None:
    batch_size = 2
    num_phases = 4
    compressed_x_dim = 3

    x = torch.randn(batch_size, num_phases * compressed_x_dim)
    y = phase_sum(x, num_phases, compressed_x_dim)

    assert y.shape == (batch_size, compressed_x_dim)


def test_phase_sum_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])

    y = phase_sum(x, num_phases=2, compressed_x_dim=3)

    expected = torch.tensor([[11.0, 22.0, 33.0]])

    assert torch.allclose(y, expected)


def test_sense_sum_shape() -> None:
    batch_size = 2
    num_phases = 4
    compressed_x_dim = 3

    x = torch.randn(batch_size, num_phases * compressed_x_dim)
    y = sense_sum(x, num_phases, compressed_x_dim)

    assert y.shape == (batch_size, num_phases)


def test_sense_sum_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])

    y = sense_sum(x, num_phases=2, compressed_x_dim=3)

    expected = torch.tensor([[6.0, 60.0]])

    assert torch.allclose(y, expected)
