import torch

from tem.models.attractor import Attractor


def test_attractor_output_shape() -> None:
    batch_size = 2
    p_dim = 12

    query = torch.randn(batch_size, p_dim)
    memory = torch.randn(batch_size, p_dim, p_dim)

    attractor = Attractor(num_iterations=5)
    output = attractor(query, memory, kappa=0.8)

    assert output.shape == (batch_size, p_dim)


def test_attractor_output_range() -> None:
    batch_size = 2
    p_dim = 12

    query = torch.randn(batch_size, p_dim)
    memory = torch.randn(batch_size, p_dim, p_dim)

    attractor = Attractor(num_iterations=5)
    output = attractor(query, memory, kappa=0.8)

    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_attractor_with_zero_memory() -> None:
    batch_size = 2
    p_dim = 12

    query = torch.randn(batch_size, p_dim)
    memory = torch.zeros(batch_size, p_dim, p_dim)

    attractor = Attractor(num_iterations=3)
    output = attractor(query, memory, kappa=0.8)

    assert output.shape == (batch_size, p_dim)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_attractor_rejects_wrong_memory_shape() -> None:
    batch_size = 2
    p_dim = 12

    query = torch.randn(batch_size, p_dim)
    memory = torch.randn(batch_size, p_dim, p_dim + 1)

    attractor = Attractor(num_iterations=5)

    try:
        attractor(query, memory, kappa=0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("Attractor should reject wrong memory shape.")
