import torch
import torch.nn.functional as F
import pytest

from tem.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    TEMConfig,
    TrainingConfig,
)
from tem.losses import compute_tem_loss
from tem.models.tem import TEM


def make_test_config() -> TEMConfig:
    """
    Create a small config for fast loss tests.

    Dimensions:

        B = 2
        T = 5
        N_x = 7
        N_a = 4
        F = 2
        C = 3

    Module 0:
        G_0 = 5
        Phi_0 = 2
        P_0 = 2 * 3 = 6

    Module 1:
        G_1 = 6
        Phi_1 = 4
        P_1 = 4 * 3 = 12
    """
    return TEMConfig(
        seed=0,
        device="cpu",
        data=DataConfig(
            batch_size=2,
            sequence_length=5,
            num_observations=7,
            num_actions=4,
        ),
        model=ModelConfig(
            num_modules=2,
            compressed_x_dim=3,
            g_dims=[5, 6],
            phase_dims=[2, 4],
            sensory_filter_alphas=[0.4, 0.6],
            sensory_scales=[1.0, 0.8],
            attractor_iterations=2,
            attractor_kappas=[0.7, 0.9],
            memory_eta_gen=0.2,
            memory_eta_inf=0.2,
            memory_lambda_gen=0.95,
            memory_lambda_inf=0.95,
            memory_clip=1.0,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            num_steps=10,
            grad_clip_norm=1.0,
        ),
        loss=LossConfig(
            beta_x_p=1.0,
            beta_x_g=1.0,
            beta_x_gt=1.0,
            beta_p=1.0,
            beta_px=1.0,
            beta_g=1.0,
            beta_g_reg=0.01,
            beta_p_reg=0.01,
        ),
    )


def make_inputs(config: TEMConfig):
    """
    Create one-hot observations and actions.

    x:
        [B, T, N_x]

    a:
        [B, T, N_a]

    visited:
        [B, T]
    """
    batch_size = config.data.batch_size
    sequence_length = config.data.sequence_length
    num_observations = config.data.num_observations
    num_actions = config.data.num_actions

    observation_ids = torch.randint(
        low=0,
        high=num_observations,
        size=(batch_size, sequence_length),
    )

    action_ids = torch.randint(
        low=0,
        high=num_actions,
        size=(batch_size, sequence_length),
    )

    x = F.one_hot(
        observation_ids,
        num_classes=num_observations,
    ).float()

    a = F.one_hot(
        action_ids,
        num_classes=num_actions,
    ).float()

    visited = torch.ones(batch_size, sequence_length)

    return x, a, visited, observation_ids


def test_compute_tem_loss_returns_scalar_components() -> None:
    """
    The full loss function should return scalar tensors for all components.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)
    losses = compute_tem_loss(output, x, config.loss)

    for name, value in losses.as_dict().items():
        assert isinstance(value, torch.Tensor), name
        assert value.dim() == 0, name
        assert torch.isfinite(value), name


def test_compute_tem_loss_accepts_full_config_object() -> None:
    """
    compute_tem_loss should accept either config.loss or the full config.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)

    losses_from_loss_config = compute_tem_loss(output, x, config.loss)
    losses_from_full_config = compute_tem_loss(output, x, config)

    assert torch.allclose(
        losses_from_loss_config.total,
        losses_from_full_config.total,
    )


def test_compute_tem_loss_accepts_class_index_targets() -> None:
    """
    target_x can be either one-hot [B, T, N_x] or class indices [B, T].

    The two forms should produce the same loss when they represent the same
    target classes.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, observation_ids = make_inputs(config)

    output = model(x, a, visited)

    losses_from_one_hot = compute_tem_loss(output, x, config.loss)
    losses_from_indices = compute_tem_loss(output, observation_ids, config.loss)

    assert torch.allclose(
        losses_from_one_hot.total,
        losses_from_indices.total,
        atol=1e-6,
    )


def test_compute_tem_loss_backward() -> None:
    """
    The total loss should support backpropagation through the TEM model.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)
    losses = compute_tem_loss(output, x, config.loss)

    losses.total.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(grad is not None for grad in gradients)

    for grad in gradients:
        if grad is not None:
            assert torch.all(torch.isfinite(grad))


def test_compute_tem_loss_with_all_zero_mask_is_zero() -> None:
    """
    If the loss mask is all zero, every component should evaluate to zero.

    This is useful for padded batches where some sequences may contribute no
    valid tokens.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)

    mask = torch.zeros(
        config.data.batch_size,
        config.data.sequence_length,
    )

    losses = compute_tem_loss(output, x, config.loss, mask=mask)

    for name, value in losses.as_dict().items():
        assert torch.allclose(value, torch.zeros_like(value)), name


def test_compute_tem_loss_with_partial_mask() -> None:
    """
    A partial mask should produce a finite scalar loss.

    This checks the common variable-length sequence case.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)

    mask = torch.ones(
        config.data.batch_size,
        config.data.sequence_length,
    )

    mask[:, -2:] = 0.0

    losses = compute_tem_loss(output, x, config.loss, mask=mask)

    assert losses.total.dim() == 0
    assert torch.isfinite(losses.total)


def test_compute_tem_loss_rejects_wrong_target_shape() -> None:
    """
    target_x must have shape [B, T] or [B, T, N_x].
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)

    wrong_target = torch.zeros(
        config.data.batch_size,
        config.data.sequence_length,
        config.data.num_observations + 1,
    )

    with pytest.raises(ValueError):
        compute_tem_loss(output, wrong_target, config.loss)


def test_compute_tem_loss_rejects_wrong_mask_shape() -> None:
    """
    mask must have shape [B, T].
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)

    wrong_mask = torch.ones(
        config.data.batch_size,
        config.data.sequence_length,
        1,
    )

    with pytest.raises(ValueError):
        compute_tem_loss(output, x, config.loss, mask=wrong_mask)


def test_loss_breakdown_detached_returns_floats() -> None:
    """
    detached() should return Python floats for logging.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited, _ = make_inputs(config)

    output = model(x, a, visited)
    losses = compute_tem_loss(output, x, config.loss)

    detached = losses.detached()

    assert isinstance(detached, dict)

    for name, value in detached.items():
        assert isinstance(name, str)
        assert isinstance(value, float)
