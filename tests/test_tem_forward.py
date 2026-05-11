import torch
import torch.nn.functional as F

from tem.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    TEMConfig,
    TrainingConfig,
)
from tem.models.tem import TEM


def make_test_config() -> TEMConfig:
    """
    Create a small TEM config for fast unit tests.

    The dimensions are intentionally small but nontrivial:

        F = 2 modules
        C = 3 compressed sensory channels

        module 0:
            G_0   = 5
            Phi_0 = 2
            P_0   = 2 * 3 = 6

        module 1:
            G_1   = 6
            Phi_1 = 4
            P_1   = 4 * 3 = 12
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
    Create simple one-hot inputs.

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

    return x, a, visited


def test_tem_forward_observation_shapes() -> None:
    """
    The model should return observation logits and probabilities
    with shape [B, T, N_x].
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited = make_inputs(config)

    output = model(x, a, visited)

    expected_shape = (
        config.data.batch_size,
        config.data.sequence_length,
        config.data.num_observations,
    )

    assert output.x_p_logits.shape == expected_shape
    assert output.x_g_logits.shape == expected_shape
    assert output.x_gt_logits.shape == expected_shape

    assert output.x_p.shape == expected_shape
    assert output.x_g.shape == expected_shape
    assert output.x_gt.shape == expected_shape


def test_tem_forward_observation_probabilities_sum_to_one() -> None:
    """
    x_p, x_g, and x_gt are softmax probabilities.
    Their last dimension should sum to one.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited = make_inputs(config)

    output = model(x, a, visited)

    expected_sums = torch.ones(
        config.data.batch_size,
        config.data.sequence_length,
    )

    assert torch.allclose(
        output.x_p.sum(dim=-1),
        expected_sums,
        atol=1e-5,
    )
    assert torch.allclose(
        output.x_g.sum(dim=-1),
        expected_sums,
        atol=1e-5,
    )
    assert torch.allclose(
        output.x_gt.sum(dim=-1),
        expected_sums,
        atol=1e-5,
    )


def test_tem_forward_module_shapes() -> None:
    """
    Check all module-wise recurrent outputs.

    For each module f:

        g[f]:     [B, T, G_f]
        g_gen[f]: [B, T, G_f]

        p[f]:     [B, T, P_f]
        p_x[f]:   [B, T, P_f]
        p_g[f]:   [B, T, P_f]
        p_gt[f]:  [B, T, P_f]

        x_s[f]:   [B, T, C]
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited = make_inputs(config)

    output = model(x, a, visited)

    batch_size = config.data.batch_size
    sequence_length = config.data.sequence_length
    compressed_x_dim = config.model.compressed_x_dim

    assert len(output.g) == config.model.num_modules
    assert len(output.g_gen) == config.model.num_modules

    assert len(output.p) == config.model.num_modules
    assert len(output.p_x) == config.model.num_modules
    assert len(output.p_g) == config.model.num_modules
    assert len(output.p_gt) == config.model.num_modules

    assert len(output.x_s) == config.model.num_modules

    for f in range(config.model.num_modules):
        g_dim = config.model.g_dims[f]
        phase_dim = config.model.phase_dims[f]
        p_dim = phase_dim * compressed_x_dim

        expected_g_shape = (
            batch_size,
            sequence_length,
            g_dim,
        )

        expected_p_shape = (
            batch_size,
            sequence_length,
            p_dim,
        )

        expected_x_s_shape = (
            batch_size,
            sequence_length,
            compressed_x_dim,
        )

        assert output.g[f].shape == expected_g_shape
        assert output.g_gen[f].shape == expected_g_shape

        assert output.p[f].shape == expected_p_shape
        assert output.p_x[f].shape == expected_p_shape
        assert output.p_g[f].shape == expected_p_shape
        assert output.p_gt[f].shape == expected_p_shape

        assert output.x_s[f].shape == expected_x_s_shape


def test_tem_forward_final_state_shapes() -> None:
    """
    Check final recurrent state shapes.

    For each module f:

        final_state.g[f]:          [B, G_f]
        final_state.x_s[f]:        [B, C]
        final_state.memory_gen[f]: [B, P_f, P_f]
        final_state.memory_inf[f]: [B, P_f, P_f]
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, visited = make_inputs(config)

    output = model(x, a, visited)

    batch_size = config.data.batch_size
    compressed_x_dim = config.model.compressed_x_dim

    final_state = output.final_state

    for f in range(config.model.num_modules):
        g_dim = config.model.g_dims[f]
        phase_dim = config.model.phase_dims[f]
        p_dim = phase_dim * compressed_x_dim

        assert final_state.g[f].shape == (batch_size, g_dim)
        assert final_state.x_s[f].shape == (batch_size, compressed_x_dim)
        assert final_state.memory_gen[f].shape == (batch_size, p_dim, p_dim)
        assert final_state.memory_inf[f].shape == (batch_size, p_dim, p_dim)


def test_tem_forward_with_zero_visited_keeps_zero_memory() -> None:
    """
    If visited is all zero and the initial memory is zero,
    the final memory should remain exactly zero.

    The model can still update g and x_s, but memory updates are masked out.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, _ = make_inputs(config)

    visited = torch.zeros(
        config.data.batch_size,
        config.data.sequence_length,
    )

    output = model(x, a, visited)

    for f in range(config.model.num_modules):
        assert torch.allclose(
            output.final_state.memory_gen[f],
            torch.zeros_like(output.final_state.memory_gen[f]),
        )

        assert torch.allclose(
            output.final_state.memory_inf[f],
            torch.zeros_like(output.final_state.memory_inf[f]),
        )


def test_tem_forward_accepts_bool_visited() -> None:
    """
    visited can be a bool tensor.

    Internally the model converts it to the same floating dtype as x.
    """
    torch.manual_seed(0)

    config = make_test_config()
    model = TEM(config)

    x, a, _ = make_inputs(config)

    visited = torch.ones(
        config.data.batch_size,
        config.data.sequence_length,
        dtype=torch.bool,
    )

    output = model(x, a, visited)

    assert output.x_p.shape == (
        config.data.batch_size,
        config.data.sequence_length,
        config.data.num_observations,
    )
