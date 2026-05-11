from __future__ import annotations

from pathlib import Path

from tem.config import load_config


def test_load_base_config() -> None:
    config_path = Path("configs/tem_base.yaml")
    config = load_config(config_path)

    assert config.seed == 42
    assert config.data.batch_size == 8
    assert config.data.sequence_length == 32
    assert config.data.num_observations == 45
    assert config.data.num_actions == 4

    assert config.model.num_modules == 3
    assert config.model.compressed_x_dim == 16

    assert len(config.model.g_dims) == config.model.num_modules
    assert len(config.model.phase_dims) == config.model.num_modules
    assert len(config.model.sensory_filter_alphas) == config.model.num_modules
    assert len(config.model.sensory_scales) == config.model.num_modules
    assert len(config.model.attractor_kappas) == config.model.num_modules
