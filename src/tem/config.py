from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    batch_size: int
    sequence_length: int
    num_observations: int
    num_actions: int


@dataclass
class ModelConfig:
    num_modules: int
    compressed_x_dim: int

    g_dims: list[int]
    phase_dims: list[int]

    sensory_filter_alphas: list[float]
    sensory_scales: list[float]

    attractor_iterations: int
    attractor_kappas: list[float]

    memory_eta_gen: float
    memory_eta_inf: float
    memory_lambda_gen: float
    memory_lambda_inf: float
    memory_clip: float


@dataclass
class TrainingConfig:
    learning_rate: float
    num_steps: int
    grad_clip_norm: float


@dataclass
class LossConfig:
    beta_x_p: float
    beta_x_g: float
    beta_x_gt: float
    beta_p: float
    beta_px: float
    beta_g: float
    beta_g_reg: float
    beta_p_reg: float


@dataclass
class TEMConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig


def load_config(path: str | Path) -> TEMConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    config = TEMConfig(
        seed=raw["seed"],
        device=raw["device"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
    )

    validate_config(config)

    return config


def validate_config(config: TEMConfig) -> None:
    model = config.model

    if model.num_modules <= 0:
        raise ValueError("model.num_modules must be positive.")

    expected_length = model.num_modules

    module_lists = {
        "g_dims": model.g_dims,
        "phase_dims": model.phase_dims,
        "sensory_filter_alphas": model.sensory_filter_alphas,
        "sensory_scales": model.sensory_scales,
        "attractor_kappas": model.attractor_kappas,
    }

    for name, values in module_lists.items():
        if len(values) != expected_length:
            raise ValueError(
                f"model.{name} must have length {expected_length}, "
                f"but got length {len(values)}."
            )

    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")

    if config.data.sequence_length <= 0:
        raise ValueError("data.sequence_length must be positive.")

    if config.data.num_observations <= 1:
        raise ValueError("data.num_observations must be greater than 1.")

    if config.data.num_actions <= 0:
        raise ValueError("data.num_actions must be positive.")

    if model.compressed_x_dim <= 0:
        raise ValueError("model.compressed_x_dim must be positive.")

    for i, value in enumerate(model.g_dims):
        if value <= 0:
            raise ValueError(f"model.g_dims[{i}] must be positive.")

    for i, value in enumerate(model.phase_dims):
        if value <= 0:
            raise ValueError(f"model.phase_dims[{i}] must be positive.")

    for i, value in enumerate(model.sensory_filter_alphas):
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"model.sensory_filter_alphas[{i}] must be in (0, 1), "
                f"but got {value}."
            )
