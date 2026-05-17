from pathlib import Path

import torch
import torch.nn.functional as F

from tem_data.batches import RandomWalkBatcher
from tem_data.environments import RectangleEnvironment
from tem.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    TEMConfig,
    TrainingConfig,
)
from tem.models.tem import TEM
from tem_training.checkpointing import load_checkpoint
from tem_training.trainer import TEMTrainer


def make_test_config() -> TEMConfig:
    return TEMConfig(
        seed=0,
        device="cpu",
        data=DataConfig(
            batch_size=2,
            sequence_length=4,
            num_observations=8,
            num_actions=4,
        ),
        model=ModelConfig(
            num_modules=2,
            compressed_x_dim=3,
            g_dims=[4, 5],
            phase_dims=[2, 3],
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
            num_steps=2,
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


def test_trainer_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)

    config = make_test_config()

    environment = RectangleEnvironment(
        height=3,
        width=3,
        num_observations=config.data.num_observations,
        seed=0,
    )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        device="cpu",
        seed=0,
    )

    model = TEM(config)

    trainer = TEMTrainer(
        model=model,
        config=config,
        batcher=batcher,
        output_dir=tmp_path,
        device="cpu",
    )

    result = trainer.train(
        num_steps=2,
        log_every=1,
        checkpoint_every=1,
        show_progress=False,
        save_final_checkpoint=True,
    )

    assert result.final_step == 2
    assert result.last_checkpoint_path is not None
    assert result.last_checkpoint_path.exists()

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    torch.manual_seed(0)

    config = make_test_config()

    environment = RectangleEnvironment(
        height=3,
        width=3,
        num_observations=config.data.num_observations,
        seed=0,
    )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        device="cpu",
        seed=0,
    )

    model = TEM(config)

    trainer = TEMTrainer(
        model=model,
        config=config,
        batcher=batcher,
        output_dir=tmp_path,
        device="cpu",
    )

    result = trainer.train(
        num_steps=2,
        log_every=1,
        checkpoint_every=1,
        show_progress=False,
        save_final_checkpoint=True,
    )

    new_model = TEM(config)

    checkpoint = load_checkpoint(
        path=result.last_checkpoint_path,
        model=new_model,
        optimizer=None,
        map_location="cpu",
    )

    assert checkpoint["step"] == 2
