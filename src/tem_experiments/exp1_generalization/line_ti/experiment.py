"""
Training loop for Experiment 1A: Line TI.

This file is intentionally train-only.

Responsibilities
----------------
- build environment
- build model
- generate batches
- optimize TEM objective
- save checkpoints
- write training metrics

It does NOT:
- run final paper evaluation
- aggregate zero-shot results
- generate paper figures

Those are handled offline by:
    evaluation.py
    plotting.py
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tem.config import TEMConfig
from tem.losses import compute_tem_loss
from tem.models.tem import TEM
from tem_data.envs.line import LineEnvironment
from tem_data.sampling.batches import WalkBatcher
from tem_training.checkpointing import save_checkpoint
from tem_training.logging import CSVLogger, format_metrics

from tem_experiments.exp1_generalization.line_ti.definitions import (
    NUM_NODES,
    TRAIN_STEPS,
)


@dataclass
class LineTITrainResult:
    final_step: int
    history: List[Dict[str, float]]
    checkpoint_path: Optional[Path]
    train_metrics_path: Path


def run_line_ti_train(
    config: TEMConfig,
    output_dir: Path,
    device: torch.device,
    num_steps: Optional[int] = None,
    log_every: int = 50,
    checkpoint_every: int = 500,
    seed: int = 0,
) -> LineTITrainResult:
    """
    Train TEM on the Line TI task.

    Each optimization step uses a fresh sensory re-assignment of the same line graph.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_metrics_path = output_dir / "train_metrics.csv"

    if num_steps is None:
        num_steps = TRAIN_STEPS

    world_gen = torch.Generator()
    world_gen.manual_seed(seed)

    env = LineEnvironment(
        num_nodes=NUM_NODES,
        num_observations=config.data.num_observations,
        generator=world_gen,
    )

    model = TEM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    logger = CSVLogger(train_metrics_path)

    history: List[Dict[str, float]] = []
    last_ckpt: Optional[Path] = None

    print(f"Line TI training — {num_steps} steps, device={device}")
    print(f"  num_nodes={NUM_NODES}, num_obs={config.data.num_observations}")
    print(
        f"  train_seq_len={config.data.sequence_length}, "
        f"batch={config.data.batch_size}"
    )

    t0 = time.time()

    for step in range(1, num_steps + 1):
        model.train()

        # new world each step
        env.resample_observations(world_gen)

        walk_gen = torch.Generator()
        walk_gen.manual_seed(seed * 100000 + step)

        batcher = WalkBatcher(
            env=env,
            batch_size=config.data.batch_size,
            seq_len=config.data.sequence_length,
            device=str(device),
            generator=walk_gen,
        )
        batch = batcher.sample_batch()

        optimizer.zero_grad(set_to_none=True)

        output = model(
            x=batch["x"],
            a=batch["a"],
            visited=batch["visited"],
        )

        losses = compute_tem_loss(
            output=output,
            target_x=batch["x"],
            loss_config=config.loss,
        )

        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"NaN/Inf loss at step {step}")

        losses.total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
        )

        optimizer.step()

        should_log = (step == 1 or step % log_every == 0 or step == num_steps)
        if should_log:
            metrics = losses.detached()
            metrics["step"] = step
            metrics["grad_norm"] = float(grad_norm)
            metrics["elapsed"] = time.time() - t0

            logger.log(metrics)
            history.append(metrics)
            print(format_metrics(metrics))

        should_ckpt = (
            checkpoint_every > 0
            and (step % checkpoint_every == 0 or step == num_steps)
        )
        if should_ckpt:
            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, config)
            last_ckpt = ckpt_path

    return LineTITrainResult(
        final_step=num_steps,
        history=history,
        checkpoint_path=last_ckpt,
        train_metrics_path=train_metrics_path,
    )
