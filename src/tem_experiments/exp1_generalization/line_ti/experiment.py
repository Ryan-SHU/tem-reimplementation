"""
Full training + evaluation loop for Line TI.

This file ties together:
    - LineEnvironment
    - WalkBatcher
    - TEM model + loss
    - zero-shot evaluation
    - CSV logging + checkpointing
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
    EVAL_EPISODES,
    EVAL_EXPLORE_LEN,
    NUM_NODES,
    NUM_OBSERVATIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_SEQ_LEN,
    TRAIN_STEPS,
)
from tem_experiments.exp1_generalization.line_ti.helpers import evaluate_zero_shot


@dataclass
class LineTIResult:
    final_step: int
    history: List[Dict[str, float]]
    eval_history: List[Dict[str, float]]
    checkpoint_path: Optional[Path]


def run_line_ti(
    config: TEMConfig,
    output_dir: Path,
    device: torch.device,
    num_steps: Optional[int] = None,
    eval_every: int = 500,
    log_every: int = 50,
    checkpoint_every: int = 2000,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = 0,
) -> LineTIResult:
    """
    Complete Line TI experiment.

    1. Build LineEnvironment
    2. Build TEM model
    3. Training loop:
       - Each step: resample world, generate walk, forward, loss, backward
       - Periodically evaluate zero-shot accuracy
    4. Return results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if num_steps is None:
        num_steps = TRAIN_STEPS

    gen = torch.Generator()
    gen.manual_seed(seed)

    # ---- Environment ----
    env = LineEnvironment(
        num_nodes=NUM_NODES,
        num_observations=config.data.num_observations,
        generator=gen,
    )

    # ---- Model ----
    model = TEM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate,
    )

    # ---- Loggers ----
    train_logger = CSVLogger(output_dir / "train_metrics.csv")
    eval_logger = CSVLogger(output_dir / "eval_metrics.csv")

    history = []
    eval_history = []
    last_ckpt = None

    print(f"Line TI experiment — {num_steps} steps, device={device}")
    print(f"  num_nodes={NUM_NODES}, num_obs={config.data.num_observations}")
    print(f"  train_seq_len={config.data.sequence_length}, batch={config.data.batch_size}")

    t0 = time.time()

    for step in range(1, num_steps + 1):
        model.train()

        # ---- New world each step ----
        env.resample_observations(gen)

        # ---- Batcher (we build inline since world changes each step) ----
        walk_gen = torch.Generator()
        walk_gen.manual_seed(gen.seed() + step)

        batcher = WalkBatcher(
            env=env,
            batch_size=config.data.batch_size,
            seq_len=config.data.sequence_length,
            device=str(device),
            generator=walk_gen,
        )
        batch = batcher.sample_batch()

        # ---- Forward ----
        optimizer.zero_grad(set_to_none=True)
        output = model(x=batch["x"], a=batch["a"], visited=batch["visited"])
        losses = compute_tem_loss(
            output=output, target_x=batch["x"], loss_config=config.loss,
        )

        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"NaN/Inf loss at step {step}")

        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip_norm,
        )
        optimizer.step()

        # ---- Log ----
        should_log = (step == 1 or step % log_every == 0 or step == num_steps)
        if should_log:
            metrics = losses.detached()
            metrics["step"] = step
            metrics["grad_norm"] = float(grad_norm)
            metrics["elapsed"] = time.time() - t0
            train_logger.log(metrics)
            history.append(metrics)
            print(format_metrics(metrics))

        # ---- Evaluate ----
        should_eval = (step % eval_every == 0 or step == num_steps)
        if should_eval:
            eval_result = evaluate_zero_shot(
                model=model,
                env=LineEnvironment(
                    num_nodes=NUM_NODES,
                    num_observations=config.data.num_observations,
                ),
                explore_len=EVAL_EXPLORE_LEN,
                num_episodes=eval_episodes,
                device=device,
                generator=gen,
            )
            eval_result["step"] = step
            eval_logger.log(eval_result)
            eval_history.append(eval_result)
            print(
                f"  [eval] step={step} "
                f"zero_shot={eval_result['zero_shot_accuracy']:.4f} "
                f"seen={eval_result['seen_accuracy']:.4f} "
                f"nodes={eval_result['avg_nodes_visited']:.1f}"
            )

        # ---- Checkpoint ----
        should_ckpt = (
            checkpoint_every > 0
            and (step % checkpoint_every == 0 or step == num_steps)
        )
        if should_ckpt:
            path = ckpt_dir / f"step_{step:07d}.pt"
            save_checkpoint(path, model, optimizer, step, config)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, config)
            last_ckpt = path

    return LineTIResult(
        final_step=num_steps,
        history=history,
        eval_history=eval_history,
        checkpoint_path=last_ckpt,
    )
