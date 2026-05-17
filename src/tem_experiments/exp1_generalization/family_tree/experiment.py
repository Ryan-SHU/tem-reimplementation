"""
Full training + evaluation loop for Family Tree.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tem.config import TEMConfig
from tem.losses import compute_tem_loss
from tem.models.tem import TEM
from tem_data.envs.family_tree import FamilyTreeEnvironment
from tem_data.sampling.batches import WalkBatcher
from tem_training.checkpointing import save_checkpoint
from tem_training.logging import CSVLogger, format_metrics

from tem_experiments.exp1_generalization.family_tree.definitions import (
    EVAL_EPISODES,
    EVAL_EXPLORE_LEN,
    NUM_OBSERVATIONS,
    TRAIN_STEPS,
)
from tem_experiments.exp1_generalization.line_ti.helpers import evaluate_zero_shot


@dataclass
class FamilyTreeResult:
    final_step: int
    history: List[Dict[str, float]]
    eval_history: List[Dict[str, float]]
    checkpoint_path: Optional[Path]


def run_family_tree(
    config: TEMConfig,
    output_dir: Path,
    device: torch.device,
    num_steps: Optional[int] = None,
    eval_every: int = 500,
    log_every: int = 50,
    checkpoint_every: int = 2000,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = 0,
) -> FamilyTreeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if num_steps is None:
        num_steps = TRAIN_STEPS

    gen = torch.Generator()
    gen.manual_seed(seed)

    env = FamilyTreeEnvironment(
        num_observations=config.data.num_observations,
        generator=gen,
    )

    model = TEM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_logger = CSVLogger(output_dir / "train_metrics.csv")
    eval_logger = CSVLogger(output_dir / "eval_metrics.csv")

    history, eval_history = [], []
    last_ckpt = None

    print(f"Family Tree experiment — {num_steps} steps, device={device}")
    t0 = time.time()

    for step in range(1, num_steps + 1):
        model.train()
        env.resample_observations(gen)

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

        optimizer.zero_grad(set_to_none=True)
        output = model(x=batch["x"], a=batch["a"], visited=batch["visited"])
        losses = compute_tem_loss(output=output, target_x=batch["x"], loss_config=config.loss)

        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"NaN/Inf loss at step {step}")

        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == num_steps:
            m = losses.detached()
            m["step"] = step
            m["grad_norm"] = float(grad_norm)
            m["elapsed"] = time.time() - t0
            train_logger.log(m)
            history.append(m)
            print(format_metrics(m))

        if step % eval_every == 0 or step == num_steps:
            eval_env = FamilyTreeEnvironment(num_observations=config.data.num_observations)
            er = evaluate_zero_shot(
                model=model, env=eval_env, explore_len=EVAL_EXPLORE_LEN,
                num_episodes=eval_episodes, device=device, generator=gen,
            )
            er["step"] = step
            eval_logger.log(er)
            eval_history.append(er)
            print(f"  [eval] step={step} zs={er['zero_shot_accuracy']:.4f} seen={er['seen_accuracy']:.4f}")

        if checkpoint_every > 0 and (step % checkpoint_every == 0 or step == num_steps):
            p = ckpt_dir / f"step_{step:07d}.pt"
            save_checkpoint(p, model, optimizer, step, config)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, config)
            last_ckpt = p

    return FamilyTreeResult(
        final_step=num_steps, history=history,
        eval_history=eval_history, checkpoint_path=last_ckpt,
    )
