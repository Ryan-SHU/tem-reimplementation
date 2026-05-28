"""Full training + evaluation loop for Family Tree."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

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
from tem_experiments.exp1_generalization.line_ti.helpers import (
    evaluate_continuous,
)


@dataclass
class FamilyTreeResult:
    final_step: int
    history: List[Dict[str, float]]
    eval_history: List[Dict[str, float]]
    checkpoint_path: Optional[Path]
    final_bins: Optional[List[Dict]] = field(default=None)


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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate,
    )

    train_logger = CSVLogger(output_dir / "train_metrics.csv")
    eval_logger = CSVLogger(output_dir / "eval_metrics.csv")

    history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    last_ckpt: Optional[Path] = None
    last_bins: Optional[List[Dict]] = None

    print(f"Family Tree experiment — {num_steps} steps, device={device}")
    t0 = time.time()

    for step in tqdm(range(1, num_steps + 1), desc="Family Tree", unit="step"):
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
        output = model(
            x=batch["x"], a=batch["a"], visited=batch["visited"],
        )
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

        if step == 1 or step % log_every == 0 or step == num_steps:
            m = losses.detached()
            m["step"] = step
            m["grad_norm"] = float(grad_norm)
            m["elapsed"] = time.time() - t0
            train_logger.log(m)
            history.append(m)
            print(format_metrics(m))

        if step % eval_every == 0 or step == num_steps:
            eval_env = FamilyTreeEnvironment(
                num_observations=config.data.num_observations,
            )
            walk_len = max(EVAL_EXPLORE_LEN * 3, 20)

            full_result = evaluate_continuous(
                model=model,
                env=eval_env,
                walk_len=walk_len,
                num_episodes=eval_episodes,
                device=device,
                generator=gen,
            )

            eval_result: Dict[str, float] = {
                "step": step,
                "zero_shot_accuracy": full_result["overall_unseen_accuracy"],
                "seen_accuracy": full_result["overall_seen_accuracy"],
                "zero_shot_correct": full_result["overall_unseen_correct"],
                "zero_shot_total": full_result["overall_unseen_total"],
                "seen_correct": full_result["overall_seen_correct"],
                "seen_total": full_result["overall_seen_total"],
            }

            eval_logger.log(eval_result)
            eval_history.append(eval_result)

            last_bins = full_result["bins"]
            bin_path = output_dir / f"eval_bins_step_{step:07d}.json"
            with bin_path.open("w", encoding="utf-8") as f:
                json.dump(last_bins, f, indent=2)

            print(
                f"  [eval] step={step} "
                f"zs={eval_result['zero_shot_accuracy']:.4f} "
                f"seen={eval_result['seen_accuracy']:.4f} "
                f"zs_n={eval_result['zero_shot_total']} "
                f"seen_n={eval_result['seen_total']}"
            )

        if checkpoint_every > 0 and (
            step % checkpoint_every == 0 or step == num_steps
        ):
            p = ckpt_dir / f"step_{step:07d}.pt"
            save_checkpoint(p, model, optimizer, step, config)
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, step, config,
            )
            last_ckpt = p

    return FamilyTreeResult(
        final_step=num_steps,
        history=history,
        eval_history=eval_history,
        checkpoint_path=last_ckpt,
        final_bins=last_bins,
    )
