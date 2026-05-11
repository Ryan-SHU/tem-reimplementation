"""
python scripts/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_debug \
  --height 6 \
  --width 6 \
  --steps 100 \
  --log-every 10 \
  --checkpoint-every 50


recover from checkpoint (steps)

python scripts/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_debug \
  --resume auto \
  --steps 100

means keep training for 100 steps from checkpoint

"""


import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from data.batches import RandomWalkBatcher
from data.environments import RectangleEnvironment
from tem.config import load_config
from tem.models.tem import TEM
from tem.utils.seed import set_seed
from training.checkpointing import find_latest_checkpoint, load_checkpoint
from training.trainer import TEMTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the rectangle experiment.
    """
    parser = argparse.ArgumentParser(
        description="Train TEM on a rectangular random-walk environment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tem_base.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/rectangle",
        help="Directory for metrics and checkpoints.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=6,
        help="Rectangle environment height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=6,
        help="Rectangle environment width.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of training steps to run. "
            "If omitted, use config.training.num_steps."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log metrics every N steps.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to use. Examples: cpu, cuda. "
            "If omitted, use config.device."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, use config.seed.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint path to resume from. "
            "Use 'auto' to resume from output-dir/checkpoints/latest.pt."
        ),
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    device_name = args.device if args.device is not None else config.device
    device = resolve_device(device_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_metadata(
        output_dir=output_dir,
        args=args,
        config=config,
        seed=seed,
        device=device,
    )

    environment = RectangleEnvironment(
        height=args.height,
        width=args.width,
        num_observations=config.data.num_observations,
        seed=seed,
    )

    if config.data.num_actions != environment.num_actions:
        raise ValueError(
            "config.data.num_actions must match the environment. "
            f"Config has {config.data.num_actions}, "
            f"but RectangleEnvironment uses {environment.num_actions}."
        )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        device=device,
        seed=seed,
    )

    model = TEM(config)

    trainer = TEMTrainer(
        model=model,
        config=config,
        batcher=batcher,
        output_dir=output_dir,
        device=device,
    )

    start_step = 0

    resume_path = resolve_resume_path(
        resume_arg=args.resume,
        output_dir=output_dir,
    )

    if resume_path is not None:
        checkpoint = load_checkpoint(
            path=resume_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            map_location=device,
        )

        start_step = int(checkpoint.get("step", 0))

        print(f"Resumed from checkpoint: {resume_path}")
        print(f"Starting from global step: {start_step}")

    result = trainer.train(
        num_steps=args.steps,
        start_step=start_step,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        show_progress=not args.no_progress,
        save_final_checkpoint=True,
    )

    print("Training finished.")
    print(f"Final step: {result.final_step}")

    if result.last_checkpoint_path is not None:
        print(f"Last checkpoint: {result.last_checkpoint_path}")

    print(f"Metrics CSV: {output_dir / 'metrics.csv'}")


def resolve_device(device_name: str) -> torch.device:
    """
    Resolve requested device.

    If cuda is requested but unavailable, fall back to CPU.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_name)


def resolve_resume_path(
    resume_arg: Optional[str],
    output_dir: Path,
) -> Optional[Path]:
    """
    Resolve checkpoint path for resuming.
    """
    if resume_arg is None:
        return None

    if resume_arg == "auto":
        latest = find_latest_checkpoint(output_dir / "checkpoints")

        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found in {output_dir / 'checkpoints'}."
            )

        return latest

    path = Path(resume_arg)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config,
    seed: int,
    device: torch.device,
) -> None:
    """
    Save experiment metadata.

    This makes each run easier to reproduce later.
    """
    metadata = {
        "args": vars(args),
        "config": asdict(config),
        "seed": seed,
        "device": str(device),
    }

    path = output_dir / "run_metadata.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
