"""
Single entry point for Experiment 1A: Line TI.

Modes
-----
train
    Train the model and save checkpoints.

python scripts/exp1_generalization/run_line_ti.py \
  --mode train \
  --config configs/experiments/exp1_line_ti.yaml \
  --run-dir runs/exp1_line_ti \
  --steps 10000 \
  --log-every 50 \
  --checkpoint-every 500


eval
    Evaluate saved checkpoints offline and save raw event CSVs + summary CSV.

python scripts/exp1_generalization/run_line_ti.py \
  --mode eval \
  --config configs/experiments/exp1_line_ti.yaml \
  --run-dir runs/exp1_line_ti \
  --checkpoint-selector all \
  --eval-episodes 200 \
  --eval-walk-len 36


plot
    Read saved training/evaluation CSV files and generate diagnostic + paper-style plots.

python scripts/exp1_generalization/run_line_ti.py \
  --mode plot \
  --run-dir runs/exp1_line_ti \
  --num-training-quantiles 5


all
    Run train -> eval -> plot in sequence.

python scripts/exp1_generalization/run_line_ti.py \
  --mode all \
  --config configs/experiments/exp1_line_ti.yaml \
  --run-dir runs/exp1_line_ti \
  --steps 10000 \
  --checkpoint-every 500 \
  --eval-episodes 200 \
  --eval-walk-len 36 \
  --num-training-quantiles 5


"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from tem.config import load_config
from tem.utils.seed import set_seed

from tem_experiments.exp1_generalization.line_ti.definitions import (
    DEFAULT_CHECKPOINT_EVERY,
    EVAL_EPISODES,
    EVAL_WALK_LEN,
    NUM_TRAINING_QUANTILES,
)
from tem_experiments.exp1_generalization.line_ti.evaluation import (
    evaluate_line_ti_checkpoints,
)
from tem_experiments.exp1_generalization.line_ti.experiment import (
    run_line_ti_train,
)
from tem_experiments.exp1_generalization.line_ti.plotting import (
    make_all_line_ti_plots,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_checkpoint_steps(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1A: Line TI")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "eval", "plot", "all"],
        help="Which stage to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_line_ti.yaml",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/exp1_line_ti",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)

    # training args
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )

    # evaluation args
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-walk-len", type=int, default=EVAL_WALK_LEN)
    parser.add_argument(
        "--checkpoint-selector",
        type=str,
        default="all",
        choices=["all", "latest"],
        help="Which checkpoints to evaluate if --checkpoint-steps is not given.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps to evaluate, e.g. 500,1000,5000",
    )

    # plotting args
    parser.add_argument(
        "--num-training-quantiles",
        type=int,
        default=NUM_TRAINING_QUANTILES,
    )

    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = _resolve_device(args.device)
    run_dir = Path(args.run_dir)

    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)

    if args.mode in {"train", "all"}:
        print("\n=== TRAIN ===")
        train_result = run_line_ti_train(
            config=config,
            output_dir=run_dir,
            device=device,
            num_steps=args.steps,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
        )
        print(f"Training done. Final step: {train_result.final_step}")
        if train_result.checkpoint_path is not None:
            print(f"Last checkpoint: {train_result.checkpoint_path}")

    if args.mode in {"eval", "all"}:
        print("\n=== EVAL ===")
        eval_result = evaluate_line_ti_checkpoints(
            config=config,
            run_dir=run_dir,
            device=device,
            selector=args.checkpoint_selector,
            checkpoint_steps=checkpoint_steps,
            eval_episodes=args.eval_episodes,
            walk_len=args.eval_walk_len,
            seed=args.seed,
        )
        print(f"Evaluation done. Summary: {eval_result.summary_csv_path}")

    if args.mode in {"plot", "all"}:
        print("\n=== PLOT ===")
        paths = make_all_line_ti_plots(
            run_dir=run_dir,
            num_training_quantiles=args.num_training_quantiles,
        )
        print("Generated plots:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
