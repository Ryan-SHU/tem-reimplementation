"""
Run Experiment 1A: Transitive Inference on line graph.

Usage:
    python scripts/exp1_generalization/run_line_ti.py \
        --config configs/experiments/exp1_line_ti.yaml \
        --output-dir runs/exp1_line_ti \
        --steps 10000 \
        --eval-every 500

    # Quick debug run:
    python scripts/exp1_generalization/run_line_ti.py \
        --config configs/experiments/exp1_line_ti.yaml \
        --output-dir runs/exp1_line_ti_debug \
        --steps 50 --eval-every 25 --eval-episodes 10
"""

import argparse
from pathlib import Path

import torch

from tem.config import load_config
from tem.utils.seed import set_seed
from tem_experiments.exp1_generalization.line_ti.experiment import run_line_ti
from tem_experiments.exp1_generalization.line_ti.plotting import plot_line_ti_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1A: Line TI")
    parser.add_argument("--config", type=str, default="configs/experiments/exp1_line_ti.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/exp1_line_ti")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = _resolve_device(args.device)

    result = run_line_ti(
        config=config,
        output_dir=Path(args.output_dir),
        device=device,
        num_steps=args.steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    plot_dir = Path(args.output_dir) / "plots"
    plot_line_ti_results(result.history, result.eval_history, plot_dir)

    print(f"\nDone. Final step: {result.final_step}")
    print(f"Plots saved to: {plot_dir}")
    if result.checkpoint_path:
        print(f"Checkpoint: {result.checkpoint_path}")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
