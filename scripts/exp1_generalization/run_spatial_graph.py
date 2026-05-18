"""
Run Experiment 1C: Spatial Graph structural generalization.

Usage:
    python scripts/exp1_generalization/run_spatial_graph.py \
        --config configs/experiments/exp1_spatial_graph.yaml \
        --output-dir runs/exp1_spatial_graph \
        --steps 10000 --height 6 --width 6
"""

import argparse
from pathlib import Path

import torch

from tem.config import load_config
from tem.utils.seed import set_seed
from tem_experiments.exp1_generalization.spatial_graph.experiment import (
    run_spatial_graph,
)
from tem_experiments.exp1_generalization.spatial_graph.plotting import (
    plot_spatial_graph_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1C: Spatial Graph")
    parser.add_argument(
        "--config", type=str,
        default="configs/experiments/exp1_spatial_graph.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/exp1_spatial_graph",
    )
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=6)
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
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )

    result = run_spatial_graph(
        config=config,
        output_dir=Path(args.output_dir),
        device=device,
        num_steps=args.steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        eval_episodes=args.eval_episodes,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    plot_dir = Path(args.output_dir) / "plots"
    plot_spatial_graph_results(
        train_history=result.history,
        eval_history=result.eval_history,
        output_dir=plot_dir,
        final_bins=result.final_bins,
    )

    print(f"\nDone. Final step: {result.final_step}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
