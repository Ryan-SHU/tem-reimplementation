"""
python scripts/analyze_representations.py \
  --config configs/tem_base.yaml \
  --checkpoint runs/rectangle_debug/checkpoints/latest.pt \
  --output-dir outputs/rectangle_debug_analysis \
  --height 6 \
  --width 6 \
  --num-batches 20

"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from tem_data.batches import RandomWalkBatcher
from tem_data.environments import RectangleEnvironment
from tem.config import load_config
from tem.models.tem import TEM
from tem.utils.seed import set_seed
from tem_training.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for representation analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze TEM grid representations on a rectangle environment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tem_base.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rectangle_analysis",
        help="Directory for analysis outputs.",
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
        "--num-batches",
        type=int,
        default=20,
        help="Number of random-walk batches to collect representations from.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. If omitted, use config.device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, use config.seed.",
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
    model.to(device)

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        map_location=device,
    )

    checkpoint_step = int(checkpoint.get("step", -1))

    model.eval()

    analysis = collect_state_g_means(
        model=model,
        batcher=batcher,
        environment=environment,
        num_batches=args.num_batches,
        device=device,
    )

    pt_path = output_dir / "state_g_means.pt"
    csv_path = output_dir / "state_g_means.csv"
    metadata_path = output_dir / "analysis_metadata.json"

    torch.save(analysis, pt_path)

    write_state_g_means_csv(
        path=csv_path,
        environment=environment,
        state_counts=analysis["state_counts"],
        mean_g=analysis["mean_g"],
    )

    metadata = {
        "args": vars(args),
        "config": asdict(config),
        "seed": seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Analysis finished.")
    print(f"Checkpoint step: {checkpoint_step}")
    print(f"Saved tensor output: {pt_path}")
    print(f"Saved CSV output: {csv_path}")


def collect_state_g_means(
    model: TEM,
    batcher: RandomWalkBatcher,
    environment: RectangleEnvironment,
    num_batches: int,
    device: torch.device,
) -> dict:
    """
    Collect average grid-state representation for each environment state.

    For each module f, we compute:

        mean_g[f][s] = average g_t[f] over all visits to state s

    Shapes:

        state_counts:
            [num_states]

        mean_g[f]:
            [num_states, G_f]

    The state id is not given to the model.
    It is only used here for post-hoc analysis.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    num_states = environment.num_states

    state_counts = torch.zeros(
        num_states,
        device=device,
        dtype=torch.float32,
    )

    g_sums = []

    for g_dim in model.g_dims:
        g_sums.append(
            torch.zeros(
                num_states,
                g_dim,
                device=device,
                dtype=torch.float32,
            )
        )

    with torch.no_grad():
        for _ in range(num_batches):
            batch = batcher.sample_batch()

            output = model(
                x=batch["x"],
                a=batch["a"],
                visited=batch["visited"],
            )

            position = batch["position"].reshape(-1).long()

            ones = torch.ones_like(
                position,
                dtype=torch.float32,
                device=device,
            )

            state_counts.index_add_(
                dim=0,
                index=position,
                source=ones,
            )

            for f in range(model.num_modules):
                g_flat = output.g[f].reshape(
                    -1,
                    model.g_dims[f],
                )

                g_sums[f].index_add_(
                    dim=0,
                    index=position,
                    source=g_flat,
                )

    safe_counts = state_counts.clamp_min(1.0).unsqueeze(1)

    mean_g = []

    for f in range(model.num_modules):
        mean_g.append(g_sums[f] / safe_counts)

    return {
        "state_counts": state_counts.detach().cpu(),
        "mean_g": [
            value.detach().cpu()
            for value in mean_g
        ],
    }


def write_state_g_means_csv(
    path: Path,
    environment: RectangleEnvironment,
    state_counts: torch.Tensor,
    mean_g: list[torch.Tensor],
) -> None:
    """
    Write state-wise mean g representations to a CSV file.

    Each row corresponds to one environment state.

    Columns:

        state
        row
        col
        count
        module_0_g_0
        module_0_g_1
        ...
        module_1_g_0
        ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "state",
        "row",
        "col",
        "count",
    ]

    for f, mean_g_f in enumerate(mean_g):
        for unit in range(mean_g_f.shape[1]):
            fieldnames.append(f"module_{f}_g_{unit}")

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for state in range(environment.num_states):
            row = state // environment.width
            col = state % environment.width

            csv_row = {
                "state": state,
                "row": row,
                "col": col,
                "count": float(state_counts[state].item()),
            }

            for f, mean_g_f in enumerate(mean_g):
                for unit in range(mean_g_f.shape[1]):
                    csv_row[f"module_{f}_g_{unit}"] = float(
                        mean_g_f[state, unit].item()
                    )

            writer.writerow(csv_row)


def resolve_device(device_name: str) -> torch.device:
    """
    Resolve requested device.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_name)


if __name__ == "__main__":
    main()
