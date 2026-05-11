import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for plotting rectangle experiment results.
    """
    parser = argparse.ArgumentParser(
        description="Plot TEM rectangle training and representation analysis results."
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="runs/rectangle_debug/metrics.csv",
        help="Path to training metrics.csv.",
    )

    parser.add_argument(
        "--state-g-csv",
        type=str,
        default="outputs/rectangle_debug_analysis/state_g_means.csv",
        help="Path to state-wise mean g representation CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rectangle_debug_plots",
        help="Directory where plots will be saved.",
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
        "--top-k-units",
        type=int,
        default=6,
        help="Number of most spatially varying g-units to plot per module.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_csv)
    state_g_path = Path(args.state_g_csv)

    generated_files = []

    if metrics_path.exists():
        metrics = read_metrics_csv(metrics_path)

        path = output_dir / "training_losses.png"
        plot_training_losses(metrics, path)
        generated_files.append(str(path))

        path = output_dir / "latent_losses.png"
        plot_latent_losses(metrics, path)
        generated_files.append(str(path))
    else:
        print(f"Metrics CSV not found, skipping training plots: {metrics_path}")

    if state_g_path.exists():
        state_rows = read_state_g_csv(state_g_path)

        path = output_dir / "state_visit_counts.png"
        plot_state_visit_counts(
            rows=state_rows,
            height=args.height,
            width=args.width,
            output_path=path,
        )
        generated_files.append(str(path))

        unit_paths = plot_top_spatial_units(
            rows=state_rows,
            height=args.height,
            width=args.width,
            output_dir=output_dir,
            top_k_units=args.top_k_units,
        )
        generated_files.extend([str(path) for path in unit_paths])
    else:
        print(f"State g CSV not found, skipping representation plots: {state_g_path}")

    metadata = {
        "metrics_csv": str(metrics_path),
        "state_g_csv": str(state_g_path),
        "output_dir": str(output_dir),
        "height": args.height,
        "width": args.width,
        "top_k_units": args.top_k_units,
        "generated_files": generated_files,
    }

    metadata_path = output_dir / "plot_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Plotting finished.")
    print(f"Saved plots to: {output_dir}")


def read_metrics_csv(path: Path) -> List[Dict[str, float]]:
    """
    Read trainer metrics from CSV.

    The trainer writes one row per logging step. This function converts
    numeric fields to floats whenever possible.
    """
    rows = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            clean_row = {}

            for key, value in row.items():
                clean_row[key] = parse_float_or_keep(value)

            rows.append(clean_row)

    if len(rows) == 0:
        raise ValueError(f"No rows found in metrics CSV: {path}")

    return rows


def read_state_g_csv(path: Path) -> List[Dict[str, float]]:
    """
    Read state-wise mean g representations from CSV.

    Expected columns include:

        state
        row
        col
        count
        module_0_g_0
        module_0_g_1
        ...
    """
    rows = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            clean_row = {}

            for key, value in row.items():
                clean_row[key] = parse_float_or_keep(value)

            rows.append(clean_row)

    if len(rows) == 0:
        raise ValueError(f"No rows found in state g CSV: {path}")

    return rows


def parse_float_or_keep(value):
    """
    Convert a CSV string to float when possible.

    Otherwise, keep the original value.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def plot_training_losses(
    metrics: List[Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plot observation prediction losses.

    These are the main supervised prediction terms:

        loss_total
        loss_x_p
        loss_x_g
        loss_x_gt
    """
    keys = [
        "loss_total",
        "loss_x_p",
        "loss_x_g",
        "loss_x_gt",
    ]

    available_keys = [
        key for key in keys
        if key in metrics[0]
    ]

    if len(available_keys) == 0:
        print("No training loss columns found. Skipping training loss plot.")
        return

    steps = [row["step"] for row in metrics]

    plt.figure(figsize=(8, 5))

    for key in available_keys:
        values = [row[key] for row in metrics]
        plt.plot(steps, values, label=key)

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("TEM training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_latent_losses(
    metrics: List[Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plot latent consistency and regularization losses.

    These terms are useful for debugging whether latent states are becoming
    unstable or saturating.
    """
    keys = [
        "loss_p",
        "loss_px",
        "loss_g",
        "loss_g_reg",
        "loss_p_reg",
        "grad_norm",
    ]

    available_keys = [
        key for key in keys
        if key in metrics[0]
    ]

    if len(available_keys) == 0:
        print("No latent loss columns found. Skipping latent loss plot.")
        return

    steps = [row["step"] for row in metrics]

    plt.figure(figsize=(8, 5))

    for key in available_keys:
        values = [row[key] for row in metrics]
        plt.plot(steps, values, label=key)

    plt.xlabel("Training step")
    plt.ylabel("Value")
    plt.title("TEM latent losses and gradient norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_state_visit_counts(
    rows: List[Dict[str, float]],
    height: int,
    width: int,
    output_path: Path,
) -> None:
    """
    Plot how often each environment state was visited during representation
    collection.

    This plot is important because a firing map is unreliable if many states
    have very low visit counts.
    """
    counts = torch.zeros(height, width)

    for row in rows:
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        count = float(row["count"])

        counts[grid_row, grid_col] = count

    plt.figure(figsize=(6, 5))
    plt.imshow(counts.numpy(), origin="upper", aspect="equal")
    plt.colorbar(label="Visit count")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("State visit counts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_spatial_units(
    rows: List[Dict[str, float]],
    height: int,
    width: int,
    output_dir: Path,
    top_k_units: int,
) -> List[Path]:
    """
    Plot the most spatially varying g-units for each module.

    A unit is ranked by variance of its mean activation over environment states.

    This is not yet a formal grid-cell score. It is a simple diagnostic:
    units with high spatial variance are easier to visually inspect.
    """
    module_to_unit_columns = find_module_unit_columns(rows[0])

    output_paths = []

    for module_id, unit_columns in module_to_unit_columns.items():
        unit_scores = []

        for column in unit_columns:
            values = torch.tensor(
                [float(row[column]) for row in rows],
                dtype=torch.float32,
            )

            score = values.var(unbiased=False).item()
            unit_scores.append((score, column))

        unit_scores = sorted(
            unit_scores,
            key=lambda item: item[0],
            reverse=True,
        )

        selected = unit_scores[:top_k_units]

        for rank, score_and_column in enumerate(selected):
            score, column = score_and_column
            unit_id = parse_unit_id(column)

            output_path = output_dir / (
                f"module_{module_id}_rank_{rank}_unit_{unit_id}_rate_map.png"
            )

            plot_single_unit_map(
                rows=rows,
                column=column,
                height=height,
                width=width,
                title=(
                    f"Module {module_id}, unit {unit_id} "
                    f"(spatial variance={score:.4f})"
                ),
                output_path=output_path,
            )

            output_paths.append(output_path)

    return output_paths


def find_module_unit_columns(
    example_row: Dict[str, float],
) -> Dict[int, List[str]]:
    """
    Find columns matching:

        module_<module_id>_g_<unit_id>

    Returns:
        {
            module_id: [column_name, ...]
        }
    """
    pattern = re.compile(r"^module_(\d+)_g_(\d+)$")

    module_to_columns = {}

    for key in example_row.keys():
        match = pattern.match(key)

        if match is None:
            continue

        module_id = int(match.group(1))

        if module_id not in module_to_columns:
            module_to_columns[module_id] = []

        module_to_columns[module_id].append(key)

    for module_id in module_to_columns:
        module_to_columns[module_id] = sorted(
            module_to_columns[module_id],
            key=parse_unit_id,
        )

    return module_to_columns


def parse_unit_id(column_name: str) -> int:
    """
    Extract unit id from a column name like:

        module_0_g_12
    """
    return int(column_name.split("_")[-1])


def plot_single_unit_map(
    rows: List[Dict[str, float]],
    column: str,
    height: int,
    width: int,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot one unit's state-wise mean activation as a 2D heatmap.
    """
    rate_map = torch.zeros(height, width)

    for row in rows:
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        value = float(row[column])

        rate_map[grid_row, grid_col] = value

    plt.figure(figsize=(6, 5))
    plt.imshow(rate_map.numpy(), origin="upper", aspect="equal")
    plt.colorbar(label="Mean g activation")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
