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

    metrics = None
    state_rows = None

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

    #  summary figure
    if metrics is not None or state_rows is not None:
        summary_png_path = output_dir / "summary_figure.png"
        summary_pdf_path = output_dir / "summary_figure.pdf"

        plot_summary_figure(
            metrics=metrics,
            rows=state_rows,
            height=args.height,
            width=args.width,
            top_k_units=args.top_k_units,
            output_path=summary_png_path,
        )

        plot_summary_figure(
            metrics=metrics,
            rows=state_rows,
            height=args.height,
            width=args.width,
            top_k_units=args.top_k_units,
            output_path=summary_pdf_path,
        )

        generated_files.append(str(summary_png_path))
        generated_files.append(str(summary_pdf_path))

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



def plot_summary_figure(
    metrics: Optional[List[Dict[str, float]]],
    rows: Optional[List[Dict[str, float]]],
    height: int,
    width: int,
    top_k_units: int,
    output_path: Path,
) -> None:
    """
    Create one paper-style summary figure.

    The figure contains:

        A. Observation prediction losses
        B. Latent losses and gradient norm
        C. State visit counts
        D/E/F... Top spatially varying g-units per module

    This is a diagnostic figure, not a formal reproduction of the original
    Cell paper figures.
    """
    if top_k_units <= 0:
        raise ValueError("top_k_units must be positive.")

    if rows is not None:
        ranked_units = rank_spatial_units(rows)
        module_ids = sorted(ranked_units.keys())
    else:
        ranked_units = {}
        module_ids = []

    num_module_rows = max(1, len(module_ids))
    num_cols = max(6, top_k_units)
    num_rows = 1 + num_module_rows

    fig_width = max(14.0, 2.6 * num_cols)
    fig_height = 4.2 + 2.4 * num_module_rows

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        nrows=num_rows,
        ncols=num_cols,
    )

    first_cut = num_cols // 3
    second_cut = 2 * num_cols // 3

    ax_train = fig.add_subplot(grid[0, 0:first_cut])
    ax_latent = fig.add_subplot(grid[0, first_cut:second_cut])
    ax_counts = fig.add_subplot(grid[0, second_cut:num_cols])

    plot_training_losses_on_axis(ax_train, metrics)
    add_panel_label(ax_train, "A")

    plot_latent_losses_on_axis(ax_latent, metrics)
    add_panel_label(ax_latent, "B")

    if rows is None:
        draw_no_data_axis(ax_counts, "No state-g CSV found")
    else:
        counts = build_count_grid(rows, height, width)
        plot_heatmap_on_axis(
            ax=ax_counts,
            heatmap=counts,
            title="State visit counts",
            colorbar=True,
            colorbar_label="Visit count",
        )

    add_panel_label(ax_counts, "C")

    if rows is None or len(module_ids) == 0:
        ax = fig.add_subplot(grid[1, :])
        draw_no_data_axis(ax, "No g-unit rate maps available")
    else:
        panel_index = 3

        for row_index, module_id in enumerate(module_ids, start=1):
            selected_units = ranked_units[module_id][:top_k_units]

            for col_index in range(num_cols):
                ax = fig.add_subplot(grid[row_index, col_index])

                if col_index >= len(selected_units):
                    ax.axis("off")
                    continue

                score, column = selected_units[col_index]
                unit_id = parse_unit_id(column)

                rate_map = build_rate_map(
                    rows=rows,
                    column=column,
                    height=height,
                    width=width,
                )

                plot_heatmap_on_axis(
                    ax=ax,
                    heatmap=rate_map,
                    title=f"M{module_id} U{unit_id}\nvar={score:.4f}",
                    colorbar=False,
                    colorbar_label="",
                )

                if col_index == 0:
                    label = chr(ord("A") + panel_index)
                    add_panel_label(ax, label)
                    ax.set_ylabel(
                        f"Module {module_id}",
                        fontsize=10,
                        fontweight="bold",
                    )
                    panel_index += 1

    fig.suptitle(
        "TEM rectangle experiment summary",
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_training_losses_on_axis(
    ax,
    metrics: Optional[List[Dict[str, float]]],
) -> None:
    """
    Draw training losses on an existing axis.

    This is used by the summary figure.
    """
    if metrics is None or len(metrics) == 0:
        draw_no_data_axis(ax, "No metrics CSV found")
        return

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
        draw_no_data_axis(ax, "No training loss columns found")
        return

    steps = [float(row["step"]) for row in metrics]

    for key in available_keys:
        values = [float(row[key]) for row in metrics]
        ax.plot(steps, values, label=key)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Observation prediction losses")
    ax.legend(fontsize=8)


def plot_latent_losses_on_axis(
    ax,
    metrics: Optional[List[Dict[str, float]]],
) -> None:
    """
    Draw latent losses and gradient norm on an existing axis.

    This is used by the summary figure.
    """
    if metrics is None or len(metrics) == 0:
        draw_no_data_axis(ax, "No metrics CSV found")
        return

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
        draw_no_data_axis(ax, "No latent loss columns found")
        return

    steps = [float(row["step"]) for row in metrics]

    for key in available_keys:
        values = [float(row[key]) for row in metrics]
        ax.plot(steps, values, label=key)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Value")
    ax.set_title("Latent losses and gradient norm")
    ax.legend(fontsize=8)


def rank_spatial_units(
    rows: List[Dict[str, float]],
) -> Dict[int, List[Tuple[float, str]]]:
    """
    Rank g-units by spatial variance.

    For each module and each unit, we compute variance over environment states:

        score(unit) = Var_s mean_g[s, unit]

    A high score means the unit changes a lot across space.

    Important:
        This is not a formal gridness score.
        It is only a simple diagnostic for selecting visually interesting units.
    """
    module_to_unit_columns = find_module_unit_columns(rows[0])
    ranked = {}

    for module_id, unit_columns in module_to_unit_columns.items():
        scores = []

        for column in unit_columns:
            values = torch.tensor(
                [float(row[column]) for row in rows],
                dtype=torch.float32,
            )

            score = values.var(unbiased=False).item()
            scores.append((score, column))

        ranked[module_id] = sorted(
            scores,
            key=lambda item: item[0],
            reverse=True,
        )

    return ranked


def build_count_grid(
    rows: List[Dict[str, float]],
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert state visit counts into a 2D grid.

    Output:
        counts: [height, width]
    """
    counts = torch.zeros(height, width)

    for row in rows:
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        count = float(row["count"])

        counts[grid_row, grid_col] = count

    return counts


def build_rate_map(
    rows: List[Dict[str, float]],
    column: str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert one unit's state-wise mean activation into a 2D rate map.

    Output:
        rate_map: [height, width]
    """
    rate_map = torch.zeros(height, width)

    for row in rows:
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        value = float(row[column])

        rate_map[grid_row, grid_col] = value

    return rate_map


def plot_heatmap_on_axis(
    ax,
    heatmap: torch.Tensor,
    title: str,
    colorbar: bool,
    colorbar_label: str,
) -> None:
    """
    Draw a 2D heatmap on an existing axis.
    """
    image = ax.imshow(
        heatmap.numpy(),
        origin="upper",
        aspect="equal",
    )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    if colorbar:
        plt.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label=colorbar_label,
        )


def add_panel_label(
    ax,
    label: str,
) -> None:
    """
    Add a paper-style panel label such as A, B, C.
    """
    ax.text(
        -0.12,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


def draw_no_data_axis(
    ax,
    message: str,
) -> None:
    """
    Draw an empty placeholder axis when an input file is missing.
    """
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)









if __name__ == "__main__":
    main()
