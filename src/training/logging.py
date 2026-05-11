import csv 
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

class CSVLogger:
    """
    A small CSV logger for training metrics.

    This logger is intentionally simple:

        - It writes one row per logging step.
        - It creates the CSV header from the first row.
        - It opens and closes the file on every log call.

    This is slower than keeping a file handle open, but it is safer for
    research runs because partial results are preserved even if training stops.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = self._read_existing_header()

    def _read_existing_header(self) -> Optional[list[str]]:
        """
        If the CSV file already exists, read its header.

        This is useful when resuming training.
        """
        if not self.path.exists():
            return None

        if self.path.stat().st_size == 0:
            return None

        with self.path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return None

        if len(header) == 0:
            return None

        return header

    def log(self, row: Dict[str, Any]) -> None:
        """
        Append one metrics row to the CSV file.
        """
        clean_row = {
            key: _to_loggable_value(value)
            for key, value in row.items()
        }

        if self.fieldnames is None:
            self.fieldnames = list(clean_row.keys())

        file_is_new = (
            not self.path.exists()
            or self.path.stat().st_size == 0
        )

        with self.path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

            if file_is_new:
                writer.writeheader()

            writer.writerow(clean_row)


def _to_loggable_value(value: Any) -> Any:
    """
    Convert common Python / PyTorch values into CSV-friendly values.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def format_metrics(
    metrics: Dict[str, Any],
    keys: Optional[Iterable[str]] = None,
    precision: int = 4,
) -> str:
    """
    Format a metrics dictionary for readable console printing.
    """
    if keys is None:
        keys = [
            "step",
            "loss_total",
            "loss_x_p",
            "loss_x_g",
            "loss_x_gt",
            "grad_norm",
        ]

    parts = []

    for key in keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")

    return " | ".join(parts)