from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a training checkpoint.

    The checkpoint contains:

        model_state_dict:
            Model parameters.

        optimizer_state_dict:
            Optimizer state, if an optimizer is provided.

        step:
            Current global training step.

        config:
            A plain dictionary version of the config, if provided.

        extra:
            Optional extra information such as final metrics.

    Notes:
        We save config as a dictionary instead of saving the dataclass object
        itself. This makes checkpoints more stable and easier to inspect.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": int(step),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if config is not None:
        checkpoint["config"] = _config_to_dict(config)

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint into a model.

    If an optimizer is provided and the checkpoint has optimizer state,
    the optimizer state is restored as well.

    Returns:
        The full checkpoint dictionary.
    """
    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a checkpoint directory.

    Priority:
        1. latest.pt, if it exists.
        2. The largest step_*.pt checkpoint.
        3. None, if no checkpoint exists.
    """
    checkpoint_dir = Path(checkpoint_dir)

    latest_path = checkpoint_dir / "latest.pt"

    if latest_path.exists():
        return latest_path

    step_paths = sorted(checkpoint_dir.glob("step_*.pt"))

    if len(step_paths) == 0:
        return None

    return step_paths[-1]


def _config_to_dict(config: Any) -> Any:
    """
    Convert a dataclass config to a plain dictionary.

    If the object is already not a dataclass, return it unchanged.
    """
    if is_dataclass(config):
        return asdict(config)

    return config
