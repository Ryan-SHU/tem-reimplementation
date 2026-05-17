import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from tqdm import tqdm

from tem.config import TEMConfig
from tem.losses import compute_tem_loss
from tem_training.checkpointing import save_checkpoint
from tem_training.logging import CSVLogger, format_metrics
from tem_training.types import BatchProvider

@dataclass
class TrainResult:
    """
    Summary returned after a training run.

    Fields:
        final_step:
            Final global training step.

        history:
            List of logged metric dictionaries.

        last_checkpoint_path:
            Path to the last checkpoint written by the trainer.
    """

    final_step: int
    history: List[Dict[str, float]]
    last_checkpoint_path: Optional[Path]





class TEMTrainer:
    """
    Minimal trainer for TEM experiments.

    This class connects:

        BatchProvider
            -> generates x, a, visited

        TEM model
            -> produces TEMSequenceOutput

        compute_tem_loss
            -> computes the full training objective

        optimizer
            -> updates model parameters

    The trainer intentionally does not know the details of the environment.
    It only requires a batcher that returns:

        batch["x"]
        batch["a"]
        batch["visited"]
    """
    def __init__(
        self,
        model: nn.Module,
        config: TEMConfig,
        batcher: BatchProvider,
        output_dir: str | Path,
        device: str | torch.device = "cuda"
    ) -> None:
        self.model = model
        self.config = config
        self.batcher = batcher
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "metrics.csv"
        self.logger = CSVLogger(self.metrics_path)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
        )


    def train(
        self,
        num_steps: Optional[int] = None,
        start_step: int = 0,
        log_every: int = 10,
        checkpoint_every: int = 500,
        show_progress: bool = True,
        save_final_checkpoint: bool = True,
    ) -> TrainResult:
        """
        Run training.

        Args:
            num_steps:
                Number of optimization steps to run in this call.
                If None, use config.training.num_steps.

            start_step:
                Global step to start from.
                This is useful when resuming from a checkpoint.

            log_every:
                Print and save metrics every log_every steps.

            checkpoint_every:
                Save a checkpoint every checkpoint_every steps.
                If <= 0, periodic checkpointing is disabled.

            show_progress:
                Whether to show a tqdm progress bar.

            save_final_checkpoint:
                Whether to save a final checkpoint at the end.

        Returns:
            TrainResult.
        """
        if num_steps is None:
            num_steps = self.config.training.num_steps

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        if log_every <= 0:
            raise ValueError("log_every must be positive.")

        history = []
        last_checkpoint_path = None

        self.model.train()

        start_time = time.time()

        step_iterator = range(start_step + 1, start_step + num_steps + 1)

        progress = tqdm(
            step_iterator,
            desc="training",
            disable=not show_progress,
        )

        for step in progress:
            batch = self.batcher.sample_batch()
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(
                x=batch["x"],
                a=batch["a"],
                visited=batch["visited"],
            )

            losses = compute_tem_loss(
                output=output,
                target_x=batch["x"],
                loss_config=self.config.loss,
            )

            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {losses.total.item()}"
                )

            losses.total.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.grad_clip_norm,
            )

            self.optimizer.step()

            metrics = losses.detached()
            metrics["step"] = step
            metrics["grad_norm"] = float(grad_norm.detach().cpu().item())
            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            metrics["elapsed_sec"] = time.time() - start_time

            progress.set_postfix(
                loss=f"{metrics['loss_total']:.4f}",
                grad=f"{metrics['grad_norm']:.4f}",
            )

            should_log = (
                step == start_step + 1
                or step % log_every == 0
                or step == start_step + num_steps
            )

            if should_log:
                self.logger.log(metrics)
                history.append(metrics)

                print(format_metrics(metrics))

            should_checkpoint = (
                checkpoint_every > 0
                and step % checkpoint_every == 0
            )

            if should_checkpoint:
                last_checkpoint_path = self._save_checkpoint(
                    step=step,
                    metrics=metrics,
                )

        final_step = start_step + num_steps

        if save_final_checkpoint:
            last_checkpoint_path = self._save_checkpoint(
                step=final_step,
                metrics=history[-1] if len(history) > 0 else {},
            )

        return TrainResult(
            final_step=final_step,
            history=history,
            last_checkpoint_path=last_checkpoint_path,
        )

    def _move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Move all tensor values in the batch to the trainer device.
        """
        moved = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value

        return moved

    def _save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save both a step-specific checkpoint and a latest checkpoint.

        The step-specific checkpoint is useful for keeping history.
        The latest checkpoint is useful for quick resume.
        """
        step_path = self.checkpoint_dir / f"step_{step:07d}.pt"
        latest_path = self.checkpoint_dir / "latest.pt"

        extra = {
            "metrics": metrics,
        }

        save_checkpoint(
            path=step_path,
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            config=self.config,
            extra=extra,
        )

        save_checkpoint(
            path=latest_path,
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            config=self.config,
            extra=extra,
        )

        return step_path