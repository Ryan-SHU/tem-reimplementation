from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from tem.models.activations import EPS
from tem.types import TEMSequenceOutput


@dataclass
class TEMLossBreakdown:
    """
    Container for the full TEM training loss.

    We keep every component as a separate tensor so that training logs can show
    exactly which term is dominating.

    Fields:
        total:
            Final weighted scalar loss used for backpropagation.

        x_p:
            Cross-entropy loss for x_p_logits.
            This is the sensory prediction decoded from the inferred bound
            place representation p_t.

        x_g:
            Cross-entropy loss for x_g_logits.
            This is the sensory prediction decoded from generative memory
            retrieval using the inferred grid state g_t.

        x_gt:
            Cross-entropy loss for x_gt_logits.
            This is the sensory prediction decoded from generative memory
            retrieval using the transition-only grid state g_gen_t.

        p:
            MSE consistency loss between p_t and p_g_t.
            This encourages generative memory retrieval from grid state to
            recover the inferred bound place representation.

        px:
            MSE consistency loss between p_t and p_x_t.
            This encourages inference memory retrieval from sensory cue to
            recover the inferred bound place representation.

        g:
            MSE consistency loss between g_t and g_gen_t.
            This keeps inferred grid states close to transition-only path
            integration when memory evidence does not strongly disagree.

        g_reg:
            L2 regularization on inferred grid states.

        p_reg:
            L2 regularization on inferred bound place states.
    """

    total: torch.Tensor
    x_p: torch.Tensor
    x_g: torch.Tensor
    x_gt: torch.Tensor
    p: torch.Tensor
    px: torch.Tensor
    g: torch.Tensor
    g_reg: torch.Tensor
    p_reg: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """
        Return all loss components as a dictionary.

        This is useful for training logs, TensorBoard, WandB, or simple
        console printing.
        """
        return {
            "loss_total": self.total,
            "loss_x_p": self.x_p,
            "loss_x_g": self.x_g,
            "loss_x_gt": self.x_gt,
            "loss_p": self.p,
            "loss_px": self.px,
            "loss_g": self.g,
            "loss_g_reg": self.g_reg,
            "loss_p_reg": self.p_reg,
        }

    def detached(self) -> dict[str, float]:
        """
        Return Python floats detached from the graph.

        Use this only for logging. Do not use this result for backpropagation.
        """
        return {
            name: value.detach().cpu().item()
            for name, value in self.as_dict().items()
        }


def compute_tem_loss(
    output: TEMSequenceOutput,
    target_x: torch.Tensor,
    loss_config: Any,
    mask: torch.Tensor | None = None,
) -> TEMLossBreakdown:
    """
    Compute the full TEM loss.

    Args:
        output:
            Output object returned by TEM.forward(...).

        target_x:
            Ground-truth sensory target.

            Accepted shapes:

                [B, T]
                    Integer class indices.

                [B, T, N_x]
                    One-hot or probability-like target distribution.
                    In this implementation, the target class is obtained by
                    argmax over the last dimension.

            In normal training, this should usually be the same x tensor that
            was passed into the model:

                output = model(x, a, visited)
                loss = compute_tem_loss(output, x, config.loss)

        loss_config:
            Either config.loss or the full config object.

            If a full TEMConfig is passed, this function automatically uses
            loss_config.loss.

            Required fields:

                beta_x_p
                beta_x_g
                beta_x_gt
                beta_p
                beta_px
                beta_g
                beta_g_reg
                beta_p_reg

        mask:
            Optional training mask with shape [B, T].

            mask[b, t] = 1 means include that time step in the loss.
            mask[b, t] = 0 means ignore that time step in the loss.

            This is mainly for padded variable-length sequences.

            Important:
                This mask is not the same as visited.

                visited controls whether memory is updated.
                mask controls whether a time step contributes to the loss.

    Returns:
        TEMLossBreakdown:
            The weighted total loss and all unweighted component losses.
    """
    cfg = _unwrap_loss_config(loss_config)

    _check_prediction_shapes(output)

    batch_size, sequence_length, num_observations = output.x_p_logits.shape

    target_indices = _target_to_class_indices(
        target_x=target_x,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_observations=num_observations,
        device=output.x_p_logits.device,
    )

    loss_mask = _prepare_mask(
        mask=mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=output.x_p_logits.device,
        dtype=output.x_p_logits.dtype,
    )

    # ------------------------------------------------------------
    # 1. Observation prediction losses
    # ------------------------------------------------------------
    #
    # Each of these compares logits [B, T, N_x] against class labels [B, T].
    #
    # x_p:
    #     decoded from inferred bound place representation p_t.
    #
    # x_g:
    #     decoded from generative retrieval p_g_t.
    #
    # x_gt:
    #     decoded from transition-only generative retrieval p_gt_t.
    loss_x_p = _masked_cross_entropy(
        logits=output.x_p_logits,
        target_indices=target_indices,
        mask=loss_mask,
    )

    loss_x_g = _masked_cross_entropy(
        logits=output.x_g_logits,
        target_indices=target_indices,
        mask=loss_mask,
    )

    loss_x_gt = _masked_cross_entropy(
        logits=output.x_gt_logits,
        target_indices=target_indices,
        mask=loss_mask,
    )

    # ------------------------------------------------------------
    # 2. Latent consistency losses
    # ------------------------------------------------------------
    #
    # p consistency:
    #     p_t should be close to p_g_t.
    #
    # px consistency:
    #     p_t should be close to p_x_t.
    #
    # g consistency:
    #     g_t should be close to g_gen_t.
    loss_p = _module_list_mse(
        lhs=output.p,
        rhs=output.p_g,
        mask=loss_mask,
        name="p_vs_p_g",
    )

    loss_px = _module_list_mse(
        lhs=output.p,
        rhs=output.p_x,
        mask=loss_mask,
        name="p_vs_p_x",
    )

    loss_g = _module_list_mse(
        lhs=output.g,
        rhs=output.g_gen,
        mask=loss_mask,
        name="g_vs_g_gen",
    )

    # ------------------------------------------------------------
    # 3. State regularization losses
    # ------------------------------------------------------------
    #
    # These are intentionally simple L2 penalties.
    # They help prevent latent states from saturating too aggressively.
    loss_g_reg = _module_list_l2(
        values=output.g,
        mask=loss_mask,
        name="g_reg",
    )

    loss_p_reg = _module_list_l2(
        values=output.p,
        mask=loss_mask,
        name="p_reg",
    )

    # ------------------------------------------------------------
    # 4. Weighted total objective
    # ------------------------------------------------------------
    #
    # The component losses above are unweighted. Weighting happens only here.
    total = (
        float(cfg.beta_x_p) * loss_x_p
        + float(cfg.beta_x_g) * loss_x_g
        + float(cfg.beta_x_gt) * loss_x_gt
        + float(cfg.beta_p) * loss_p
        + float(cfg.beta_px) * loss_px
        + float(cfg.beta_g) * loss_g
        + float(cfg.beta_g_reg) * loss_g_reg
        + float(cfg.beta_p_reg) * loss_p_reg
    )

    return TEMLossBreakdown(
        total=total,
        x_p=loss_x_p,
        x_g=loss_x_g,
        x_gt=loss_x_gt,
        p=loss_p,
        px=loss_px,
        g=loss_g,
        g_reg=loss_g_reg,
        p_reg=loss_p_reg,
    )


def _unwrap_loss_config(loss_config: Any) -> Any:
    """
    Accept either config.loss or the full config object.

    This lets both of the following work:

        compute_tem_loss(output, x, config.loss)
        compute_tem_loss(output, x, config)
    """
    if hasattr(loss_config, "loss"):
        return loss_config.loss

    return loss_config


def _check_prediction_shapes(output: TEMSequenceOutput) -> None:
    """
    Check the three observation-logit tensors before computing loss.

    Expected shapes:

        x_p_logits:  [B, T, N_x]
        x_g_logits:  [B, T, N_x]
        x_gt_logits: [B, T, N_x]
    """
    if output.x_p_logits.dim() != 3:
        raise ValueError(
            "output.x_p_logits must have shape [B, T, N_x], "
            f"but got {tuple(output.x_p_logits.shape)}."
        )

    expected_shape = tuple(output.x_p_logits.shape)

    if tuple(output.x_g_logits.shape) != expected_shape:
        raise ValueError(
            "output.x_g_logits must have the same shape as output.x_p_logits. "
            f"Expected {expected_shape}, got {tuple(output.x_g_logits.shape)}."
        )

    if tuple(output.x_gt_logits.shape) != expected_shape:
        raise ValueError(
            "output.x_gt_logits must have the same shape as output.x_p_logits. "
            f"Expected {expected_shape}, got {tuple(output.x_gt_logits.shape)}."
        )


def _target_to_class_indices(
    target_x: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    num_observations: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert target observations into class indices.

    Accepted inputs:

        target_x: [B, T]
            Already class indices.

        target_x: [B, T, N_x]
            One-hot or probability-like sensory target.
            We use argmax over the last dimension.

    Output:

        target_indices: [B, T]
    """
    if target_x.dim() == 2:
        if tuple(target_x.shape) != (batch_size, sequence_length):
            raise ValueError(
                "target_x with shape [B, T] must match model output. "
                f"Expected {(batch_size, sequence_length)}, "
                f"got {tuple(target_x.shape)}."
            )

        return target_x.to(device=device, dtype=torch.long)

    if target_x.dim() == 3:
        expected_shape = (batch_size, sequence_length, num_observations)

        if tuple(target_x.shape) != expected_shape:
            raise ValueError(
                "target_x with shape [B, T, N_x] must match model output. "
                f"Expected {expected_shape}, got {tuple(target_x.shape)}."
            )

        return target_x.to(device=device).argmax(dim=-1).long()

    raise ValueError(
        "target_x must have shape [B, T] or [B, T, N_x], "
        f"but got {tuple(target_x.shape)}."
    )


def _prepare_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepare a [B, T] loss mask.

    If no mask is provided, every time step contributes to the loss.
    """
    if mask is None:
        return torch.ones(
            batch_size,
            sequence_length,
            device=device,
            dtype=dtype,
        )

    if tuple(mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            "mask must have shape [B, T]. "
            f"Expected {(batch_size, sequence_length)}, got {tuple(mask.shape)}."
        )

    return mask.to(device=device, dtype=dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the masked mean of a [B, T] tensor.

    If mask is all zero, this returns zero instead of NaN.
    """
    if values.shape != mask.shape:
        raise ValueError(
            "values and mask must have the same shape. "
            f"values shape: {tuple(values.shape)}, mask shape: {tuple(mask.shape)}."
        )

    numerator = (values * mask).sum()
    denominator = mask.sum().clamp_min(EPS)

    return numerator / denominator


def _masked_cross_entropy(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross entropy over [B, T, N_x] logits with a [B, T] mask.

    PyTorch cross_entropy expects:

        logits: [N, C]
        target: [N]

    So we flatten batch and time into one dimension, compute per-token loss,
    then reshape back to [B, T] before applying the mask.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must have shape [B, T, N_x], but got {tuple(logits.shape)}."
        )

    batch_size, sequence_length, num_observations = logits.shape

    if tuple(target_indices.shape) != (batch_size, sequence_length):
        raise ValueError(
            "target_indices must have shape [B, T]. "
            f"Expected {(batch_size, sequence_length)}, "
            f"got {tuple(target_indices.shape)}."
        )

    if tuple(mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            "mask must have shape [B, T]. "
            f"Expected {(batch_size, sequence_length)}, got {tuple(mask.shape)}."
        )

    flat_logits = logits.reshape(batch_size * sequence_length, num_observations)
    flat_targets = target_indices.reshape(batch_size * sequence_length)

    flat_loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
    )

    loss_per_step = flat_loss.reshape(batch_size, sequence_length)

    return _masked_mean(loss_per_step, mask)


def _module_list_mse(
    lhs: list[torch.Tensor],
    rhs: list[torch.Tensor],
    mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Compute average masked MSE over a list of module tensors.

    Each module tensor must have shape:

        [B, T, D_f]

    where D_f can differ across modules.

    The loss is computed as:

        1. MSE over feature dimension for each module:
               [B, T, D_f] -> [B, T]

        2. masked mean over batch/time:
               [B, T] -> scalar

        3. average over modules:
               list[scalar] -> scalar

    This gives each module equal weight, independent of its dimensionality.
    """
    if len(lhs) != len(rhs):
        raise ValueError(
            f"{name}: lhs and rhs must have the same number of modules. "
            f"Got {len(lhs)} and {len(rhs)}."
        )

    if len(lhs) == 0:
        raise ValueError(f"{name}: module lists must not be empty.")

    module_losses = []

    for f, (lhs_f, rhs_f) in enumerate(zip(lhs, rhs)):
        if tuple(lhs_f.shape) != tuple(rhs_f.shape):
            raise ValueError(
                f"{name}: module {f} tensors must have the same shape. "
                f"lhs shape: {tuple(lhs_f.shape)}, rhs shape: {tuple(rhs_f.shape)}."
            )

        if lhs_f.dim() != 3:
            raise ValueError(
                f"{name}: module {f} tensor must have shape [B, T, D], "
                f"but got {tuple(lhs_f.shape)}."
            )

        if tuple(lhs_f.shape[:2]) != tuple(mask.shape):
            raise ValueError(
                f"{name}: module {f} batch/time dimensions must match mask. "
                f"tensor shape: {tuple(lhs_f.shape)}, mask shape: {tuple(mask.shape)}."
            )

        mse_per_step = (lhs_f - rhs_f).pow(2).mean(dim=-1)
        module_losses.append(_masked_mean(mse_per_step, mask))

    return torch.stack(module_losses).mean()


def _module_list_l2(
    values: list[torch.Tensor],
    mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Compute average masked L2 penalty over a list of module tensors.

    Each module tensor must have shape:

        [B, T, D_f]

    The loss is:

        mean over features
        masked mean over batch/time
        average over modules
    """
    if len(values) == 0:
        raise ValueError(f"{name}: module list must not be empty.")

    module_losses = []

    for f, value_f in enumerate(values):
        if value_f.dim() != 3:
            raise ValueError(
                f"{name}: module {f} tensor must have shape [B, T, D], "
                f"but got {tuple(value_f.shape)}."
            )

        if tuple(value_f.shape[:2]) != tuple(mask.shape):
            raise ValueError(
                f"{name}: module {f} batch/time dimensions must match mask. "
                f"tensor shape: {tuple(value_f.shape)}, mask shape: {tuple(mask.shape)}."
            )

        l2_per_step = value_f.pow(2).mean(dim=-1)
        module_losses.append(_masked_mean(l2_per_step, mask))

    return torch.stack(module_losses).mean()
