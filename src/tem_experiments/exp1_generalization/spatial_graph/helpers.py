"""Helpers for Spatial Graph experiment. Re-uses generic zero-shot evaluation."""

from tem_experiments.exp1_generalization.line_ti.helpers import (
    evaluate_zero_shot,
    evaluate_zero_shot_single,
)

__all__ = ["evaluate_zero_shot", "evaluate_zero_shot_single"]
