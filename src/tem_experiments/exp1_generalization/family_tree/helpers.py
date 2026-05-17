"""
Helper functions for Family Tree experiment.

Identical structure to Line TI helpers — the zero-shot evaluation
protocol is the same: resample observations, explore, query un-traversed edges.
"""

from typing import Dict, Optional

import torch

from tem_data.base import GraphEnvironment
from tem_experiments.exp1_generalization.line_ti.helpers import (
    evaluate_zero_shot,
    evaluate_zero_shot_single,
)

# Re-export — the evaluation logic is graph-agnostic
__all__ = ["evaluate_zero_shot", "evaluate_zero_shot_single"]
