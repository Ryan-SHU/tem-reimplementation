from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass
class TEMDimensions:
    batch_size: int
    sequence_length: int
    num_observations: int
    num_actions: int

    num_modules: int
    compressed_x_dim: int

    g_dims: list[int]
    phase_dims: list[int]

    @property
    def p_dims(self) -> list[int]:
        return [
            phase_dim * self.compressed_x_dim
            for phase_dim in self.phase_dims
        ]

    @property
    def total_g_dim(self) -> int:
        return sum(self.g_dims)

    @property
    def total_p_dim(self) -> int:
        return sum(self.p_dims)

    @property
    def decoder_input_dim(self) -> int:
        return self.num_modules * self.compressed_x_dim


@dataclass
class TEMInput:
    x: Tensor
    a: Tensor
    visited: Tensor


@dataclass
class TEMState:
    g: list[Tensor]
    x_s: list[Tensor]
    memory_gen: list[Tensor]
    memory_inf: list[Tensor]


@dataclass
class TEMStepOutput:
    x_p_logits: Tensor
    x_g_logits: Tensor
    x_gt_logits: Tensor

    x_p: Tensor
    x_g: Tensor
    x_gt: Tensor

    g: list[Tensor]
    g_gen: list[Tensor]

    p: list[Tensor]
    p_x: list[Tensor]
    p_g: list[Tensor]
    p_gt: list[Tensor]

    x_s: list[Tensor]

    state: TEMState


@dataclass
class TEMSequenceOutput:
    x_p_logits: Tensor
    x_g_logits: Tensor
    x_gt_logits: Tensor

    x_p: Tensor
    x_g: Tensor
    x_gt: Tensor

    g: list[Tensor]
    g_gen: list[Tensor]

    p: list[Tensor]
    p_x: list[Tensor]
    p_g: list[Tensor]
    p_gt: list[Tensor]

    x_s: list[Tensor]

    final_state: TEMState
