import torch

from data.environments import RectangleEnvironment


def generate_random_walk(
    environment: RectangleEnvironment,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    Generate a batch of random walks in an environment.

    Returned tensors:

        position:
            [B, T]
            Discrete state ids.

        action:
            [B, T]
            Discrete action ids.

        visited:
            [B, T]
            Memory-update mask.

    Convention:

        position[:, 0] is the randomly sampled initial state.

        action[:, 0] is a dummy action set to 0.

        For t >= 1:

            action[:, t] is sampled from valid actions at position[:, t - 1].

            position[:, t] = environment.next_state(
                position[:, t - 1],
                action[:, t],
            )

    This convention matches the recurrent TEM input format:

        x_t is the observation at the current state.
        a_t is the action associated with arriving at the current state.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    position = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    action = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    visited = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.float32,
    )

    current_position = environment.sample_start_states(
        batch_size=batch_size,
        generator=generator,
    )

    position[:, 0] = current_position

    for t in range(1, sequence_length):
        current_action = environment.sample_valid_actions(
            current_position,
            generator=generator,
        )

        next_position = environment.next_state(
            current_position,
            current_action,
        )

        action[:, t] = current_action
        position[:, t] = next_position

        current_position = next_position

    return {
        "position": position,
        "action": action,
        "visited": visited,
    }
