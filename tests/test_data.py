import torch

from data.batches import RandomWalkBatcher
from data.environments import RectangleEnvironment
from data.walks import generate_random_walk


def test_rectangle_environment_basic_properties() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    assert environment.num_states == 12
    assert environment.num_actions == 4
    assert environment.transition_table.shape == (12, 4)
    assert environment.valid_action_mask.shape == (12, 4)
    assert environment.observation_ids.shape == (12,)


def test_rectangle_environment_corner_transitions() -> None:
    """
    In a 3x4 grid, state 0 is the top-left corner.

    Actions:
        0 = up    -> wall, stay at 0
        1 = down  -> state 4
        2 = left  -> wall, stay at 0
        3 = right -> state 1
    """
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    state = torch.tensor([0, 0, 0, 0])
    action = torch.tensor([0, 1, 2, 3])

    next_state = environment.next_state(state, action)

    expected = torch.tensor([0, 4, 0, 1])

    assert torch.equal(next_state, expected)


def test_rectangle_environment_valid_actions_for_corner() -> None:
    """
    Top-left corner can only move down or right.
    """
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    valid_actions = environment.valid_actions_for_state(0)

    assert valid_actions == [1, 3]


def test_rectangle_environment_state_coord_conversion() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    state = torch.tensor([0, 1, 4, 11])

    row, col = environment.state_to_coord(state)
    recovered_state = environment.coord_to_state(row, col)

    assert torch.equal(recovered_state, state)


def test_rectangle_environment_observation_ids_are_valid() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    state = torch.arange(environment.num_states)
    observation_id = environment.get_observation_ids(state)

    assert observation_id.shape == (environment.num_states,)
    assert torch.all(observation_id >= 0)
    assert torch.all(observation_id < environment.num_observations)


def test_generate_random_walk_shapes() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    batch_size = 5
    sequence_length = 7

    walk = generate_random_walk(
        environment=environment,
        batch_size=batch_size,
        sequence_length=sequence_length,
        generator=generator,
    )

    assert walk["position"].shape == (batch_size, sequence_length)
    assert walk["action"].shape == (batch_size, sequence_length)
    assert walk["visited"].shape == (batch_size, sequence_length)


def test_generate_random_walk_values_are_valid() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    walk = generate_random_walk(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        generator=generator,
    )

    position = walk["position"]
    action = walk["action"]
    visited = walk["visited"]

    assert torch.all(position >= 0)
    assert torch.all(position < environment.num_states)

    assert torch.all(action >= 0)
    assert torch.all(action < environment.num_actions)

    assert torch.allclose(visited, torch.ones_like(visited))


def test_generate_random_walk_transitions_are_consistent() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    walk = generate_random_walk(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        generator=generator,
    )

    position = walk["position"]
    action = walk["action"]

    for t in range(1, position.shape[1]):
        expected_next = environment.next_state(
            position[:, t - 1],
            action[:, t],
        )

        assert torch.equal(position[:, t], expected_next)


def test_random_walk_batcher_shapes() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        device="cpu",
        seed=0,
    )

    batch = batcher.sample()

    assert batch["x"].shape == (5, 7, environment.num_observations)
    assert batch["a"].shape == (5, 7, environment.num_actions)
    assert batch["visited"].shape == (5, 7)
    assert batch["position"].shape == (5, 7)
    assert batch["observation_id"].shape == (5, 7)
    assert batch["action_id"].shape == (5, 7)


def test_random_walk_batcher_one_hot_values() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        device="cpu",
        seed=0,
    )

    batch = batcher.sample()

    x = batch["x"]
    a = batch["a"]

    assert torch.allclose(
        x.sum(dim=-1),
        torch.ones(5, 7),
    )

    assert torch.allclose(
        a.sum(dim=-1),
        torch.ones(5, 7),
    )


def test_random_walk_batcher_observation_matches_environment() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    batcher = RandomWalkBatcher(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        device="cpu",
        seed=0,
    )

    batch = batcher.sample()

    expected_observation_id = environment.get_observation_ids(
        batch["position"].cpu()
    )

    assert torch.equal(
        batch["observation_id"].cpu(),
        expected_observation_id,
    )


def test_random_walk_batcher_reproducibility() -> None:
    environment = RectangleEnvironment(
        height=3,
        width=4,
        num_observations=20,
        seed=0,
    )

    batcher_1 = RandomWalkBatcher(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        device="cpu",
        seed=123,
    )

    batcher_2 = RandomWalkBatcher(
        environment=environment,
        batch_size=5,
        sequence_length=7,
        device="cpu",
        seed=123,
    )

    batch_1 = batcher_1.sample()
    batch_2 = batcher_2.sample()

    assert torch.equal(batch_1["position"], batch_2["position"])
    assert torch.equal(batch_1["action_id"], batch_2["action_id"])
    assert torch.equal(batch_1["observation_id"], batch_2["observation_id"])
    assert torch.allclose(batch_1["x"], batch_2["x"])
    assert torch.allclose(batch_1["a"], batch_2["a"])
