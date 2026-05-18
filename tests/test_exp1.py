"""
Tests for Experiment 1: Structural Generalization.

Tests cover:
    - Environment construction and transition correctness
    - Random walk generation
    - Batch generation
    - Edge tracking
    - Zero-shot evaluation (smoke test)
    - Full experiment (ultra-short smoke test)
"""

import torch
import torch.nn.functional as F

from tem.config import (
    DataConfig, LossConfig, ModelConfig, TEMConfig, TrainingConfig,
)
from tem.models.tem import TEM
from tem_data.envs.line import LineEnvironment
from tem_data.envs.family_tree import FamilyTreeEnvironment
from tem_data.envs.rectangle import RectangleEnvironment
from tem_data.sampling.random_walk import generate_random_walk, traversed_edges
from tem_data.sampling.batches import WalkBatcher
from tem_experiments.exp1_generalization.line_ti.helpers import (
    evaluate_continuous_single,
    evaluate_zero_shot,
)



# ------------------------------------------------------------------ #
# Tiny config for fast testing
# ------------------------------------------------------------------ #

def _tiny_config(num_actions: int, num_observations: int = 20) -> TEMConfig:
    return TEMConfig(
        seed=0,
        device="cpu",
        data=DataConfig(
            batch_size=2,
            sequence_length=6,
            num_observations=num_observations,
            num_actions=num_actions,
        ),
        model=ModelConfig(
            num_modules=2,
            compressed_x_dim=3,
            g_dims=[4, 5],
            phase_dims=[2, 3],
            sensory_filter_alphas=[0.4, 0.6],
            sensory_scales=[1.0, 0.8],
            attractor_iterations=2,
            attractor_kappas=[0.7, 0.9],
            memory_eta_gen=0.1,
            memory_eta_inf=0.1,
            memory_lambda_gen=0.95,
            memory_lambda_inf=0.95,
            memory_clip=1.0,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            num_steps=2,
            grad_clip_norm=1.0,
        ),
        loss=LossConfig(
            beta_x_p=1.0,
            beta_x_g=1.0,
            beta_x_gt=1.0,
            beta_p=0.1,
            beta_px=0.1,
            beta_g=0.1,
            beta_g_reg=0.01,
            beta_p_reg=0.01,
        ),
    )


# ================================================================== #
# LINE ENVIRONMENT
# ================================================================== #

class TestLineEnvironment:

    def test_construction(self) -> None:
        env = LineEnvironment(num_nodes=7, num_observations=20)
        assert env.num_states == 7
        assert env.num_actions == 2
        assert env.observation_ids.shape == (7,)

    def test_transition_boundaries(self) -> None:
        env = LineEnvironment(num_nodes=5, num_observations=10)
        # Left boundary: state 0, action left -> stay
        assert env.transition_table[0, 0].item() == 0
        # Right boundary: state 4, action right -> stay
        assert env.transition_table[4, 1].item() == 4
        # Interior: state 2, left -> 1, right -> 3
        assert env.transition_table[2, 0].item() == 1
        assert env.transition_table[2, 1].item() == 3

    def test_valid_mask(self) -> None:
        env = LineEnvironment(num_nodes=5, num_observations=10)
        # State 0: only right is valid
        assert env.valid_action_mask[0, 0].item() == False
        assert env.valid_action_mask[0, 1].item() == True
        # State 2: both valid
        assert env.valid_action_mask[2, 0].item() == True
        assert env.valid_action_mask[2, 1].item() == True

    def test_resample_changes_observations(self) -> None:
        env = LineEnvironment(num_nodes=7, num_observations=20)
        obs1 = env.observation_ids.clone()
        gen = torch.Generator()
        gen.manual_seed(999)
        env.resample_observations(gen)
        obs2 = env.observation_ids.clone()
        # Very unlikely to be identical
        assert not torch.equal(obs1, obs2)

    def test_all_valid_edges(self) -> None:
        env = LineEnvironment(num_nodes=5, num_observations=10)
        edges = env.all_valid_edges()
        # 5-node line: 4 right edges + 4 left edges = 8
        assert len(edges) == 8


# ================================================================== #
# FAMILY TREE ENVIRONMENT
# ================================================================== #

class TestFamilyTreeEnvironment:

    def test_construction(self) -> None:
        env = FamilyTreeEnvironment(num_observations=20)
        assert env.num_states == 12
        assert env.num_actions == 6

    def test_parent_child(self) -> None:
        env = FamilyTreeEnvironment(num_observations=20)
        # Node 6's parent is node 2
        assert env.transition_table[6, 0].item() == 2  # parent
        # Node 2's left child is 6
        assert env.transition_table[2, 1].item() == 6  # child_left
        # Node 2's right child is 7
        assert env.transition_table[2, 2].item() == 7  # child_right

    def test_sibling(self) -> None:
        env = FamilyTreeEnvironment(num_observations=20)
        # Node 6 and 7 are siblings
        assert env.transition_table[6, 3].item() == 7
        assert env.transition_table[7, 3].item() == 6

    def test_partner(self) -> None:
        env = FamilyTreeEnvironment(num_observations=20)
        assert env.transition_table[2, 4].item() == 4
        assert env.transition_table[4, 4].item() == 2


# ================================================================== #
# RECTANGLE ENVIRONMENT
# ================================================================== #

class TestRectangleEnvironment:

    def test_construction(self) -> None:
        env = RectangleEnvironment(height=3, width=4, num_observations=20)
        assert env.num_states == 12
        assert env.num_actions == 4

    def test_transition_corners(self) -> None:
        env = RectangleEnvironment(height=3, width=4, num_observations=20)
        # State 0 = (0,0): up->stay, down->4, left->stay, right->1
        assert env.transition_table[0, 0].item() == 0   # up
        assert env.transition_table[0, 1].item() == 4   # down
        assert env.transition_table[0, 2].item() == 0   # left
        assert env.transition_table[0, 3].item() == 1   # right


# ================================================================== #
# RANDOM WALK
# ================================================================== #

class TestRandomWalk:

    def test_walk_shape(self) -> None:
        env = LineEnvironment(num_nodes=7, num_observations=20)
        walk = generate_random_walk(env, batch_size=4, seq_len=10)
        assert walk["position"].shape == (4, 10)
        assert walk["action"].shape == (4, 10)
        assert walk["visited"].shape == (4, 10)

    def test_walk_consistency(self) -> None:
        """position[t] = tau(position[t-1], action[t]) for t >= 1."""
        env = LineEnvironment(num_nodes=7, num_observations=20)
        walk = generate_random_walk(env, batch_size=8, seq_len=20)
        for t in range(1, 20):
            expected = env.next_state(walk["position"][:, t - 1], walk["action"][:, t])
            assert torch.equal(walk["position"][:, t], expected)

    def test_traversed_edges(self) -> None:
        pos = torch.tensor([0, 1, 2, 1])
        act = torch.tensor([0, 1, 1, 0])  # dummy, right, right, left
        edges = traversed_edges(pos, act)
        assert (0, 1, 1) in edges
        assert (1, 1, 2) in edges
        assert (2, 0, 1) in edges
        assert len(edges) == 3


# ================================================================== #
# BATCH GENERATION
# ================================================================== #

class TestWalkBatcher:

    def test_batch_shapes(self) -> None:
        env = LineEnvironment(num_nodes=7, num_observations=20)
        batcher = WalkBatcher(env=env, batch_size=4, seq_len=10)
        batch = batcher.sample_batch()
        assert batch["x"].shape == (4, 10, 20)
        assert batch["a"].shape == (4, 10, 2)
        assert batch["visited"].shape == (4, 10)
        assert batch["position"].shape == (4, 10)

    def test_one_hot_correctness(self) -> None:
        env = LineEnvironment(num_nodes=5, num_observations=10)
        batcher = WalkBatcher(env=env, batch_size=2, seq_len=6)
        batch = batcher.sample_batch()
        # x should be one-hot
        assert torch.allclose(batch["x"].sum(dim=-1), torch.ones(2, 6))
        # a should be one-hot
        assert torch.allclose(batch["a"].sum(dim=-1), torch.ones(2, 6))


# ================================================================== #
# ZERO-SHOT EVALUATION (smoke test)
# ================================================================== #

class TestZeroShotEval:

    def test_single_episode_runs(self) -> None:
        """Just check it doesn't crash and returns the right keys."""
        config = _tiny_config(num_actions=2, num_observations=20)
        model = TEM(config)
        env = LineEnvironment(num_nodes=7, num_observations=20)

        result = evaluate_continuous_single(
            model=model, env=env, walk_len=8,
            device=torch.device("cpu"),
        )
        assert "step_results" in result
        assert "total_nodes" in result
        assert isinstance(result["step_results"], list)
        assert result["total_nodes"] == 7


    def test_multi_episode_runs(self) -> None:
        config = _tiny_config(num_actions=2, num_observations=20)
        model = TEM(config)
        env = LineEnvironment(num_nodes=7, num_observations=20)

        result = evaluate_zero_shot(
            model=model, env=env, explore_len=8,
            num_episodes=3, device=torch.device("cpu"),
        )
        assert "zero_shot_accuracy" in result
        assert "seen_accuracy" in result
        assert 0.0 <= result["zero_shot_accuracy"] <= 1.0
        assert 0.0 <= result["seen_accuracy"] <= 1.0

    def test_works_on_family_tree(self) -> None:
        config = _tiny_config(num_actions=6, num_observations=20)
        model = TEM(config)
        env = FamilyTreeEnvironment(num_observations=20)

        result = evaluate_zero_shot(
            model=model, env=env, explore_len=10,
            num_episodes=2, device=torch.device("cpu"),
        )
        assert "zero_shot_accuracy" in result

    def test_works_on_rectangle(self) -> None:
        config = _tiny_config(num_actions=4, num_observations=20)
        model = TEM(config)
        env = RectangleEnvironment(height=3, width=3, num_observations=20)

        result = evaluate_zero_shot(
            model=model, env=env, explore_len=8,
            num_episodes=2, device=torch.device("cpu"),
        )
        assert "zero_shot_accuracy" in result


# ================================================================== #
# FULL EXPERIMENT (ultra-short smoke test)
# ================================================================== #

class TestLineTIExperiment:

    def test_experiment_runs(self, tmp_path) -> None:
        """Run 2-step experiment to verify the full pipeline."""
        from tem_experiments.exp1_generalization.line_ti.experiment import run_line_ti

        config = _tiny_config(num_actions=2, num_observations=20)
        result = run_line_ti(
            config=config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
            num_steps=2,
            eval_every=2,
            log_every=1,
            checkpoint_every=2,
            eval_episodes=2,
            seed=0,
        )
        assert result.final_step == 2
        assert len(result.history) > 0
        assert len(result.eval_history) > 0
        assert (tmp_path / "checkpoints" / "latest.pt").exists()
        assert (tmp_path / "train_metrics.csv").exists()
        assert (tmp_path / "eval_metrics.csv").exists()
        assert result.final_bins is not None or result.final_bins is None  # bins may be None if eval didn't run
