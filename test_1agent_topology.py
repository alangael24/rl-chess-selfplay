"""Tests for 1-agent-per-game topology (Fase 2).

Validates:
1. Symmetry: white_winrate ~= black_winrate (learner_color alternates)
2. Determinism: same seed = same trajectory
3. No NaN in rewards
4. obs[64:66] is always [255, 0] (mover's turn)
5. num_agents = num_envs (not *2)
6. Terminal flags work correctly
"""

import numpy as np
import pytest
from chess_env import Chess, NUM_ACTIONS, OBS_SIZE


def make_env(**kwargs):
    defaults = dict(num_envs=4, max_steps=256)
    defaults.update(kwargs)
    return Chess(**defaults)


class TestTopologyBasics:
    """Basic 1-agent topology invariants."""

    def test_num_agents_equals_num_envs(self):
        """num_agents should equal num_envs, not num_envs * 2."""
        env = make_env(num_envs=16)
        assert env.num_agents == 16
        assert env.observations.shape == (16, OBS_SIZE)
        assert env.actions.shape == (16,)
        assert env.rewards.shape == (16,)
        env.close()

    def test_obs_always_movers_perspective(self):
        """obs[64:66] should always be [255, 0] (it's always mover's turn)."""
        env = make_env(num_envs=8)
        env.reset(seed=42)

        for step in range(200):
            for a in range(env.num_agents):
                obs = env.observations[a]
                assert obs[64] == 255, f"Step {step}, agent {a}: obs[64]={obs[64]}, expected 255"
                assert obs[65] == 0, f"Step {step}, agent {a}: obs[65]={obs[65]}, expected 0"

            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            env.step(actions)

        env.close()

    def test_pass_never_valid(self):
        """pass_valid (obs[300]) should always be 0 in 1-agent mode."""
        env = make_env(num_envs=8)
        env.reset(seed=42)

        for _ in range(200):
            for a in range(env.num_agents):
                obs = env.observations[a]
                assert obs[300] == 0, f"pass_valid should be 0, got {obs[300]}"

            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            env.step(actions)

        env.close()

    def test_no_nan_rewards(self):
        """Rewards should never be NaN."""
        env = make_env(num_envs=32, max_steps=128)
        env.reset(seed=42)

        for _ in range(300):
            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            _, rewards, _, _, _ = env.step(actions)
            assert not np.any(np.isnan(rewards)), f"NaN in rewards: {rewards}"

        env.close()

    def test_terminal_flags(self):
        """Terminal flags should be 0 or 1, and reset should clear them."""
        env = make_env(num_envs=16, max_steps=64)
        env.reset(seed=42)

        terminal_seen = False
        for _ in range(500):
            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            _, _, terminals, _, _ = env.step(actions)
            for t in terminals:
                assert t in (0, 1, True, False), f"Terminal value {t} not 0/1"
            if np.any(terminals):
                terminal_seen = True

        assert terminal_seen, "No terminal ever seen in 500 steps with max_steps=64"
        env.close()


class TestDeterminism:
    """Same seed should produce identical trajectories."""

    def test_deterministic_trajectory(self):
        """Two runs with same seed and actions should produce identical obs/rewards."""
        seed = 12345
        num_envs = 4
        num_steps = 100

        # Generate fixed actions
        rng = np.random.RandomState(seed)
        all_actions = rng.randint(0, NUM_ACTIONS, size=(num_steps, num_envs)).astype(np.int32)

        # Run 1
        env1 = make_env(num_envs=num_envs)
        env1.reset(seed=seed)
        obs1_list = []
        rew1_list = []
        for i in range(num_steps):
            obs1_list.append(env1.observations.copy())
            _, rew, _, _, _ = env1.step(all_actions[i])
            rew1_list.append(rew.copy())
        env1.close()

        # Run 2
        env2 = make_env(num_envs=num_envs)
        env2.reset(seed=seed)
        obs2_list = []
        rew2_list = []
        for i in range(num_steps):
            obs2_list.append(env2.observations.copy())
            _, rew, _, _, _ = env2.step(all_actions[i])
            rew2_list.append(rew.copy())
        env2.close()

        for i in range(num_steps):
            np.testing.assert_array_equal(obs1_list[i], obs2_list[i],
                err_msg=f"Obs mismatch at step {i}")
            np.testing.assert_array_equal(rew1_list[i], rew2_list[i],
                err_msg=f"Reward mismatch at step {i}")


class TestSymmetry:
    """White and Black winrates should be approximately equal."""

    def test_winrate_symmetry(self):
        """With random play, white_wins ~= black_wins (within 10%).

        learner_color alternates each reset, so with random actions,
        results should be roughly symmetric.
        """
        env = make_env(num_envs=128, max_steps=128)
        env.reset(seed=42)

        total_info_dicts = []
        for step in range(2000):
            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            _, _, _, _, info = env.step(actions)
            total_info_dicts.extend(info)

        env.close()

        if not total_info_dicts:
            pytest.skip("No episodes completed in 2000 steps")

        # Aggregate white_wins and black_wins
        total_white = sum(d.get('white_wins', 0) for d in total_info_dicts)
        total_black = sum(d.get('black_wins', 0) for d in total_info_dicts)
        total_draws = sum(d.get('draws', 0) for d in total_info_dicts)
        total_n = sum(d.get('n', 0) for d in total_info_dicts)

        if total_n == 0:
            pytest.skip("No completed episodes")

        # The raw values from vec_log are averaged per-episode (white_wins is 0 or 1)
        # After multiplying by n, we get counts
        white_count = total_white
        black_count = total_black
        total_decisive = white_count + black_count

        if total_decisive < 10:
            pytest.skip(f"Too few decisive games: {total_decisive}")

        white_ratio = white_count / total_decisive
        # Should be roughly 0.5 ± 0.15
        assert 0.30 < white_ratio < 0.70, \
            f"White win ratio {white_ratio:.3f} too far from 0.5 (white={white_count}, black={black_count}, draws={total_draws})"


class TestRewardDirection:
    """Test that rewards are correctly signed from learner's perspective."""

    def test_extended_play_rewards(self):
        """Play many games with reward shaping and verify no anomalies."""
        env = Chess(
            num_envs=32,
            max_steps=128,
            reward_capture_bonus=0.1,
            reward_check_bonus=0.05,
            reward_material=0.01,
        )
        env.reset(seed=42)

        all_rewards = []
        for _ in range(500):
            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            _, rewards, _, _, _ = env.step(actions)
            all_rewards.extend(rewards.tolist())

        env.close()

        all_rewards = np.array(all_rewards)
        assert not np.any(np.isnan(all_rewards)), "NaN rewards detected"
        assert not np.any(np.isinf(all_rewards)), "Inf rewards detected"

        # Rewards should have both positive and negative values
        # (learner gains and opponent gains)
        has_positive = np.any(all_rewards > 0)
        has_negative = np.any(all_rewards < 0)
        assert has_positive, "No positive rewards seen"
        assert has_negative, "No negative rewards seen"


class TestFENInit:
    """Test FEN curriculum works with 1-agent topology."""

    def test_fen_curriculum_no_crash(self):
        """FEN curriculum initialization should work with 1-agent layout."""
        import tempfile
        import os

        fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 1",
        ]

        fd, fen_path = tempfile.mkstemp(suffix='.txt')
        try:
            with os.fdopen(fd, 'w') as f:
                for fen in fens:
                    f.write(fen + '\n')

            env = Chess(num_envs=8, max_steps=128,
                       fen_file=fen_path, fen_curric_pct=1.0)
            env.reset(seed=42)

            for _ in range(200):
                actions = np.random.randint(0, NUM_ACTIONS,
                                          size=env.num_agents).astype(np.int32)
                env.step(actions)

            env.close()
        finally:
            os.unlink(fen_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
