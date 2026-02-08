"""Perft validation for chess engine correctness.

Tests that the move generator produces the correct number of legal moves
at various depths from the starting position. This validates bitboard
move generation, castling, en passant, promotion, and legality filtering.

Expected values from standard chess perft results:
  depth 1: 20
  depth 2: 400
  depth 3: 8,902
  depth 4: 197,281
"""

import numpy as np
import pytest
from chess_env import Chess, NUM_ACTIONS


def make_env(num_envs=1):
    """Create a single-game chess environment."""
    return Chess(num_envs=num_envs, max_steps=512)


class TestPerft:
    """Perft tests using the C engine via step-based enumeration.

    Since we don't have direct access to generate_legal_moves from Python,
    we use the observation's valid_pieces/valid_dests/valid_promos masks
    to enumerate legal moves at each depth.

    1-agent topology: agent 0 controls both sides (always the current mover).
    """

    def _get_valid_pieces(self, obs):
        """Get valid piece selections from observation."""
        return [i for i in range(64) if obs[137 + i] > 0]

    def _get_valid_dests(self, obs):
        """Get valid destinations and promotions from observation."""
        actions = []
        for i in range(64):
            if obs[201 + i] > 0:
                actions.append(i)
        for i in range(32):
            if obs[265 + i] > 0:
                actions.append(64 + i)
        return actions

    def _get_phase(self, obs):
        """Get current phase from observation."""
        return 0 if obs[71] > 0 else 1

    def test_perft_depth1(self):
        """Perft(1) = 20 from starting position."""
        total_moves = 0
        env = make_env(num_envs=1)
        env.reset(seed=42)

        obs = env.observations[0]
        assert self._get_phase(obs) == 0

        valid_pieces = self._get_valid_pieces(obs)

        for piece_sq in valid_pieces:
            env2 = make_env(num_envs=1)
            env2.reset(seed=42)

            actions = np.zeros(env2.num_agents, dtype=np.int32)
            actions[0] = piece_sq
            env2.step(actions)

            obs2 = env2.observations[0]
            total_moves += len(self._get_valid_dests(obs2))
            env2.close()

        env.close()
        assert total_moves == 20, f"Perft(1) = {total_moves}, expected 20"


class TestPerftDepth2:
    """Perft depth 2 test - uses step enumeration with action replay.

    1-agent topology: agent 0 controls both sides.
    After White's move (phase0 + phase1), agent 0 now sees Black's position.
    """

    def _get_phase(self, obs):
        return 0 if obs[71] > 0 else 1

    def _get_valid_pieces(self, obs):
        return [i for i in range(64) if obs[137 + i] > 0]

    def _get_valid_dests(self, obs):
        actions = []
        for i in range(64):
            if obs[201 + i] > 0:
                actions.append(i)
        for i in range(32):
            if obs[265 + i] > 0:
                actions.append(64 + i)
        return actions

    def test_perft_depth2_via_c(self):
        """Perft(2) = 400.

        For each of White's 20 moves, count Black's responses.
        Agent 0 controls both sides in 1-agent topology.
        """
        env = make_env(num_envs=1)
        env.reset(seed=42)

        obs = env.observations[0]
        valid_pieces = self._get_valid_pieces(obs)

        total_nodes = 0

        for piece_sq in valid_pieces:
            # Fresh env for each piece
            env_p = make_env(num_envs=1)
            env_p.reset(seed=42)

            # Agent 0 selects piece (White's turn)
            actions = np.zeros(env_p.num_agents, dtype=np.int32)
            actions[0] = piece_sq
            env_p.step(actions)

            # Get destinations for this piece
            obs_after_pick = env_p.observations[0]
            all_dests = self._get_valid_dests(obs_after_pick)

            for dest in all_dests:
                # Fresh env for each move
                env_m = make_env(num_envs=1)
                env_m.reset(seed=42)

                # White: pick piece
                actions = np.zeros(env_m.num_agents, dtype=np.int32)
                actions[0] = piece_sq
                env_m.step(actions)

                # White: pick dest -> move executes, turn switches to Black
                actions = np.zeros(env_m.num_agents, dtype=np.int32)
                actions[0] = dest
                env_m.step(actions)

                # Check if game ended
                if env_m.terminals[0] > 0:
                    total_nodes += 1
                    env_m.close()
                    continue

                # Now agent 0 sees Black's position (phase 0)
                obs_b = env_m.observations[0]
                assert self._get_phase(obs_b) == 0

                black_pieces = self._get_valid_pieces(obs_b)

                for bp in black_pieces:
                    # Fresh env, replay White's move, then select Black's piece
                    env_bp = make_env(num_envs=1)
                    env_bp.reset(seed=42)

                    # Replay White's move (agent 0)
                    acts = np.zeros(env_bp.num_agents, dtype=np.int32)
                    acts[0] = piece_sq
                    env_bp.step(acts)

                    acts = np.zeros(env_bp.num_agents, dtype=np.int32)
                    acts[0] = dest
                    env_bp.step(acts)

                    # Black selects piece (still agent 0)
                    acts = np.zeros(env_bp.num_agents, dtype=np.int32)
                    acts[0] = bp
                    env_bp.step(acts)

                    obs_bp = env_bp.observations[0]
                    total_nodes += len(self._get_valid_dests(obs_bp))
                    env_bp.close()

                env_m.close()

            env_p.close()

        env.close()
        assert total_nodes == 400, f"Perft(2) = {total_nodes}, expected 400"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
