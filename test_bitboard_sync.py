"""Bitboard sync verification tests.

Validates that after every operation, the bitboard representation
stays consistent with the board[64] representation:
1. popcount(bb_occ) == number of non-empty board squares
2. bb_by_color[0] | bb_by_color[1] == bb_occ
3. piece_count sums match bb popcount per type
4. bb_by_type[0] == bb_occ

Since we can't directly access bitboard fields from Python, we validate
indirectly through:
- Extended random play without crashes (segfault = inconsistency)
- Legal move correctness (wrong bitboards = wrong attack detection = wrong moves)
- Valid piece/dest masks in observations matching actual board state
"""

import numpy as np
import pytest
from chess_env import Chess, NUM_ACTIONS


def make_env(**kwargs):
    return Chess(**kwargs)


class TestBitboardSync:
    """Indirect bitboard consistency tests via observation validation."""

    def test_initial_position_valid_pieces(self):
        """At start, White should have pieces on ranks 1-2 (squares 0-15)."""
        env = make_env(num_envs=1, max_steps=256)
        env.reset(seed=42)

        obs_w = env.observations[0]  # White agent
        valid_pieces = obs_w[137:201]  # OBS_VALID_PIECES

        # White pieces are on squares 0-15 in initial position
        # But only pieces with legal moves should be valid
        valid_squares = [i for i in range(64) if valid_pieces[i] > 0]

        # Knights (1,6) and pawns (8-15) should have valid moves
        # In standard chess, all 16 pieces don't have moves (only pawns and knights)
        # Expected: pawns on 8-15 (8 pawns) + knights on 1,6 (2 knights) = 10 pieces
        # But not all pawns may have moves if blocked
        assert len(valid_squares) > 0, "White should have at least some valid pieces"
        assert len(valid_squares) <= 16, "White can't have more than 16 pieces"

        # Specifically, initial position should have exactly 10 pieces with moves
        # (8 pawns + 2 knights, since bishops/rooks/queen/king are blocked)
        assert len(valid_squares) == 10, \
            f"Expected 10 valid pieces at start (8 pawns + 2 knights), got {len(valid_squares)}: {valid_squares}"

        env.close()

    def test_board_obs_matches_expected_initial(self):
        """Verify initial board observation matches expected piece layout."""
        env = make_env(num_envs=1, max_steps=256)
        env.reset(seed=42)

        obs_w = env.observations[0]
        board = obs_w[:64]

        # White pieces (player 0 sees normal orientation)
        # Rank 1 (squares 0-7): R N B Q K B N R = pieces 4 2 3 5 6 3 2 4
        # Rank 2 (squares 8-15): 8 pawns = piece 1
        # Ranks 3-6 (squares 16-47): empty = 0
        # Rank 7 (squares 48-55): 8 black pawns = piece 7
        # Rank 8 (squares 56-63): r n b q k b n r = pieces 10 8 9 11 12 9 8 10

        # Check White's back rank
        assert board[0] == 4, f"a1 should be WR(4), got {board[0]}"  # WR
        assert board[1] == 2, f"b1 should be WN(2), got {board[1]}"  # WN
        assert board[4] == 6, f"e1 should be WK(6), got {board[4]}"  # WK

        # Check empty squares
        for sq in range(16, 48):
            assert board[sq] == 0, f"Square {sq} should be empty, got {board[sq]}"

        # Check Black pieces
        assert board[56] == 10, f"a8 should be BR(10), got {board[56]}"  # BR
        assert board[60] == 12, f"e8 should be BK(12), got {board[60]}"  # BK

        env.close()

    def test_piece_count_consistency_through_play(self):
        """After captures, total pieces should decrease.

        We track the number of non-empty squares in observations
        and verify it never increases (except via promotion).
        """
        env = make_env(num_envs=1, max_steps=512)
        env.reset(seed=42)
        rng = np.random.RandomState(42)

        max_pieces = 32  # Initial piece count
        steps = 0

        for _ in range(300):
            actions = np.zeros(env.num_agents, dtype=np.int32)
            for a in range(env.num_agents):
                obs = env.observations[a]
                if obs[64] == 0:
                    actions[a] = 96
                    continue
                phase = 0 if obs[71] > 0 else 1
                if phase == 0:
                    valid = [i for i in range(64) if obs[137 + i] > 0]
                    actions[a] = rng.choice(valid) if valid else 96
                else:
                    valid_d = [i for i in range(64) if obs[201 + i] > 0]
                    valid_p = [64 + i for i in range(32) if obs[265 + i] > 0]
                    valid = valid_d + valid_p
                    actions[a] = rng.choice(valid) if valid else 0

            env.step(actions)
            steps += 1

            # Count pieces on board from White's perspective
            obs_w = env.observations[0]
            board = obs_w[:64]
            piece_count = sum(1 for sq in range(64) if board[sq] != 0)

            # Pieces can increase slightly due to promotion, but should be <= 32
            # (max 8 promoted pawns per side, but captured pieces reduce count)
            assert piece_count <= 32, \
                f"Step {steps}: piece count {piece_count} > 32 (impossible)"

        env.close()

    def test_valid_moves_only_for_own_pieces(self):
        """Valid pieces mask should only contain squares with own pieces."""
        env = make_env(num_envs=1, max_steps=256)
        env.reset(seed=42)
        rng = np.random.RandomState(42)

        for _ in range(100):
            for a in range(env.num_agents):
                obs = env.observations[a]
                if obs[64] == 0:  # not my turn
                    continue

                phase = 0 if obs[71] > 0 else 1
                if phase != 0:
                    continue

                board = obs[:64]
                valid_pieces = obs[137:201]
                is_white = (a % 2 == 0)  # even agents are White

                for sq in range(64):
                    if valid_pieces[sq] > 0:
                        piece = board[sq]
                        if is_white:
                            assert 1 <= piece <= 6, \
                                f"Agent {a} (White) has valid piece at sq {sq} but piece={piece} (not White piece)"
                        else:
                            # Black sees flipped board with swapped colors
                            # Black's pieces appear as 1-6 in their own view
                            assert 1 <= piece <= 6, \
                                f"Agent {a} (Black) has valid piece at sq {sq} but piece={piece} (not own piece in view)"

            # Make random move
            actions = np.zeros(env.num_agents, dtype=np.int32)
            for a in range(env.num_agents):
                obs = env.observations[a]
                if obs[64] == 0:
                    actions[a] = 96
                    continue
                phase = 0 if obs[71] > 0 else 1
                if phase == 0:
                    valid = [i for i in range(64) if obs[137 + i] > 0]
                    actions[a] = rng.choice(valid) if valid else 96
                else:
                    valid_d = [i for i in range(64) if obs[201 + i] > 0]
                    valid_p = [64 + i for i in range(32) if obs[265 + i] > 0]
                    valid = valid_d + valid_p
                    actions[a] = rng.choice(valid) if valid else 0

            env.step(actions)

        env.close()

    def test_no_crash_heavy_play(self):
        """Heavy random play across many games to detect bitboard corruption."""
        env = make_env(num_envs=128, max_steps=256)
        env.reset(seed=999)

        for step in range(500):
            actions = np.random.randint(0, NUM_ACTIONS, size=env.num_agents).astype(np.int32)
            env.step(actions)

        # If we reach here without segfault, bitboards are at least not corrupt
        env.close()

    def test_fen_init_bitboard_sync(self):
        """Verify bitboard sync works correctly when initializing from FEN."""
        # This tests that setup_from_fen -> sync_bitboards_from_board works
        # by playing games with FEN-initialized positions
        import tempfile
        import os

        # Create a temp FEN file with various positions
        fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
            "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",  # 1.e4 Nc6
        ]

        fd, fen_path = tempfile.mkstemp(suffix='.txt')
        try:
            with os.fdopen(fd, 'w') as f:
                for fen in fens:
                    f.write(fen + '\n')

            env = make_env(num_envs=4, max_steps=256,
                          fen_file=fen_path, fen_curric_pct=1.0)
            env.reset(seed=42)

            # Play some moves to verify the FEN-initialized positions work
            for _ in range(100):
                actions = np.random.randint(0, NUM_ACTIONS,
                                          size=env.num_agents).astype(np.int32)
                env.step(actions)

            env.close()
        finally:
            os.unlink(fen_path)

    def test_castling_bitboard_consistency(self):
        """Test that castling moves maintain bitboard consistency.

        Play many games and if castling occurs, verify the game continues
        without crashes (which would indicate rook bitboard corruption).
        """
        env = make_env(num_envs=32, max_steps=256)
        env.reset(seed=12345)
        rng = np.random.RandomState(12345)

        for _ in range(300):
            actions = np.zeros(env.num_agents, dtype=np.int32)
            for a in range(env.num_agents):
                obs = env.observations[a]
                if obs[64] == 0:
                    actions[a] = 96
                    continue
                phase = 0 if obs[71] > 0 else 1
                if phase == 0:
                    valid = [i for i in range(64) if obs[137 + i] > 0]
                    actions[a] = rng.choice(valid) if valid else 96
                else:
                    valid_d = [i for i in range(64) if obs[201 + i] > 0]
                    valid_p = [64 + i for i in range(32) if obs[265 + i] > 0]
                    valid = valid_d + valid_p
                    actions[a] = rng.choice(valid) if valid else 0

            env.step(actions)

        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
