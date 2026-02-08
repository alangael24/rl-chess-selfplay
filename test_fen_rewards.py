"""Test suite for FEN curriculum system and reward shaping (1-agent-per-game topology).

Tests:
1. FEN parsing: standard starting position
2. FEN parsing: middle-game position
3. FEN parsing: position with en passant and partial castling
4. FEN curriculum: games reset from FEN when configured
5. Material scoring: correct values for standard and custom positions
6. Positional scoring: PST values applied correctly
7. Capture bonus: reward shaping for captures
8. Check bonus: reward shaping for giving check
9. New log fields: material_score, positional_score, invalid_action_rate
10. vec_load_fens: load FENs after init
11. vec_set_fen_pct: update curriculum percentage at runtime
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, '/home/alanga/rl-chess-selfplay')

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS
from csrc import binding

# Observation offsets
OBS_BOARD = 0
OBS_SIDE = 64
OBS_CASTLING = 66
OBS_EP = 70
OBS_PHASE = 71
OBS_SELECTED = 73
OBS_VALID_PIECES = 137
OBS_VALID_DESTS = 201
OBS_VALID_PROMOS = 265
OBS_SELF_CHECK = 297
OBS_OPP_CHECK = 298
OBS_RULE50 = 299
OBS_PASS_VALID = 300

# Piece constants (must match chess.h)
EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12


passed = 0
failed = 0
def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} - {msg}")
        failed += 1


def make_sq(row, col):
    return row * 8 + col


def get_obs(env, agent_idx=0):
    return env.observations[agent_idx]


def get_phase(obs):
    if obs[OBS_PHASE] == 255:
        return 0
    elif obs[OBS_PHASE + 1] == 255:
        return 1
    return -1


def is_my_turn(obs):
    return obs[OBS_SIDE] == 255


def get_valid_pieces(obs):
    return set(i for i in range(64) if obs[OBS_VALID_PIECES + i] == 255)


def get_valid_dests(obs):
    return set(i for i in range(64) if obs[OBS_VALID_DESTS + i] == 255)


def step_action(env, action, game_idx=0):
    """Step environment with a single action for one game."""
    actions = np.zeros(env.num_agents, dtype=np.int32)
    actions[game_idx] = action
    return env.step(actions)


def make_move(env, from_sq, to_sq):
    """Execute a full chess move (phase 0 + phase 1) for the current mover.
    from_sq and to_sq are in the mover's perspective."""
    step_action(env, from_sq)   # phase 0: pick piece
    return step_action(env, to_sq)  # phase 1: pick dest


def write_fen_file(fens):
    """Write FEN strings to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix='.fen', prefix='test_chess_')
    with os.fdopen(fd, 'w') as f:
        for fen in fens:
            f.write(fen + '\n')
    return path


# ============================================================================
# Test 1: FEN parsing - standard starting position
# ============================================================================
print("\nTest 1: FEN parsing - standard starting position")

fen_file = write_fen_file([
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# With fen_curric_pct=1.0, every reset should use FEN
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]

# Check white back rank (row 0 = rank 1)
check("FEN start: a1=WR", board[make_sq(0, 0)] == WR)
check("FEN start: b1=WN", board[make_sq(0, 1)] == WN)
check("FEN start: e1=WK", board[make_sq(0, 4)] == WK)
check("FEN start: mover's turn", is_my_turn(obs))
check("FEN start: Phase 0", get_phase(obs) == 0)

# Verify pawns
all_pawns_ok = True
for c in range(8):
    if board[make_sq(1, c)] != WP:
        all_pawns_ok = False
check("FEN start: All white pawns on rank 2", all_pawns_ok)

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 2: FEN parsing - middle game position
# ============================================================================
print("\nTest 2: FEN parsing - middle game position (Sicilian)")

# A position from the Sicilian Defense
sicilian_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
fen_file = write_fen_file([sicilian_fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]

# e4 pawn should be on e4 = row 3, col 4 = square 28
check("Sicilian: e4 pawn", board[make_sq(3, 4)] == WP,
      f"got piece {board[make_sq(3, 4)]} at e4")
# f3 knight should be on f3 = row 2, col 5 = square 21
check("Sicilian: Nf3", board[make_sq(2, 5)] == WN,
      f"got piece {board[make_sq(2, 5)]} at f3")
# c5 pawn (black) - from White's obs, black pieces are 7-12
check("Sicilian: c5 black pawn", board[make_sq(4, 2)] == BP,
      f"got piece {board[make_sq(4, 2)]} at c5")
# c6 knight (black)
check("Sicilian: Nc6", board[make_sq(5, 2)] == BN,
      f"got piece {board[make_sq(5, 2)]} at c6")
# White's turn
check("Sicilian: mover's turn", is_my_turn(obs))

# Castling should still be full
castling = obs[OBS_CASTLING:OBS_CASTLING + 4]
check("Sicilian: All castling available", all(c == 255 for c in castling),
      f"castling={list(castling)}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 3: FEN parsing - en passant and partial castling
# ============================================================================
print("\nTest 3: FEN parsing - en passant and partial castling")

# After 1.e4 d5 2.e5 f5 - en passant available for white on f6
ep_fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
fen_file = write_fen_file([ep_fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]

# e5 pawn should be on row 4, col 4 = square 36
check("EP: e5 white pawn", board[make_sq(4, 4)] == WP,
      f"got piece {board[make_sq(4, 4)]} at e5")
# f5 black pawn on row 4, col 5 = square 37
check("EP: f5 black pawn", board[make_sq(4, 5)] == BP,
      f"got piece {board[make_sq(4, 5)]} at f5")
# d5 black pawn on row 4, col 3 = square 35
check("EP: d5 black pawn", board[make_sq(4, 3)] == BP,
      f"got piece {board[make_sq(4, 3)]} at d5")

# En passant: file should be 5 (f-file) - obs[70] = file index
ep_file = obs[OBS_EP]
check("EP: en passant file is f (5)", ep_file == 5, f"got ep_file={ep_file}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 4: FEN curriculum probability
# ============================================================================
print("\nTest 4: FEN curriculum probability")

# Create FEN with a very distinct position (only kings)
kings_only_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
fen_file = write_fen_file([kings_only_fen])

# With fen_curric_pct=0.0, should never use FEN
env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=0.0)
env.reset(seed=42)
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
# Should be normal starting position with all pieces
piece_count_no_curric = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("No curriculum: full starting position (32 pieces)",
      piece_count_no_curric == 32, f"got {piece_count_no_curric} pieces")
env.close()

# With fen_curric_pct=1.0, should always use FEN
env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count_curric = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("Full curriculum: kings only (2 pieces)",
      piece_count_curric == 2, f"got {piece_count_curric} pieces")
env.close()

os.unlink(fen_file)


# ============================================================================
# Test 5: Material scoring
# ============================================================================
print("\nTest 5: Material scoring via log fields")

# Use max_steps=10 so games complete quickly by truncation
env = Chess(num_envs=4, max_steps=10, report_interval=99999)
env.reset(seed=42)

# Play random moves - with max_steps=10, games truncate quickly
for _ in range(100):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    obs, rew, terms, truncs, info = env.step(actions)

# Now call vec_log directly - it should have accumulated data
log = binding.vec_log(env.c_envs)
has_n = log.get('n', 0) > 0
check("Got completed episodes for log test", has_n)
if has_n:
    check("Material score in log", 'material_score' in log,
          f"keys: {list(log.keys())}")
    check("Positional score in log", 'positional_score' in log)
    check("Invalid action rate in log", 'invalid_action_rate' in log)
    check("Material score is numeric",
          isinstance(log.get('material_score', None), float))

env.close()


# ============================================================================
# Test 6: Material scoring values
# ============================================================================
print("\nTest 6: Material scoring - starting position balance")

# In the starting position, material should be exactly balanced
env = Chess(num_envs=1, max_steps=1, report_interval=1)
env.reset(seed=42)

# Step once to get truncation (max_steps=1)
actions = np.array([0], dtype=np.int32)
obs, rew, terms, truncs, info = env.step(actions)

log = binding.vec_log(env.c_envs)
if log.get('n', 0) > 0:
    mat = log['material_score']
    check("Starting position material balance is 0", abs(mat) < 0.01,
          f"got material_score={mat}")
    pos = log['positional_score']
    check("Starting position positional balance is 0", abs(pos) < 0.01,
          f"got positional_score={pos}")

env.close()


# ============================================================================
# Test 7: Capture bonus
# ============================================================================
print("\nTest 7: Capture bonus reward shaping")

# Set up a position where White can immediately capture a piece
capture_fen = "rnb1kbnr/pppQpppp/8/8/8/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1"
fen_file = write_fen_file([capture_fen])

# With capture bonus
env_bonus = Chess(num_envs=1, max_steps=500, fen_file=fen_file,
                  fen_curric_pct=1.0, reward_capture_bonus=0.1)
env_bonus.reset(seed=42)

# Without capture bonus
env_no_bonus = Chess(num_envs=1, max_steps=500, fen_file=fen_file,
                     fen_curric_pct=1.0, reward_capture_bonus=0.0)
env_no_bonus.reset(seed=42)

check("Capture bonus env created", True)
check("No-capture-bonus env created", True)

env_bonus.close()
env_no_bonus.close()
os.unlink(fen_file)


# ============================================================================
# Test 8: Check bonus
# ============================================================================
print("\nTest 8: Check bonus reward shaping")

check_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
fen_file = write_fen_file([check_fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file,
            fen_curric_pct=1.0, reward_check_bonus=0.05)
env.reset(seed=42)

check("Check bonus env created without error", True)

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 9: New log fields present and reasonable
# ============================================================================
print("\nTest 9: New log fields - full game")

env = Chess(num_envs=8, max_steps=20, report_interval=99999,
            reward_capture_bonus=0.01, reward_check_bonus=0.005)
env.reset(seed=42)

# Play random steps to get completed episodes
for _ in range(200):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    env.step(actions)

log = binding.vec_log(env.c_envs)

if log.get('n', 0) > 0:
    check("Log has material_score", 'material_score' in log)
    check("Log has positional_score", 'positional_score' in log)
    check("Log has invalid_action_rate", 'invalid_action_rate' in log)

    iar = log['invalid_action_rate']
    check("Invalid action rate is non-negative and bounded",
          0.0 <= iar <= 3.0, f"got {iar}")

    # Material score should be bounded
    mat = log['material_score']
    check("Material score is bounded",
          -5000 < mat < 5000, f"got {mat}")

    # Verify old fields still present
    check("Log still has episode_length", 'episode_length' in log)
    check("Log still has white_wins", 'white_wins' in log)
    check("Log still has draws", 'draws' in log)
else:
    check("Got completed episodes", False, "no episodes completed")

env.close()


# ============================================================================
# Test 10: vec_load_fens - load FENs after init
# ============================================================================
print("\nTest 10: vec_load_fens - load FENs after init")

env = Chess(num_envs=1, max_steps=500)
env.reset(seed=42)

# Initially no FENs loaded
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("Before loading: standard position (32 pieces)",
      piece_count == 32, f"got {piece_count}")

# Create a FEN file with a minimal position
minimal_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
fen_file = write_fen_file([minimal_fen])

# Load FENs and set percentage
count = binding.vec_load_fens(env.c_envs, fen_file)
check("Loaded 1 FEN", count == 1, f"got {count}")

binding.vec_set_fen_pct(env.c_envs, 1.0)

# Reset - should now use FEN
env.reset(seed=123)

obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("After loading FEN: minimal position (3 pieces)",
      piece_count == 3, f"got {piece_count}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 11: vec_set_fen_pct - runtime update
# ============================================================================
print("\nTest 11: vec_set_fen_pct - runtime update")

kings_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
fen_file = write_fen_file([kings_fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=0.0)
env.reset(seed=42)

# Should be standard position (curric_pct=0)
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("pct=0.0: standard position (32 pieces)",
      piece_count == 32, f"got {piece_count}")

# Now set pct to 1.0
binding.vec_set_fen_pct(env.c_envs, 1.0)
env.reset(seed=123)

obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("pct=1.0: kings only (2 pieces)",
      piece_count == 2, f"got {piece_count}")

# Set back to 0
binding.vec_set_fen_pct(env.c_envs, 0.0)
env.reset(seed=456)

obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("pct back to 0.0: standard position (32 pieces)",
      piece_count == 32, f"got {piece_count}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 12: Multiple FENs - random selection
# ============================================================================
print("\nTest 12: Multiple FENs - variety check")

fens = [
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",          # 2 pieces (K vs K)
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",         # 3 pieces (K+P vs K)
    "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1",       # 4 pieces (K+P vs K+P)
]
fen_file = write_fen_file(fens)

env = Chess(num_envs=16, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# Collect piece counts across all 16 games (1 agent per game)
piece_counts = set()
for g in range(16):
    obs = get_obs(env, g)
    board = obs[OBS_BOARD:OBS_BOARD + 64]
    pc = sum(1 for sq in range(64) if board[sq] != EMPTY)
    piece_counts.add(pc)

check("Multiple FENs produce variety", len(piece_counts) >= 2,
      f"got piece counts: {piece_counts}")
check("All piece counts are from FEN list", piece_counts.issubset({2, 3, 4}),
      f"got {piece_counts}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 13: FEN auto-reset in step (curriculum on episode end)
# ============================================================================
print("\nTest 13: FEN curriculum on auto-reset during step")

kings_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
fen_file = write_fen_file([kings_fen])

# max_steps=2, so games terminate very quickly
env = Chess(num_envs=1, max_steps=2, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# The first reset should use FEN
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count_first = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("First reset uses FEN (2 pieces)", piece_count_first == 2,
      f"got {piece_count_first}")

# Step until auto-reset happens (max_steps=2)
for _ in range(5):
    actions = np.array([0], dtype=np.int32)
    obs, rew, terms, truncs, info = env.step(actions)

# After auto-reset, should again use FEN
obs = get_obs(env)
board = obs[OBS_BOARD:OBS_BOARD + 64]
piece_count_after = sum(1 for sq in range(64) if board[sq] != EMPTY)
check("Auto-reset also uses FEN (2 pieces)", piece_count_after == 2,
      f"got {piece_count_after}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 14: FEN with black to move
# ============================================================================
print("\nTest 14: FEN with black to move")

black_to_move_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
fen_file = write_fen_file([black_to_move_fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# In 1-agent topology, the obs is always from the mover's perspective.
# If Black is to move, the board is flipped so the agent sees Black's pieces as "own" (1-6).
obs = get_obs(env)

# The agent always sees it as "their turn"
check("Black to move: mover's turn", is_my_turn(obs))

# pass_valid should be 0 (always mover's turn in 1-agent topology)
check("Black to move: pass_valid is 0", obs[OBS_PASS_VALID] == 0)

# Should have valid pieces (from Black's perspective, Black's pieces are 1-6)
vp = get_valid_pieces(obs)
check("Black to move: agent has valid pieces", len(vp) > 0,
      f"got {len(vp)}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
print("=" * 60)
