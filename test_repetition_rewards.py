"""Test suite for threefold repetition detection, new reward shaping, and richer logs.

Tests:
1. Threefold repetition detection (make repeated knight moves)
2. GAME_REPETITION ends the game as a draw
3. Repetition penalty applied when entering repeated positions
4. Material delta reward shaping
5. Positional delta reward shaping
6. Castling reward bonus
7. New log fields present and reasonable (chess_moves, repetitions, derived fields)
8. Position history resets on c_reset
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

PASS_ACTION = 96

# Piece constants
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


def get_obs(env, agent_idx):
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


def step_with_actions(env, white_action, black_action):
    actions = np.zeros(env.num_agents, dtype=np.int32)
    actions[0] = white_action
    actions[1] = black_action
    return env.step(actions)


def write_fen_file(fens):
    fd, path = tempfile.mkstemp(suffix='.fen', prefix='test_chess_')
    with os.fdopen(fd, 'w') as f:
        for fen in fens:
            f.write(fen + '\n')
    return path


def make_move(env, player, from_sq, to_sq):
    """Make a two-phase move: pick piece then pick destination.
    from_sq and to_sq are in player's perspective (already flipped for black).
    Returns (obs, rew, terms, truncs, info) from the second step.
    """
    if player == 0:
        # White's turn: white picks piece, black passes
        step_with_actions(env, from_sq, PASS_ACTION)
        return step_with_actions(env, to_sq, PASS_ACTION)
    else:
        # Black's turn: white passes, black picks piece
        step_with_actions(env, PASS_ACTION, from_sq)
        return step_with_actions(env, PASS_ACTION, to_sq)


# ============================================================================
# Test 1: Threefold repetition detection
# ============================================================================
print("\nTest 1: Threefold repetition detection")

# Use a position where we can shuffle knights back and forth.
# Starting position: White Ng1, Nf3, Ng1, Nf3, Ng1 = 3 occurrences of initial
# We'll use a FEN to have a simpler starting state.
# Actually, let's use the standard position and play Ng1-f3-g1-f3-g1 for white
# and Ng8-f6-g8-f6-g8 for black. The position after each pair of back-and-forth
# returns to the same state.

# Simpler approach: use FEN with just kings and knights, play knight shuffle
fen = "4k3/8/8/8/8/8/8/4K2N w - - 0 1"
fen_file = write_fen_file([fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# Position: White king on e1 (sq 4), White knight on h1 (sq 7)
# Black king on e8 (sq 60)
white_obs = get_obs(env, 0)
board = white_obs[OBS_BOARD:OBS_BOARD + 64]
check("Setup: WK on e1", board[make_sq(0, 4)] == WK)
check("Setup: WN on h1", board[make_sq(0, 7)] == WN)

# Move White knight: h1(7) -> f2(13) [from white's perspective: sq 7 to sq 13]
# Then Black king: e8 -> d8 (from black's perspective: flipped. Black sees e1=sq4 as their king)
# Actually let's think carefully about the two-phase system and perspective.

# White perspective: board is as-is (row 0 = rank 1)
# Wh1 = sq 7 (row 0, col 7). In white's obs, it's sq 7.
# Valid dest for knight from h1: f2=sq(1,5)=13, g3=sq(2,6)=22
# Let's pick f2 = sq 13

# For this test, we need to be smarter. Let's use a position where both sides
# can shuffle knights and return to the same position.

env.close()
os.unlink(fen_file)

# Better approach: Use a position with knights that can go back and forth
# Position: Wk e1, Wn g1; Bk e8, Bn g8
fen = "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1"
fen_file = write_fen_file([fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

white_obs = get_obs(env, 0)
board = white_obs[OBS_BOARD:OBS_BOARD + 64]
check("Setup: WK e1", board[make_sq(0, 4)] == WK)
check("Setup: WN g1", board[make_sq(0, 6)] == WN)

# White's perspective squares:
# g1 = sq(0,6) = 6
# f3 = sq(2,5) = 21
# h3 = sq(2,7) = 23

# Black's perspective (flipped):
# Bg8 in abs coords = sq(7,6) = 62. In black's perspective = flip(62) = sq(0,6) = 6
# Bf6 in abs coords = sq(5,5) = 45. In black's perspective = flip(45) = sq(2,5) = 21
# Bh6 in abs coords = sq(5,7) = 47. In black's perspective = flip(47) = sq(2,7) = 23

# Cycle:
# 1. White: Ng1->f3 (6->21), Black: Ng8->f6 (6->21 from black's perspective)
# 2. White: Nf3->g1 (21->6), Black: Nf6->g8 (21->6 from black's perspective)
# After cycle 1+2, we're back to the initial position. Do this twice = 3 occurrences.

game_ended = False
# Cycle 1: move knights out
make_move(env, 0, 6, 21)   # White: Ng1 -> f3
check("After W Ng1-f3: not terminal", env.terminals[0] == 0)

make_move(env, 1, 6, 21)   # Black: Ng8 -> f6
check("After B Ng8-f6: not terminal", env.terminals[0] == 0)

# Cycle 2: move knights back
make_move(env, 0, 21, 6)   # White: Nf3 -> g1
check("After W Nf3-g1: not terminal", env.terminals[0] == 0)

make_move(env, 1, 21, 6)   # Black: Nf6 -> g8
# Now we're back to starting position for the 2nd time (initial = 1st, this = 2nd)
check("After B Nf6-g8 (2nd occurrence): not terminal", env.terminals[0] == 0)

# Cycle 3: move knights out again
make_move(env, 0, 6, 21)   # White: Ng1 -> f3
check("After W Ng1-f3 (cycle 3): not terminal", env.terminals[0] == 0)

make_move(env, 1, 6, 21)   # Black: Ng8 -> f6
check("After B Ng8-f6 (cycle 3): not terminal", env.terminals[0] == 0)

# Cycle 4: move knights back = 3rd occurrence of starting position
make_move(env, 0, 21, 6)   # White: Nf3 -> g1
check("After W Nf3-g1 (cycle 4): not terminal yet", env.terminals[0] == 0)

make_move(env, 1, 21, 6)   # Black: Nf6 -> g8 -> 3rd occurrence!
# Now after black moves, the position is the same as initial for the 3rd time
check("After 3rd repetition: game ended (terminal)", env.terminals[0] == 1,
      f"terminals={env.terminals[0]}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 2: GAME_REPETITION ends as draw
# ============================================================================
print("\nTest 2: GAME_REPETITION counted as draw in log")

fen = "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1"
fen_file = write_fen_file([fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
            report_interval=99999)
env.reset(seed=42)

# Play the same shuffle pattern to trigger repetition
for _ in range(2):
    make_move(env, 0, 6, 21)   # White: Ng1 -> f3
    make_move(env, 1, 6, 21)   # Black: Ng8 -> f6
    make_move(env, 0, 21, 6)   # White: Nf3 -> g1
    make_move(env, 1, 21, 6)   # Black: Nf6 -> g8

check("Game terminated by repetition", env.terminals[0] == 1)

# Check the log
log = binding.vec_log(env.c_envs)
has_n = log.get('n', 0) > 0
check("Log has completed episodes", has_n)
if has_n:
    check("Repetition counted as draw", log.get('draws', 0) > 0,
          f"draws={log.get('draws', 0)}")
    check("Repetitions field > 0", log.get('repetitions', 0) > 0,
          f"repetitions={log.get('repetitions', 0)}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 3: Repetition penalty applied
# ============================================================================
print("\nTest 3: Repetition penalty reward")

fen = "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1"
fen_file = write_fen_file([fen])

# With repetition penalty
env_pen = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                reward_repetition=-0.05)
env_pen.reset(seed=42)

# Without repetition penalty
env_nopen = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                  reward_repetition=0.0)
env_nopen.reset(seed=42)

# Do one full cycle (move out and back)
for env in [env_pen, env_nopen]:
    make_move(env, 0, 6, 21)   # White: Ng1 -> f3
    make_move(env, 1, 6, 21)   # Black: Ng8 -> f6
    make_move(env, 0, 21, 6)   # White: Nf3 -> g1
    make_move(env, 1, 21, 6)   # Black: Nf6 -> g8

# Now we're at 2nd occurrence. Next cycle should trigger penalty on the 2nd occurrence
# of the "moved out" position, and again when returning to initial (3rd of initial).

# Track rewards during a move that creates a repeated position
# First move of cycle 2:
make_move(env_pen, 0, 6, 21)   # White: Ng1 -> f3 (2nd time at this position)
rew_pen_white = env_pen.rewards[0]

make_move(env_nopen, 0, 6, 21)
rew_nopen_white = env_nopen.rewards[0]

check("Repetition penalty applied (penalized < unpenalized)",
      rew_pen_white < rew_nopen_white,
      f"penalized={rew_pen_white}, unpenalized={rew_nopen_white}")

env_pen.close()
env_nopen.close()
os.unlink(fen_file)


# ============================================================================
# Test 4: Material delta reward
# ============================================================================
print("\nTest 4: Material delta reward shaping")

# Position where White can capture a black pawn immediately
# White pawn on d4, Black pawn on e5 - White can take exd5
# Actually, let's use a simpler FEN where there's a clear capture available
# WN on d5 can capture Bp on f6
capture_fen = "4k3/8/5p2/3N4/8/8/8/4K3 w - - 0 1"
fen_file = write_fen_file([capture_fen])

# With material reward
env_mat = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                reward_material=0.01)
env_mat.reset(seed=42)

# Without material reward
env_nomat = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                  reward_material=0.0)
env_nomat.reset(seed=42)

# White knight d5 = sq(4,3) = 35. In white's perspective: sq 35
# Black pawn f6 = sq(5,5) = 45. In white's perspective: sq 45
# Knight on d5 can go to f6

make_move(env_mat, 0, 35, 45)    # Nd5 x f6 (capture pawn)
rew_mat = env_mat.rewards[0]

make_move(env_nomat, 0, 35, 45)  # Same capture
rew_nomat = env_nomat.rewards[0]

# Capturing a pawn (100 centipawns) with reward_material=0.01 should add 0.01 * 100/100 = 0.01
check("Material delta reward: capture gives higher reward",
      rew_mat > rew_nomat,
      f"with_mat={rew_mat}, without_mat={rew_nomat}")

env_mat.close()
env_nomat.close()
os.unlink(fen_file)


# ============================================================================
# Test 5: Positional delta reward
# ============================================================================
print("\nTest 5: Positional delta reward shaping")

# Move a knight to a better position (center vs edge)
# Knight on a1 (bad position) moving to c2 (slightly better) or b3 (better)
# PST for knight: a1=-50, b3=-30+5=edge values... let's just verify direction

fen = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1"
fen_file = write_fen_file([fen])

env_pos = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                reward_position=0.01)
env_pos.reset(seed=42)

env_nopos = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                  reward_position=0.0)
env_nopos.reset(seed=42)

# Na1 = sq(0,0) = 0, can go to b3 = sq(2,1) = 17
# PST: a1 has -50, b3 = knight_pst[17] = row 2, col 1 => -30+5 = index 17 => check
# Row 2 of KNIGHT_PST = -30, 5, 10, 15, 15, 10, 5, -30 => b3 (col 1) = 5
# So delta = 5 - (-50) = 55 centipawns. With scale 0.01: 0.01 * 55/100 = 0.0055

make_move(env_pos, 0, 0, 17)    # Na1 -> b3
rew_pos = env_pos.rewards[0]

make_move(env_nopos, 0, 0, 17)  # Same move
rew_nopos = env_nopos.rewards[0]

check("Positional reward: better position gives bonus",
      rew_pos > rew_nopos,
      f"with_pos={rew_pos}, without_pos={rew_nopos}")

env_pos.close()
env_nopos.close()
os.unlink(fen_file)


# ============================================================================
# Test 6: Castling reward bonus
# ============================================================================
print("\nTest 6: Castling reward bonus")

# Position where White can castle kingside: King on e1, Rook on h1, no pieces between
castle_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
fen_file = write_fen_file([castle_fen])

env_castle = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                   reward_castling=0.1)
env_castle.reset(seed=42)

env_nocastle = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                     reward_castling=0.0)
env_nocastle.reset(seed=42)

# Castling kingside: King e1 (sq 4) -> g1 (sq 6)
# White's perspective: pick king at sq 4, then destination sq 6
make_move(env_castle, 0, 4, 6)    # O-O
rew_castle = env_castle.rewards[0]

make_move(env_nocastle, 0, 4, 6)  # Same move
rew_nocastle = env_nocastle.rewards[0]

check("Castling bonus applied", rew_castle > rew_nocastle,
      f"with_castle={rew_castle}, without_castle={rew_nocastle}")
check("Castling bonus ~0.1", abs(rew_castle - rew_nocastle - 0.1) < 0.02,
      f"diff={rew_castle - rew_nocastle}")

# Verify the castling actually happened
white_obs = get_obs(env_castle, 0)
board = white_obs[OBS_BOARD:OBS_BOARD + 64]
check("King moved to g1", board[make_sq(0, 6)] == WK)
check("Rook moved to f1", board[make_sq(0, 5)] == WR)

env_castle.close()
env_nocastle.close()
os.unlink(fen_file)


# ============================================================================
# Test 7: New log fields present and reasonable
# ============================================================================
print("\nTest 7: New log fields (chess_moves, repetitions, derived fields)")

env = Chess(num_envs=8, max_steps=20, report_interval=99999,
            reward_capture_bonus=0.01, reward_check_bonus=0.005,
            reward_repetition=-0.01)
env.reset(seed=42)

# Play random moves to get completed episodes
for _ in range(200):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    env.step(actions)

log = binding.vec_log(env.c_envs)
has_n = log.get('n', 0) > 0
check("Got completed episodes", has_n)

if has_n:
    # New raw fields
    check("Log has chess_moves", 'chess_moves' in log,
          f"keys: {list(log.keys())}")
    check("Log has repetitions", 'repetitions' in log)

    # Derived fields
    check("Log has draw_rate", 'draw_rate' in log)
    check("Log has white_winrate", 'white_winrate' in log)
    check("Log has black_winrate", 'black_winrate' in log)
    check("Log has score", 'score' in log)

    # Verify derived field values are consistent
    draws = log.get('draws', 0)
    white_wins = log.get('white_wins', 0)
    black_wins = log.get('black_wins', 0)
    draw_rate = log.get('draw_rate', -1)
    white_winrate = log.get('white_winrate', -1)
    black_winrate = log.get('black_winrate', -1)
    score = log.get('score', -1)

    check("draw_rate == draws", abs(draw_rate - draws) < 0.001,
          f"draw_rate={draw_rate}, draws={draws}")
    check("white_winrate == white_wins", abs(white_winrate - white_wins) < 0.001,
          f"white_winrate={white_winrate}, white_wins={white_wins}")
    check("black_winrate == black_wins", abs(black_winrate - black_wins) < 0.001,
          f"black_winrate={black_winrate}, black_wins={black_wins}")
    check("score = white_winrate + 0.5*draw_rate",
          abs(score - (white_winrate + 0.5 * draw_rate)) < 0.001,
          f"score={score}, expected={white_winrate + 0.5 * draw_rate}")

    # Sanity: rates should sum to ~1
    total = white_winrate + black_winrate + draw_rate
    check("Win rates + draw rate sums to ~1.0",
          abs(total - 1.0) < 0.01,
          f"total={total}")

    # chess_moves should be >= 0
    cm = log.get('chess_moves', -1)
    check("chess_moves >= 0", cm >= 0, f"chess_moves={cm}")

    # n field
    check("n > 0", log['n'] > 0, f"n={log['n']}")

env.close()


# ============================================================================
# Test 8: Position history resets on c_reset
# ============================================================================
print("\nTest 8: Position history resets properly")

fen = "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1"
fen_file = write_fen_file([fen])

env = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0)
env.reset(seed=42)

# Do one full cycle (not enough for repetition)
make_move(env, 0, 6, 21)  # White: Ng1 -> f3
make_move(env, 1, 6, 21)  # Black: Ng8 -> f6
make_move(env, 0, 21, 6)  # White: Nf3 -> g1
make_move(env, 1, 21, 6)  # Black: Nf6 -> g8

check("Not terminal after 1 cycle", env.terminals[0] == 0)

# Reset the environment
env.reset(seed=123)

# Do the same cycle again - should NOT trigger repetition because history was reset
make_move(env, 0, 6, 21)
make_move(env, 1, 6, 21)
make_move(env, 0, 21, 6)
make_move(env, 1, 21, 6)

check("Not terminal after 1 cycle post-reset", env.terminals[0] == 0,
      f"terminals={env.terminals[0]}")

env.close()
os.unlink(fen_file)


# ============================================================================
# Test 9: Non-capturing move doesn't give material delta reward
# ============================================================================
print("\nTest 9: Non-capturing move gives no material delta")

fen = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1"
fen_file = write_fen_file([fen])

env_mat = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                reward_material=0.01, reward_valid_move=0.0)
env_mat.reset(seed=42)

env_nomat = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                  reward_material=0.0, reward_valid_move=0.0)
env_nomat.reset(seed=42)

# Move knight to empty square - no material change
make_move(env_mat, 0, 0, 17)    # Na1 -> b3
rew_mat = env_mat.rewards[0]

make_move(env_nomat, 0, 0, 17)
rew_nomat = env_nomat.rewards[0]

# Only difference should be from positional delta (knight PST changed)
# Material delta should be 0 for both (no capture), but position reward is 0 too
# So both should be equal (no material change from non-capture)
# Actually reward_material only applies to material score change. Moving a knight
# doesn't change your material (knight is still there, just moved). So delta = 0.
check("Non-capture: no material delta difference",
      abs(rew_mat - rew_nomat) < 0.001,
      f"with_mat={rew_mat}, without_mat={rew_nomat}")

env_mat.close()
env_nomat.close()
os.unlink(fen_file)


# ============================================================================
# Test 10: Queenside castling also gets bonus
# ============================================================================
print("\nTest 10: Queenside castling bonus")

castle_fen = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
fen_file = write_fen_file([castle_fen])

env_castle = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                   reward_castling=0.15)
env_castle.reset(seed=42)

env_nocastle = Chess(num_envs=1, max_steps=500, fen_file=fen_file, fen_curric_pct=1.0,
                     reward_castling=0.0)
env_nocastle.reset(seed=42)

# O-O-O: King e1(4) -> c1(2)
make_move(env_castle, 0, 4, 2)
rew_castle = env_castle.rewards[0]

make_move(env_nocastle, 0, 4, 2)
rew_nocastle = env_nocastle.rewards[0]

check("Queenside castling bonus applied", rew_castle > rew_nocastle,
      f"with={rew_castle}, without={rew_nocastle}")
check("Queenside castling bonus ~0.15", abs(rew_castle - rew_nocastle - 0.15) < 0.02,
      f"diff={rew_castle - rew_nocastle}")

# Verify castling happened
white_obs = get_obs(env_castle, 0)
board = white_obs[OBS_BOARD:OBS_BOARD + 64]
check("King moved to c1", board[make_sq(0, 2)] == WK)
check("Rook moved to d1", board[make_sq(0, 3)] == WR)

env_castle.close()
env_nocastle.close()
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
