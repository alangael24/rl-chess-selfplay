"""Test suite for draw configuration flags.

Tests reward_draw, enable_50_move_rule, and enable_threefold_repetition.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/alanga/rl-chess-selfplay')

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS

# Observation offsets
OBS_BOARD = 0
OBS_SIDE = 64
OBS_PHASE = 71
OBS_VALID_PIECES = 137
OBS_VALID_DESTS = 201
OBS_PASS_VALID = 300
PASS_ACTION = 96

passed = 0
failed = 0


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {msg}")
        failed += 1


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
    return [i for i in range(64) if obs[OBS_VALID_PIECES + i] == 255]


def get_valid_dests(obs):
    return [i for i in range(64) if obs[OBS_VALID_DESTS + i] == 255]


def step_with_actions(env, white_action, black_action):
    actions = np.array([white_action, black_action], dtype=np.int32)
    return env.step(actions)


def pick_and_move(env, player):
    """Execute a full chess move (phase 0 + phase 1) for the given player.
    Returns True if a move was executed, False if stuck."""
    obs = get_obs(env, player)
    if not is_my_turn(obs):
        return False

    valid_pieces = get_valid_pieces(obs)
    if not valid_pieces:
        return False

    # Phase 0: pick a piece
    piece_sq = valid_pieces[0]
    if player == 0:
        step_with_actions(env, piece_sq, PASS_ACTION)
    else:
        step_with_actions(env, PASS_ACTION, piece_sq)

    # Phase 1: pick a destination
    obs = get_obs(env, player)
    valid_dests = get_valid_dests(obs)
    if not valid_dests:
        return False

    dest_sq = valid_dests[0]
    if player == 0:
        step_with_actions(env, dest_sq, PASS_ACTION)
    else:
        step_with_actions(env, PASS_ACTION, dest_sq)

    return True


def play_knight_cycle(env, count=1):
    """Play Ng1-f3/Ng8-f6 and back to force repetition.
    Each cycle = 4 chess moves (2 per side).
    Returns the number of completed cycles."""
    # Square mapping:
    # g1=6, f3=21, g8=62(abs)->flip to 6(black), f6=45(abs)->flip to 21(black)
    # From black's perspective: g8 is flip(62)=make_sq(7-7,6)=make_sq(0,6)=6
    # f6 is flip(45)=make_sq(7-5,5)=make_sq(2,5)=21
    for c in range(count):
        # Check terminal
        if env.terminals[0] == 1:
            return c

        # White: Ng1-f3 (g1=6, f3=21)
        step_with_actions(env, 6, PASS_ACTION)   # pick knight at g1
        if env.terminals[0] == 1:
            return c
        step_with_actions(env, 21, PASS_ACTION)  # move to f3
        if env.terminals[0] == 1:
            return c

        # Black: Ng8-f6 (black perspective: g8=6, f6=21)
        step_with_actions(env, PASS_ACTION, 6)   # pick knight at g8
        if env.terminals[0] == 1:
            return c
        step_with_actions(env, PASS_ACTION, 21)  # move to f6
        if env.terminals[0] == 1:
            return c

        # White: Nf3-g1 (f3=21, g1=6)
        step_with_actions(env, 21, PASS_ACTION)  # pick knight at f3
        if env.terminals[0] == 1:
            return c
        step_with_actions(env, 6, PASS_ACTION)   # move to g1
        if env.terminals[0] == 1:
            return c

        # Black: Nf6-g8 (black perspective: f6=21, g8=6)
        step_with_actions(env, PASS_ACTION, 21)  # pick knight at f6
        if env.terminals[0] == 1:
            return c
        step_with_actions(env, PASS_ACTION, 6)   # move to g8
        if env.terminals[0] == 1:
            return c

    return count


# ============================================================================
# Test 1: Default behavior - reward_draw=0.0 (no draw reward)
# ============================================================================
print("\nTest 1: Default reward_draw=0.0 (backward compat)")
# Use reward_invalid_move=0 to isolate draw reward from invalid-pass penalties
env = Chess(num_envs=1, max_steps=10, reward_invalid_move=0.0, reward_invalid_piece=0.0)
env.reset(seed=42)
# Play until truncation
for _ in range(100):
    if env.terminals[0] == 1:
        break
    actions = np.array([PASS_ACTION, PASS_ACTION], dtype=np.int32)
    env.step(actions)

# The env should truncate at max_steps=10. Check rewards are 0.
check("Default draw: terminal", env.terminals[0] == 1)
check("Default draw: reward=0 for white", env.rewards[0] == 0.0,
      f"got {env.rewards[0]}")
check("Default draw: reward=0 for black", env.rewards[1] == 0.0,
      f"got {env.rewards[1]}")
env.close()

# ============================================================================
# Test 2: Positive reward_draw on truncation
# ============================================================================
print("\nTest 2: Positive reward_draw=0.5 on truncation")
env = Chess(num_envs=1, max_steps=10, reward_draw=0.5,
            reward_invalid_move=0.0, reward_invalid_piece=0.0)
env.reset(seed=42)
for _ in range(100):
    if env.terminals[0] == 1:
        break
    actions = np.array([PASS_ACTION, PASS_ACTION], dtype=np.int32)
    env.step(actions)

check("Positive draw reward: terminal", env.terminals[0] == 1)
check("Positive draw reward: white=0.5", abs(env.rewards[0] - 0.5) < 0.001,
      f"got {env.rewards[0]}")
check("Positive draw reward: black=0.5", abs(env.rewards[1] - 0.5) < 0.001,
      f"got {env.rewards[1]}")
env.close()

# ============================================================================
# Test 3: Negative reward_draw on truncation
# ============================================================================
print("\nTest 3: Negative reward_draw=-0.5 on truncation")
env = Chess(num_envs=1, max_steps=10, reward_draw=-0.5,
            reward_invalid_move=0.0, reward_invalid_piece=0.0)
env.reset(seed=42)
for _ in range(100):
    if env.terminals[0] == 1:
        break
    actions = np.array([PASS_ACTION, PASS_ACTION], dtype=np.int32)
    env.step(actions)

check("Negative draw reward: terminal", env.terminals[0] == 1)
check("Negative draw reward: white=-0.5", abs(env.rewards[0] - (-0.5)) < 0.001,
      f"got {env.rewards[0]}")
check("Negative draw reward: black=-0.5", abs(env.rewards[1] - (-0.5)) < 0.001,
      f"got {env.rewards[1]}")
env.close()

# ============================================================================
# Test 4: reward_draw on threefold repetition
# ============================================================================
print("\nTest 4: reward_draw=0.3 on threefold repetition")
env = Chess(num_envs=1, max_steps=1000, reward_draw=0.3)
env.reset(seed=42)

# Play knight back and forth to trigger threefold repetition
# Need 3 cycles: initial + cycle1 = 2nd occurrence, +cycle2 = 3rd occurrence
cycles = play_knight_cycle(env, count=3)

check("Repetition: game terminated", env.terminals[0] == 1)
# Rewards should include 0.3 for the draw
check("Repetition: white reward includes draw",
      abs(env.rewards[0] - 0.3) < 0.1 or env.rewards[0] >= 0.25,
      f"got {env.rewards[0]}")
check("Repetition: black reward includes draw",
      abs(env.rewards[1] - 0.3) < 0.1 or env.rewards[1] >= 0.25,
      f"got {env.rewards[1]}")
env.close()

# ============================================================================
# Test 5: reward_draw on stalemate/insufficient material
# ============================================================================
print("\nTest 5: reward_draw on stalemate (via insufficient material FEN)")
# K vs K position - should be detected as GAME_INSUFFICIENT
import tempfile, os
fen_content = "8/8/8/4k3/8/8/8/4K3 w - - 0 1\n"  # K vs K
fen_path = os.path.join(tempfile.gettempdir(), "test_draw_kk.fen")
with open(fen_path, 'w') as f:
    f.write(fen_content)

env = Chess(num_envs=1, max_steps=1000, reward_draw=0.25,
            fen_file=fen_path, fen_curric_pct=1.0)
env.reset(seed=42)

# Play a few moves - insufficient material should be detected
for _ in range(20):
    if env.terminals[0] == 1:
        break
    obs_w = get_obs(env, 0)
    if is_my_turn(obs_w):
        vp = get_valid_pieces(obs_w)
        if vp:
            step_with_actions(env, vp[0], PASS_ACTION)
            obs_w = get_obs(env, 0)
            vd = get_valid_dests(obs_w)
            if vd:
                step_with_actions(env, vd[0], PASS_ACTION)
            else:
                step_with_actions(env, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env, PASS_ACTION, PASS_ACTION)
    else:
        obs_b = get_obs(env, 1)
        vp = get_valid_pieces(obs_b)
        if vp:
            step_with_actions(env, PASS_ACTION, vp[0])
            obs_b = get_obs(env, 1)
            vd = get_valid_dests(obs_b)
            if vd:
                step_with_actions(env, PASS_ACTION, vd[0])
            else:
                step_with_actions(env, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env, PASS_ACTION, PASS_ACTION)

check("K vs K: game terminated", env.terminals[0] == 1)
# Both rewards should include the draw reward
check("K vs K: white got draw reward",
      abs(env.rewards[0] - 0.25) < 0.05,
      f"got {env.rewards[0]}")
check("K vs K: black got draw reward",
      abs(env.rewards[1] - 0.25) < 0.05,
      f"got {env.rewards[1]}")
env.close()
os.remove(fen_path)

# ============================================================================
# Test 6: enable_threefold_repetition=0 disables repetition draw
# ============================================================================
print("\nTest 6: enable_threefold_repetition=0 disables repetition draw")
env = Chess(num_envs=1, max_steps=1000, enable_threefold_repetition=0)
env.reset(seed=42)

# Play knight cycles that would normally trigger threefold repetition
cycles = play_knight_cycle(env, count=4)

# With threefold disabled, game should NOT have terminated from repetition
# (may still be ongoing or terminated from something else)
check("Threefold disabled: completed 4 cycles without repetition end",
      cycles == 4,
      f"only completed {cycles} cycles")
env.close()

# ============================================================================
# Test 7: enable_50_move_rule=0 disables 50-move draw
# ============================================================================
print("\nTest 7: enable_50_move_rule=0 disables 50-move draw")
# Use a FEN position with halfmove clock near 100
fen_near50 = "8/8/8/4k3/8/8/4K3/4R3 w - - 98 1\n"  # K+R vs K, halfmove=98
fen_path = os.path.join(tempfile.gettempdir(), "test_50move.fen")
with open(fen_path, 'w') as f:
    f.write(fen_near50)

# With 50-move rule enabled (default): should end after 2 non-capture moves
env_enabled = Chess(num_envs=1, max_steps=1000,
                    fen_file=fen_path, fen_curric_pct=1.0,
                    enable_50_move_rule=1)
env_enabled.reset(seed=42)
terminated_at_50 = False
for _ in range(20):
    if env_enabled.terminals[0] == 1:
        terminated_at_50 = True
        break
    obs_w = get_obs(env_enabled, 0)
    if is_my_turn(obs_w):
        vp = get_valid_pieces(obs_w)
        if vp:
            step_with_actions(env_enabled, vp[0], PASS_ACTION)
            obs_w = get_obs(env_enabled, 0)
            vd = get_valid_dests(obs_w)
            if vd:
                step_with_actions(env_enabled, vd[0], PASS_ACTION)
            else:
                step_with_actions(env_enabled, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env_enabled, PASS_ACTION, PASS_ACTION)
    else:
        obs_b = get_obs(env_enabled, 1)
        vp = get_valid_pieces(obs_b)
        if vp:
            step_with_actions(env_enabled, PASS_ACTION, vp[0])
            obs_b = get_obs(env_enabled, 1)
            vd = get_valid_dests(obs_b)
            if vd:
                step_with_actions(env_enabled, PASS_ACTION, vd[0])
            else:
                step_with_actions(env_enabled, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env_enabled, PASS_ACTION, PASS_ACTION)

check("50-move enabled: game terminated", terminated_at_50)
env_enabled.close()

# With 50-move rule disabled: should NOT end from 50-move rule
env_disabled = Chess(num_envs=1, max_steps=1000,
                     fen_file=fen_path, fen_curric_pct=1.0,
                     enable_50_move_rule=0)
env_disabled.reset(seed=42)
moves_without_terminal = 0
for _ in range(20):
    if env_disabled.terminals[0] == 1:
        break
    moves_without_terminal += 1
    obs_w = get_obs(env_disabled, 0)
    if is_my_turn(obs_w):
        vp = get_valid_pieces(obs_w)
        if vp:
            step_with_actions(env_disabled, vp[0], PASS_ACTION)
            obs_w = get_obs(env_disabled, 0)
            vd = get_valid_dests(obs_w)
            if vd:
                step_with_actions(env_disabled, vd[0], PASS_ACTION)
            else:
                step_with_actions(env_disabled, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env_disabled, PASS_ACTION, PASS_ACTION)
    else:
        obs_b = get_obs(env_disabled, 1)
        vp = get_valid_pieces(obs_b)
        if vp:
            step_with_actions(env_disabled, PASS_ACTION, vp[0])
            obs_b = get_obs(env_disabled, 1)
            vd = get_valid_dests(obs_b)
            if vd:
                step_with_actions(env_disabled, PASS_ACTION, vd[0])
            else:
                step_with_actions(env_disabled, PASS_ACTION, PASS_ACTION)
        else:
            step_with_actions(env_disabled, PASS_ACTION, PASS_ACTION)

check("50-move disabled: survived more steps",
      moves_without_terminal > 4,
      f"only {moves_without_terminal} steps before terminal")
env_disabled.close()
os.remove(fen_path)

# ============================================================================
# Test 8: Both flags disabled - game continues longer
# ============================================================================
print("\nTest 8: Both draw rules disabled - game continues through repetitions")
env = Chess(num_envs=1, max_steps=1000,
            enable_threefold_repetition=0, enable_50_move_rule=0)
env.reset(seed=42)

# Do many knight cycles - should not terminate from repetition or 50-move
cycles = play_knight_cycle(env, count=6)
check("Both disabled: completed 6 cycles", cycles == 6,
      f"only completed {cycles}")
check("Both disabled: game still ongoing", env.terminals[0] == 0)
env.close()

# ============================================================================
# Test 9: reward_draw works with flags disabled (on truncation)
# ============================================================================
print("\nTest 9: reward_draw with rules disabled still works on truncation")
env = Chess(num_envs=1, max_steps=10, reward_draw=-0.3,
            enable_threefold_repetition=0, enable_50_move_rule=0,
            reward_invalid_move=0.0, reward_invalid_piece=0.0)
env.reset(seed=42)
for _ in range(100):
    if env.terminals[0] == 1:
        break
    actions = np.array([PASS_ACTION, PASS_ACTION], dtype=np.int32)
    env.step(actions)

check("Truncation with flags disabled: terminal", env.terminals[0] == 1)
check("Truncation with flags disabled: white=-0.3",
      abs(env.rewards[0] - (-0.3)) < 0.001,
      f"got {env.rewards[0]}")
check("Truncation with flags disabled: black=-0.3",
      abs(env.rewards[1] - (-0.3)) < 0.001,
      f"got {env.rewards[1]}")
env.close()

# ============================================================================
# Test 10: Default values preserve backward compatibility
# ============================================================================
print("\nTest 10: Default values preserve backward compatibility")
env = Chess(num_envs=1, max_steps=1000)
env.reset(seed=42)

# With defaults, threefold repetition should still trigger
cycles = play_knight_cycle(env, count=3)
check("Defaults: repetition triggers", env.terminals[0] == 1)
check("Defaults: no extra reward from draw",
      abs(env.rewards[0]) < 0.01 and abs(env.rewards[1]) < 0.01,
      f"white={env.rewards[0]}, black={env.rewards[1]}")
env.close()

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
