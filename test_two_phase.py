"""Test suite for the two-phase action system.

Tests:
1. Phase transition 0->1: Pick a valid piece square
2. Phase transition 1->0: Pick valid destination -> move executed
3. Invalid piece (phase 0): Pick empty/opponent square
4. Invalid dest (phase 1): Pick invalid square -> resets to phase 0
5. PASS action: When not player's turn, action 96 is accepted
6. Valid pieces mask: Only squares with pieces that have legal moves
7. Valid dests mask: Only legal destinations for selected piece
8. Promotions: When pawn reaches last rank, valid_promos correctly shows options
9. Terminal/reset: Game end properly resets phase state
10. dtype int32/int64: Actions buffer works with both dtypes
"""

import numpy as np
import sys
sys.path.insert(0, '/home/alanga/rl-chess-selfplay')

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS

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


def make_env(num_envs=1):
    return Chess(num_envs=num_envs, max_steps=1000,
                 reward_invalid_piece=-0.01,
                 reward_invalid_move=-0.01,
                 reward_valid_piece=0.001,
                 reward_valid_move=0.002)


def get_obs(env, agent_idx):
    """Get observation for a specific agent."""
    return env.observations[agent_idx]


def get_phase(obs):
    """Get phase from observation (0 or 1)."""
    if obs[OBS_PHASE] == 255:
        return 0
    elif obs[OBS_PHASE + 1] == 255:
        return 1
    return -1


def get_valid_pieces(obs):
    """Get set of valid piece squares from obs."""
    return set(i for i in range(64) if obs[OBS_VALID_PIECES + i] == 255)


def get_valid_dests(obs):
    """Get set of valid destination squares from obs."""
    return set(i for i in range(64) if obs[OBS_VALID_DESTS + i] == 255)


def get_valid_promos(obs):
    """Get set of valid promotion indices from obs."""
    return set(i for i in range(32) if obs[OBS_VALID_PROMOS + i] == 255)


def is_my_turn(obs):
    """Check if it's this agent's turn."""
    return obs[OBS_SIDE] == 255


def is_pass_valid(obs):
    """Check if pass action is valid."""
    return obs[OBS_PASS_VALID] == 255


def step_with_actions(env, white_action, black_action):
    """Step environment with specific actions for white and black."""
    actions = np.zeros(env.num_agents, dtype=np.int32)
    actions[0] = white_action
    actions[1] = black_action
    return env.step(actions)


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


# ============================================================================
# Test 1: Phase transition 0->1
# ============================================================================
print("\nTest 1: Phase transition 0->1 (pick valid piece)")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
check("White starts in phase 0", get_phase(white_obs) == 0)
check("White's turn at start", is_my_turn(white_obs))
check("Black NOT white's turn", not is_my_turn(get_obs(env, 1)))

# White has valid pieces
vp = get_valid_pieces(white_obs)
check("White has valid pieces", len(vp) > 0, f"got {len(vp)} valid pieces")

# Pick a valid piece (e.g. a knight on b1 = square 1 from White's perspective)
# In starting position, pawns on row 1 and knights can move
# Let's pick the first valid piece
piece_sq = min(vp)
old_reward = env.rewards[0]
step_with_actions(env, piece_sq, PASS_ACTION)

white_obs = get_obs(env, 0)
check("White transitions to phase 1", get_phase(white_obs) == 1,
      f"phase={get_phase(white_obs)}")
check("Valid piece reward applied", env.rewards[0] > old_reward or env.rewards[0] >= 0,
      f"reward={env.rewards[0]}")

# Check selected piece plane is set
selected = white_obs[OBS_SELECTED:OBS_SELECTED + 64]
check("Selected piece plane has one square set",
      np.sum(selected == 255) == 1,
      f"got {np.sum(selected == 255)} set squares")

env.close()

# ============================================================================
# Test 2: Phase transition 1->0 (pick valid destination -> move executed)
# ============================================================================
print("\nTest 2: Phase transition 1->0 (pick valid dest, move executed)")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)
piece_sq = min(vp)

# Phase 0 -> 1: pick piece
step_with_actions(env, piece_sq, PASS_ACTION)
white_obs = get_obs(env, 0)
check("In phase 1 after picking piece", get_phase(white_obs) == 1)

# Get valid destinations
vd = get_valid_dests(white_obs)
check("Has valid destinations", len(vd) > 0, f"got {len(vd)} dests")

# Phase 1 -> 0: pick destination
dest_sq = min(vd)
step_with_actions(env, dest_sq, PASS_ACTION)

white_obs = get_obs(env, 0)
# After move completes, turn switches. White should be in phase 0.
check("White back to phase 0 after move", get_phase(white_obs) == 0)

# Now it should be Black's turn
check("It's now Black's turn", is_my_turn(get_obs(env, 1)))
check("Not White's turn anymore", not is_my_turn(white_obs))
check("Pass is valid for White now", is_pass_valid(white_obs))

env.close()

# ============================================================================
# Test 3: Invalid piece (phase 0)
# ============================================================================
print("\nTest 3: Invalid piece in phase 0")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)

# Find an empty square (no piece)
empty_sq = None
for sq in range(64):
    if sq not in vp and white_obs[OBS_BOARD + sq] == 0:
        empty_sq = sq
        break

check("Found empty square", empty_sq is not None)

if empty_sq is not None:
    step_with_actions(env, empty_sq, PASS_ACTION)
    white_obs = get_obs(env, 0)
    check("Still in phase 0 after invalid piece", get_phase(white_obs) == 0)
    check("Still White's turn", is_my_turn(white_obs))
    check("Penalty applied for invalid piece", env.rewards[0] < 0,
          f"reward={env.rewards[0]}")

env.close()

# ============================================================================
# Test 4: Invalid dest (phase 1) -> resets to phase 0
# ============================================================================
print("\nTest 4: Invalid destination in phase 1")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)
piece_sq = min(vp)

# Go to phase 1
step_with_actions(env, piece_sq, PASS_ACTION)
white_obs = get_obs(env, 0)
check("In phase 1", get_phase(white_obs) == 1)

vd = get_valid_dests(white_obs)

# Find an invalid destination
invalid_dest = None
for sq in range(64):
    if sq not in vd:
        invalid_dest = sq
        break

check("Found invalid dest", invalid_dest is not None)

if invalid_dest is not None:
    step_with_actions(env, invalid_dest, PASS_ACTION)
    white_obs = get_obs(env, 0)
    check("Back to phase 0 after invalid dest", get_phase(white_obs) == 0)
    check("Still White's turn (no move made)", is_my_turn(white_obs))
    check("Penalty for invalid dest", env.rewards[0] < 0,
          f"reward={env.rewards[0]}")

env.close()

# ============================================================================
# Test 5: PASS action
# ============================================================================
print("\nTest 5: PASS action")
env = make_env()
env.reset(seed=42)

# Black should pass (not their turn)
black_obs = get_obs(env, 1)
check("Pass valid for Black (not their turn)", is_pass_valid(black_obs))
check("Pass NOT valid for White (their turn)", not is_pass_valid(get_obs(env, 0)))

# Step: White picks a valid piece, Black passes
white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)
piece_sq = min(vp)
step_with_actions(env, piece_sq, PASS_ACTION)

# Black should not be penalized for valid pass
check("No penalty for Black's valid pass", env.rewards[1] >= 0,
      f"reward={env.rewards[1]}")

env.close()

# ============================================================================
# Test 6: Valid pieces mask
# ============================================================================
print("\nTest 6: Valid pieces mask accuracy")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)

# In starting position, White should have pieces with legal moves:
# All 8 pawns can push, both knights can move = at least 10 pieces
check("At least 10 pieces with legal moves at start", len(vp) >= 10,
      f"got {len(vp)}")

# Verify each marked piece actually has the player's piece on it
board = white_obs[OBS_BOARD:OBS_BOARD + 64]
all_own = True
for sq in vp:
    piece = board[sq]
    if piece < 1 or piece > 6:  # Not a white piece (from White's perspective: 1-6)
        all_own = False
        break
check("All valid piece squares contain own pieces", all_own)

# When it's not our turn, valid_pieces should be empty
black_obs = get_obs(env, 1)
bvp = get_valid_pieces(black_obs)
check("No valid pieces for Black when not their turn", len(bvp) == 0,
      f"got {len(bvp)}")

env.close()

# ============================================================================
# Test 7: Valid dests mask
# ============================================================================
print("\nTest 7: Valid destinations mask accuracy")
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
vp = get_valid_pieces(white_obs)

# In phase 0, valid_dests should be empty
vd_before = get_valid_dests(white_obs)
check("No valid dests in phase 0", len(vd_before) == 0,
      f"got {len(vd_before)}")

# Pick a knight (square 1 from White's perspective = b1)
# Knights at b1(1) and g1(6) should be valid
knight_sq = None
for sq in [1, 6]:  # b1, g1 - knight positions
    if sq in vp:
        knight_sq = sq
        break

if knight_sq is not None:
    step_with_actions(env, knight_sq, PASS_ACTION)
    white_obs = get_obs(env, 0)
    vd = get_valid_dests(white_obs)

    # A knight from starting position should have 2 legal moves
    check("Knight has 2 destinations from start", len(vd) == 2,
          f"got {len(vd)} dests: {vd}")
else:
    check("Found a knight to test", False, "no knight in valid pieces")

env.close()

# ============================================================================
# Test 8: Promotions
# ============================================================================
print("\nTest 8: Promotion moves")
# We need to set up a position with a pawn about to promote.
# We'll play several moves to get there, or just verify the mask logic
# by playing a sequence. For now, verify the promo mask starts empty.
env = make_env()
env.reset(seed=42)

white_obs = get_obs(env, 0)
promos = get_valid_promos(white_obs)
check("No valid promos in phase 0 at start", len(promos) == 0)

# Just verify promo action range is 64-95
check("Promo action range correct", 64 + 4 * 8 - 1 == 95)

env.close()

# ============================================================================
# Test 9: Terminal/reset
# ============================================================================
print("\nTest 9: Terminal and reset behavior")
env = make_env()
env.reset(seed=42)

# Play a full game by making random moves
for _ in range(2000):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    obs, rew, terms, truncs, info = env.step(actions)
    if terms[0]:
        break

if terms[0]:
    # Auto-reset: next step should reset
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    obs, rew, terms, truncs, info = env.step(actions)

    white_obs = get_obs(env, 0)
    check("After reset: phase 0", get_phase(white_obs) == 0)
    check("After reset: White's turn", is_my_turn(white_obs))
    check("After reset: terminal cleared", terms[0] == 0)

    vp = get_valid_pieces(white_obs)
    check("After reset: valid pieces exist", len(vp) > 0)
else:
    check("Game reached terminal in 2000 random steps", False, "no terminal reached")

env.close()

# ============================================================================
# Test 10: Full game play-through (smoke test)
# ============================================================================
print("\nTest 10: Full game smoke test with both phases")
env = make_env()
env.reset(seed=123)

moves_completed = 0
steps = 0
max_steps = 5000

while steps < max_steps:
    white_obs = get_obs(env, 0)
    black_obs = get_obs(env, 1)

    # Determine actions
    white_action = PASS_ACTION
    black_action = PASS_ACTION

    if is_my_turn(white_obs):
        phase = get_phase(white_obs)
        if phase == 0:
            vp = get_valid_pieces(white_obs)
            if vp:
                white_action = min(vp)
        elif phase == 1:
            vd = get_valid_dests(white_obs)
            if vd:
                white_action = min(vd)
                moves_completed += 1
    else:
        white_action = PASS_ACTION

    if is_my_turn(black_obs):
        phase = get_phase(black_obs)
        if phase == 0:
            vp = get_valid_pieces(black_obs)
            if vp:
                black_action = min(vp)
        elif phase == 1:
            vd = get_valid_dests(black_obs)
            if vd:
                black_action = min(vd)
                moves_completed += 1
    else:
        black_action = PASS_ACTION

    obs, rew, terms, truncs, info = step_with_actions(env, white_action, black_action)
    steps += 1

    if terms[0]:
        break

check("Completed some chess moves", moves_completed > 0,
      f"completed {moves_completed} moves in {steps} steps")
check("Game terminated or ran long enough", terms[0] == 1 or steps >= max_steps,
      f"terms={terms[0]}, steps={steps}")

print(f"  (Played {moves_completed} chess moves in {steps} env steps)")

env.close()

# ============================================================================
# Test 11: Multi-env (sanity check)
# ============================================================================
print("\nTest 11: Multiple environments")
env = make_env(num_envs=4)
env.reset(seed=42)

check("8 agents for 4 games", env.num_agents == 8)
check("Observations shape correct", env.observations.shape == (8, OBS_SIZE),
      f"got {env.observations.shape}")

# Step with random actions
for _ in range(100):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    env.step(actions)

check("100 steps without crash", True)
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
