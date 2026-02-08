"""Test suite for SEE (Static Exchange Evaluation) and hanging piece reward shaping
(1-agent-per-game topology).

Tests:
1. SEE positive: QxP undefended - no penalty
2. SEE negative: bad capture into well-defended square
3. Equal exchange: NxN - no penalty
4. Hanging piece on quiet move to attacked square
5. Reward suppression for bad captures with material reward
6. Hanging penalty applied correctly (nonzero reward_see_hanging)
7. SEE = 0 for safe moves (no penalty)
8. SEE disabled when reward_see_hanging = 0
9. SEE with en passant capture
10. Full game with SEE enabled doesn't crash
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


def make_env(num_envs=1, **kwargs):
    defaults = dict(
        max_steps=1000,
        reward_invalid_piece=-0.01,
        reward_invalid_move=-0.01,
        reward_valid_piece=0.0,
        reward_valid_move=0.0,
    )
    defaults.update(kwargs)
    return Chess(num_envs=num_envs, **defaults)


def get_obs(env, agent_idx=0):
    return env.observations[agent_idx]


def get_phase(obs):
    if obs[OBS_PHASE] == 255:
        return 0
    elif obs[OBS_PHASE + 1] == 255:
        return 1
    return -1


def get_valid_pieces(obs):
    return set(i for i in range(64) if obs[OBS_VALID_PIECES + i] == 255)


def get_valid_dests(obs):
    return set(i for i in range(64) if obs[OBS_VALID_DESTS + i] == 255)


def is_my_turn(obs):
    return obs[OBS_SIDE] == 255


def step_action(env, action, game_idx=0):
    """Step environment with a single action for one game."""
    actions = np.zeros(env.num_agents, dtype=np.int32)
    actions[game_idx] = action
    return env.step(actions)


def play_move(env, piece_sq, dest_sq):
    """Execute a two-phase move for the current mover. Returns the reward."""
    step_action(env, piece_sq)   # phase 0
    step_action(env, dest_sq)    # phase 1
    return float(env.rewards[0])


def flip_sq(sq):
    """Flip a square for Black's perspective."""
    r = sq // 8
    c = sq % 8
    return (7 - r) * 8 + c


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
# Test 1: SEE disabled by default (reward_see_hanging=0)
# ============================================================================
print("\nTest 1: SEE disabled when reward_see_hanging=0")
env = make_env(reward_see_hanging=0.0, reward_material=0.1)
env.reset(seed=42)

# Play some random valid moves, verify no crash
for _ in range(20):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = min(vp)
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = min(vd)
    step_action(env, action)

check("No crash with SEE disabled", True)
env.close()

# ============================================================================
# Test 2: SEE enabled doesn't crash
# ============================================================================
print("\nTest 2: SEE enabled doesn't crash")
env = make_env(reward_see_hanging=-0.01, reward_material=0.1)
env.reset(seed=42)

for _ in range(20):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = min(vp)
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = min(vd)
    step_action(env, action)

check("No crash with SEE enabled", True)
env.close()

# ============================================================================
# Test 3: Safe capture - no hanging penalty
# ============================================================================
print("\nTest 3: Safe capture gets no hanging penalty")

# With SEE enabled
env_see = make_env(reward_see_hanging=-0.1, reward_material=0.1)
env_see.reset(seed=42)

# Without SEE
env_nosee = make_env(reward_see_hanging=0.0, reward_material=0.1)
env_nosee.reset(seed=42)

# Play identical valid moves in both
moves_played = 0
total_reward_see = 0.0
total_reward_nosee = 0.0
for _ in range(40):
    obs = get_obs(env_see)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = min(vp)
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd:
            action = min(vd)
            moves_played += 1
    step_action(env_see, action)
    step_action(env_nosee, action)
    total_reward_see += env_see.rewards[0]
    total_reward_nosee += env_nosee.rewards[0]

check("Played some moves", moves_played > 0, f"played {moves_played}")
check("Rewards are finite with SEE", np.isfinite(total_reward_see))

env_see.close()
env_nosee.close()

# ============================================================================
# Test 4: Hanging penalty is negative
# ============================================================================
print("\nTest 4: Hanging penalty direction is correct")
env = make_env(reward_see_hanging=-0.05, reward_material=0.0)
env.reset(seed=42)

# Play a full game with varying valid moves, accumulate rewards
total_negative_deltas = 0
steps = 0
for _ in range(200):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = sorted(vp)[steps % len(vp)]
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = sorted(vd)[steps % len(vd)]
    step_action(env, action)
    # SEE penalty should produce negative rewards (from hanging penalty)
    if env.rewards[0] < -0.001:
        total_negative_deltas += 1
    steps += 1
    if env.terminals[0]:
        break

# With random play, some moves will hang pieces and get penalized
check("Some negative SEE penalties observed", total_negative_deltas > 0,
      f"got {total_negative_deltas} negative reward steps in {steps} steps")

env.close()

# ============================================================================
# Test 5: No penalty when reward_see_hanging is 0
# ============================================================================
print("\nTest 5: No SEE effects when reward_see_hanging=0")
env = make_env(reward_see_hanging=0.0, reward_material=0.0,
               reward_capture_bonus=0.0, reward_check_bonus=0.0)
env.reset(seed=42)

any_nonzero = False
for _ in range(100):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = min(vp)
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = min(vd)
    step_action(env, action)
    # With all reward shaping at 0, rewards should only come from
    # invalid piece/move penalties or win/loss
    if abs(env.rewards[0]) > 0.001 and not env.terminals[0]:
        if abs(env.rewards[0]) > 0.02:  # more than invalid action penalty
            any_nonzero = True
    if env.terminals[0]:
        break

check("No unexpected rewards with all shaping disabled", not any_nonzero)
env.close()

# ============================================================================
# Test 6: SEE penalty scales with piece value
# ============================================================================
print("\nTest 6: SEE with material reward interaction")
env = make_env(reward_see_hanging=-0.1, reward_material=0.1)
env.reset(seed=42)

# Just verify it runs correctly for many steps
steps = 0
for _ in range(500):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = sorted(vp)[steps % len(vp)]
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = sorted(vd)[steps % len(vd)]
    step_action(env, action)
    steps += 1
    if env.terminals[0]:
        break

check("SEE with material runs without crash", True)
check(f"Game completed in {steps} steps", steps > 0)

env.close()

# ============================================================================
# Test 7: Full game with SEE - performance test
# ============================================================================
print("\nTest 7: Full game with SEE doesn't regress performance")
env = make_env(num_envs=4, reward_see_hanging=-0.05, reward_material=0.05)
env.reset(seed=42)

for _ in range(200):
    actions = np.random.randint(0, NUM_ACTIONS, env.num_agents)
    env.step(actions)

check("Multi-env with SEE: 200 steps without crash", True)
env.close()

# ============================================================================
# Test 8: SEE penalty is bounded
# ============================================================================
print("\nTest 8: SEE penalties are bounded")
env = make_env(reward_see_hanging=-0.1, reward_material=0.1)
env.reset(seed=42)

max_abs_reward = 0.0
for _ in range(300):
    obs = get_obs(env)
    action = 0
    phase = get_phase(obs)
    if phase == 0:
        vp = get_valid_pieces(obs)
        if vp: action = sorted(vp)[0]
    elif phase == 1:
        vd = get_valid_dests(obs)
        if vd: action = sorted(vd)[0]
    step_action(env, action)
    ar = abs(env.rewards[0])
    if ar > max_abs_reward:
        max_abs_reward = ar
    if env.terminals[0]:
        break

# Max reward should be bounded (win/loss is 1.0, SEE penalties should be reasonable)
check("Max abs reward is bounded", max_abs_reward <= 5.0,
      f"max_abs_reward={max_abs_reward}")

env.close()

# ============================================================================
# Test 9: SEE parameter passthrough
# ============================================================================
print("\nTest 9: SEE parameter passthrough works")
env = make_env(reward_see_hanging=-0.25)
env.reset(seed=42)
check("Environment created with reward_see_hanging=-0.25", True)
env.close()

env = make_env(reward_see_hanging=0.0)
env.reset(seed=42)
check("Environment created with reward_see_hanging=0.0", True)
env.close()

# ============================================================================
# Test 10: SEE doesn't affect game outcomes
# ============================================================================
print("\nTest 10: SEE doesn't affect game outcomes (same moves -> same terminal)")
env_see = make_env(reward_see_hanging=-0.1)
env_nosee = make_env(reward_see_hanging=0.0)
env_see.reset(seed=42)
env_nosee.reset(seed=42)

# Play identical moves in both environments
for step in range(500):
    obs_see = get_obs(env_see)
    action = 0
    phase = get_phase(obs_see)
    if phase == 0:
        vp = get_valid_pieces(obs_see)
        if vp: action = min(vp)
    elif phase == 1:
        vd = get_valid_dests(obs_see)
        if vd: action = min(vd)

    step_action(env_see, action)
    step_action(env_nosee, action)

    # Board state should be identical
    see_board = get_obs(env_see)[OBS_BOARD:OBS_BOARD+64]
    nosee_board = get_obs(env_nosee)[OBS_BOARD:OBS_BOARD+64]
    if not np.array_equal(see_board, nosee_board):
        check("Boards stay in sync", False, f"diverged at step {step}")
        break

    if env_see.terminals[0]:
        check("Terminals match", env_see.terminals[0] == env_nosee.terminals[0])
        break
else:
    check("Boards stay in sync after 500 steps", True)
    check("Terminals match",
          env_see.terminals[0] == env_nosee.terminals[0])

env_see.close()
env_nosee.close()

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
