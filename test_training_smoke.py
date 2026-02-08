"""Smoke test: verify Policy + LSTMWrapper + PuffeRL training loop works
(1-agent-per-game topology).

Tests:
1. Policy encode/decode contract (returns 2 values each)
2. LSTMWrapper forward_eval with state dict (PufferLib contract)
3. LSTMWrapper forward (training path) with batched time steps
4. PuffeRL training loop runs for a few epochs without crash
"""

import numpy as np
import torch
import sys
sys.path.insert(0, '/home/alanga/rl-chess-selfplay')

import pufferlib
import pufferlib.vector
import pufferlib.models

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS
from train_chess import Policy, ChessLSTM

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
# Test 1: Policy encode/decode contract
# ============================================================================
print("\nTest 1: Policy encode/decode contract")
NUM_GAMES = 2
env = Chess(num_envs=NUM_GAMES)
env.reset(seed=42)

# In 1-agent topology: num_agents = num_envs
NUM_AGENTS = NUM_GAMES
check("num_agents == num_envs", env.num_agents == NUM_AGENTS,
      f"got {env.num_agents}")

base_policy = Policy(env, hidden_size=256, num_blocks=2)
obs_tensor = torch.as_tensor(env.observations).float()

hidden = base_policy.encode_observations(obs_tensor)
check("encode_observations returns tensor", isinstance(hidden, torch.Tensor))
check("hidden shape is (batch, hidden_size)", hidden.shape == (NUM_AGENTS, 256),
      f"got {hidden.shape}")

logits, value = base_policy.decode_actions(hidden)
check("decode_actions returns 2 values", True)
check("logits shape", logits.shape == (NUM_AGENTS, NUM_ACTIONS), f"got {logits.shape}")
check("value shape", value.shape == (NUM_AGENTS, 1), f"got {value.shape}")

# forward_eval also returns 2
out = base_policy.forward_eval(obs_tensor)
check("forward_eval returns 2 values", len(out) == 2, f"got {len(out)}")

env.close()

# ============================================================================
# Test 2: LSTMWrapper forward_eval with state dict
# ============================================================================
print("\nTest 2: LSTMWrapper forward_eval with state dict (PufferLib contract)")
env = Chess(num_envs=NUM_GAMES)
env.reset(seed=42)

base_policy = Policy(env, hidden_size=256, num_blocks=2)
lstm_policy = ChessLSTM(env, base_policy, input_size=256, hidden_size=256)

obs_tensor = torch.as_tensor(env.observations).float()
batch_size = obs_tensor.shape[0]

# PufferLib passes state as a dict
state = dict(
    reward=torch.zeros(batch_size),
    done=torch.zeros(batch_size),
    env_id=slice(0, batch_size),
    mask=torch.ones(batch_size, dtype=torch.bool),
    lstm_h=torch.zeros(batch_size, 256),
    lstm_c=torch.zeros(batch_size, 256),
)

logits, value = lstm_policy.forward_eval(obs_tensor, state)
check("LSTMWrapper forward_eval returns 2 values", True)
check("logits shape", logits.shape == (batch_size, NUM_ACTIONS), f"got {logits.shape}")
check("value shape", value.shape == (batch_size, 1), f"got {value.shape}")

# State dict should be mutated in-place with updated LSTM states
check("state['lstm_h'] updated in-place", state['lstm_h'].shape == (batch_size, 256))
check("state['lstm_c'] updated in-place", state['lstm_c'].shape == (batch_size, 256))
check("lstm_h not all zeros (LSTM ran)", not torch.all(state['lstm_h'] == 0).item())

env.close()

# ============================================================================
# Test 3: LSTMWrapper forward (training path)
# ============================================================================
print("\nTest 3: LSTMWrapper forward (training path with time batching)")
env = Chess(num_envs=NUM_GAMES)
env.reset(seed=42)

base_policy = Policy(env, hidden_size=256, num_blocks=2)
lstm_policy = ChessLSTM(env, base_policy, input_size=256, hidden_size=256)

B, T = 4, 8  # batch=4, time_steps=8
obs_batch = torch.randn(B, T, OBS_SIZE)
# Clamp board values to valid range
obs_batch[:, :, :64] = torch.clamp(obs_batch[:, :, :64].abs() * 12, 0, 12).byte().float()

state = dict(
    lstm_h=torch.zeros(1, B, 256),  # (num_layers, B, hidden)
    lstm_c=torch.zeros(1, B, 256),
)

logits, values = lstm_policy.forward(obs_batch, state)
check("Training forward returns 2 values", True)
check("Training logits shape", logits.shape == (B * T, NUM_ACTIONS),
      f"got {logits.shape}")
check("Training values shape", values.shape == (B, T),
      f"got {values.shape}")

env.close()

# ============================================================================
# Test 4: PuffeRL training loop smoke test
# ============================================================================
print("\nTest 4: PuffeRL training loop (3 epochs)")

try:
    from pufferlib import pufferl
    HAS_PUFFERL = True
except ImportError:
    HAS_PUFFERL = False
    print("  SKIP: pufferlib._C not available (needs CUDA build)")

if HAS_PUFFERL:
    NUM_GAMES = 16
    vecenv = pufferlib.vector.make(
        Chess,
        env_kwargs={
            'num_envs': NUM_GAMES,
            'max_steps': 64,
            'illegal_move_penalty': -0.1,
            'reward_invalid_piece': -0.01,
            'reward_invalid_move': -0.01,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_policy = Policy(vecenv, hidden_size=128, num_blocks=1)
    policy = ChessLSTM(vecenv, base_policy, input_size=128, hidden_size=128).to(device)

    check("policy.hidden_size accessible", policy.hidden_size == 128)

    args = pufferl.load_config('default')
    args['train']['total_timesteps'] = 2048
    args['train']['learning_rate'] = 1e-4
    args['train']['batch_size'] = 512
    args['train']['minibatch_size'] = 256
    args['train']['max_minibatch_size'] = 256
    args['train']['bptt_horizon'] = 8
    args['train']['update_epochs'] = 1
    args['train']['use_rnn'] = True
    args['train']['device'] = device
    args['train']['precision'] = 'float32'
    args['train']['checkpoint_interval'] = 99999
    args['train']['compile'] = False

    try:
        trainer = pufferl.PuffeRL(args['train'], vecenv, policy)
        check("PuffeRL created successfully", True)

        epochs_run = 0
        for _ in range(3):
            trainer.evaluate()
            trainer.train()
            epochs_run += 1

        check(f"Ran {epochs_run} training epochs", epochs_run == 3)
        trainer.close()
        check("Training loop completed without crash", True)
    except Exception as e:
        check("PuffeRL training loop", False, str(e))
        import traceback
        traceback.print_exc()

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
