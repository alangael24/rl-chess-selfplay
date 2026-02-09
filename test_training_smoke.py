"""Smoke test: feed-forward Policy + PuffeRL training loop.

Tests:
1. Policy encode/decode contract (returns logits/value with correct shapes)
2. Policy forward_eval works on environment observations
3. PuffeRL loop runs for a few epochs with use_rnn=False
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alanga/rl-chess-selfplay")

import pufferlib
import pufferlib.vector

from chess_env import Chess, NUM_ACTIONS
from train_chess import Policy

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


print("\nTest 1: Policy encode/decode contract")
NUM_GAMES = 4
env = Chess(num_envs=NUM_GAMES)
env.reset(seed=42)

obs_tensor = torch.as_tensor(env.observations).float()
policy = Policy(env, hidden_size=256)
hidden = policy.encode_observations(obs_tensor)
check("encode_observations returns tensor", isinstance(hidden, torch.Tensor))
check("hidden shape is (batch, hidden_size)", hidden.shape == (NUM_GAMES, 256), f"got {hidden.shape}")

logits, value = policy.decode_actions(hidden)
check("logits shape", logits.shape == (NUM_GAMES, NUM_ACTIONS), f"got {logits.shape}")
check("value shape", value.shape == (NUM_GAMES, 1), f"got {value.shape}")
env.close()


print("\nTest 2: Policy forward_eval on env observations")
env = Chess(num_envs=NUM_GAMES)
env.reset(seed=7)
obs_tensor = torch.as_tensor(env.observations).float()
policy = Policy(env, hidden_size=256)
out = policy.forward_eval(obs_tensor)
check("forward_eval returns 2 values", len(out) == 2, f"got {len(out)}")
check("forward_eval logits shape", out[0].shape == (NUM_GAMES, NUM_ACTIONS), f"got {out[0].shape}")
check("forward_eval value shape", out[1].shape == (NUM_GAMES, 1), f"got {out[1].shape}")
env.close()


print("\nTest 3: PuffeRL training loop (feed-forward, 3 epochs)")
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
            "num_envs": NUM_GAMES,
            "max_steps": 64,
            "illegal_move_penalty": -0.1,
            "reward_invalid_piece": -0.01,
            "reward_invalid_move": -0.01,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(vecenv, hidden_size=128).to(device)

    args = pufferl.load_config("default")
    args["train"]["total_timesteps"] = 2048
    args["train"]["learning_rate"] = 1e-4
    args["train"]["batch_size"] = 512
    args["train"]["minibatch_size"] = 256
    args["train"]["max_minibatch_size"] = 256
    args["train"]["update_epochs"] = 1
    args["train"]["use_rnn"] = False
    args["train"]["device"] = device
    args["train"]["precision"] = "float32"
    args["train"]["checkpoint_interval"] = 99999
    args["train"]["compile"] = False

    try:
        trainer = pufferl.PuffeRL(args["train"], vecenv, policy)
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


print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
print("=" * 60)

