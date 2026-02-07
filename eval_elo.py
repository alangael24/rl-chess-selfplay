"""Elo evaluation for Chess self-play checkpoints.

Plays matches between trained checkpoints and/or a random baseline,
computes Elo ratings from win/draw/loss results, and prints a summary.

Usage:
  # Random vs random sanity check
  python3 eval_elo.py --num-games 20

  # Evaluate a checkpoint against random
  python3 eval_elo.py --checkpoint path/to/model.pt --num-games 100

  # Two checkpoints against each other
  python3 eval_elo.py --checkpoint model_a.pt --baseline model_b.pt --num-games 100
"""

import numpy as np
import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS
from train_chess import Policy, ChessLSTM


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def elo_from_winrate(winrate, base_elo=1000):
    """Estimate Elo difference from win rate against a baseline."""
    if winrate <= 0.0:
        return base_elo - 400
    if winrate >= 1.0:
        return base_elo + 400
    return base_elo - 400.0 * np.log10(1.0 / winrate - 1.0)


class RandomPolicy:
    """Uniform random policy — returns random logits each call."""
    def forward_eval(self, obs, state=None):
        batch = obs.shape[0]
        logits = torch.randn(batch, NUM_ACTIONS, device=obs.device)
        value = torch.zeros(batch, 1, device=obs.device)
        return logits, value


class LSTMPolicyWrapper:
    """Wraps a ChessLSTM for stateful evaluation across steps.

    Maintains per-agent LSTM hidden state and passes it as the dict
    that LSTMWrapper.forward_eval expects.
    """
    def __init__(self, lstm_policy, num_agents, device='cpu'):
        self.policy = lstm_policy
        self.hidden_size = lstm_policy.hidden_size
        self.device = device
        self.h = torch.zeros(num_agents, self.hidden_size, device=device)
        self.c = torch.zeros(num_agents, self.hidden_size, device=device)

    def forward_eval(self, obs, state=None):
        # Select hidden state for the agents in this batch
        batch = obs.shape[0]
        st = dict(
            lstm_h=self.h[:batch],
            lstm_c=self.c[:batch],
        )
        logits, value = self.policy.forward_eval(obs, st)
        # Update stored state from the mutated dict
        self.h[:batch] = st['lstm_h'].detach()
        self.c[:batch] = st['lstm_c'].detach()
        return logits, value

    def reset_state(self):
        self.h.zero_()
        self.c.zero_()


def _load_state_dict(checkpoint_path, device='cpu'):
    """Load and extract state dict from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'policy_state_dict' in ckpt:
        return ckpt['policy_state_dict']
    elif 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def load_policy(checkpoint_path, hidden_size=256, num_blocks=2, device='cpu'):
    """Load a trained base Policy (no LSTM) from a checkpoint."""
    policy = Policy(env=None, hidden_size=hidden_size, num_blocks=num_blocks)
    raw_sd = _load_state_dict(checkpoint_path, device)

    # Strip LSTM wrapper prefix if present (e.g. "policy.board_stem.weight")
    cleaned = {}
    for k, v in raw_sd.items():
        if k.startswith('policy.'):
            cleaned[k[len('policy.'):]] = v
        else:
            cleaned[k] = v

    # Filter to only keys that exist in the base Policy
    policy_keys = set(policy.state_dict().keys())
    filtered = {k: v for k, v in cleaned.items() if k in policy_keys}

    if not filtered:
        print(f"WARNING: No matching keys found in checkpoint. "
              f"Checkpoint keys (first 10): {list(cleaned.keys())[:10]}")
        print(f"Policy keys (first 10): {list(policy_keys)[:10]}")
    else:
        missing = policy_keys - set(filtered.keys())
        if missing:
            print(f"WARNING: {len(missing)} missing keys: {list(missing)[:5]}...")
        policy.load_state_dict(filtered, strict=False)

    policy.to(device)
    policy.eval()
    return policy


def load_lstm_policy(checkpoint_path, num_agents, hidden_size=256, num_blocks=2, device='cpu'):
    """Load the full ChessLSTM wrapper with LSTM weights from a checkpoint."""
    # Create a dummy env-like object for ChessLSTM constructor
    import gymnasium
    class _DummyEnv:
        single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
    base = Policy(env=None, hidden_size=hidden_size, num_blocks=num_blocks)
    lstm_policy = ChessLSTM(_DummyEnv(), base, input_size=hidden_size, hidden_size=hidden_size)

    raw_sd = _load_state_dict(checkpoint_path, device)
    # Try loading directly (full ChessLSTM state dict)
    try:
        lstm_policy.load_state_dict(raw_sd, strict=True)
    except RuntimeError:
        # Keys may not match exactly — try lenient load
        lstm_policy.load_state_dict(raw_sd, strict=False)
        loaded = set(raw_sd.keys()) & set(lstm_policy.state_dict().keys())
        total = set(lstm_policy.state_dict().keys())
        if len(loaded) < len(total):
            print(f"WARNING: Loaded {len(loaded)}/{len(total)} keys into ChessLSTM")

    lstm_policy.to(device)
    lstm_policy.eval()
    return LSTMPolicyWrapper(lstm_policy, num_agents, device)


def play_match(policy_white, policy_black, num_games=100, max_steps=512, device='cpu'):
    """Play num_games between white and black policies.

    Returns (white_wins, draws, black_wins).
    """
    env = Chess(num_envs=num_games, max_steps=max_steps)
    env.reset(seed=42)

    # Reset LSTM state if applicable
    if hasattr(policy_white, 'reset_state'):
        policy_white.reset_state()
    if hasattr(policy_black, 'reset_state'):
        policy_black.reset_state()

    results = np.zeros(num_games, dtype=np.int32)  # 0=ongoing, 1=white, 2=draw, 3=black

    for step in range(max_steps):
        obs = torch.as_tensor(env.observations, dtype=torch.float32, device=device)

        white_obs = obs[0::2]
        black_obs = obs[1::2]

        with torch.no_grad():
            w_logits, _ = policy_white.forward_eval(white_obs)
            b_logits, _ = policy_black.forward_eval(black_obs)
            w_actions = torch.argmax(w_logits, dim=-1)
            b_actions = torch.argmax(b_logits, dim=-1)

        actions = np.zeros(num_games * 2, dtype=np.int32)
        actions[0::2] = w_actions.cpu().numpy()
        actions[1::2] = b_actions.cpu().numpy()

        env.step(actions)

        for g in range(num_games):
            if results[g] != 0:
                continue
            widx = 2 * g
            if env.terminals[widx] or env.terminals[widx + 1]:
                wr = env.rewards[widx]
                if wr > 0:
                    results[g] = 1
                elif wr < 0:
                    results[g] = 3
                else:
                    results[g] = 2

        if np.all(results != 0):
            break

    # Remaining games count as draws
    results[results == 0] = 2

    env.close()
    w = int(np.sum(results == 1))
    d = int(np.sum(results == 2))
    b = int(np.sum(results == 3))
    return w, d, b


def main():
    parser = argparse.ArgumentParser(description='Elo evaluation for chess checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--baseline', type=str, default='random',
                        help='Baseline: "random" or path to .pt file')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games per color match')
    parser.add_argument('--max-steps', type=int, default=512,
                        help='Max steps per game')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-blocks', type=int, default=2)
    parser.add_argument('--use-lstm', action='store_true',
                        help='Load full ChessLSTM (with LSTM weights) instead of base Policy')
    args = parser.parse_args()

    device = args.device
    num_agents = args.num_games * 2

    # Load policies
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        if args.use_lstm:
            print("  (with LSTM)")
            policy_a = load_lstm_policy(args.checkpoint, num_agents,
                                        args.hidden_size, args.num_blocks, device)
        else:
            policy_a = load_policy(args.checkpoint, args.hidden_size, args.num_blocks, device)
        name_a = os.path.basename(args.checkpoint)
    else:
        print("No checkpoint provided, using random policy as player A")
        policy_a = RandomPolicy()
        name_a = "Random"

    if args.baseline == 'random':
        policy_b = RandomPolicy()
        name_b = "Random"
    else:
        print(f"Loading baseline: {args.baseline}")
        if args.use_lstm:
            print("  (with LSTM)")
            policy_b = load_lstm_policy(args.baseline, num_agents,
                                        args.hidden_size, args.num_blocks, device)
        else:
            policy_b = load_policy(args.baseline, args.hidden_size, args.num_blocks, device)
        name_b = os.path.basename(args.baseline)

    print(f"\nMatch: {name_a} vs {name_b}")
    print(f"Games per side: {args.num_games}, max steps: {args.max_steps}")
    print()

    # Play both colors
    print(f"Playing {args.num_games} games: {name_a} (White) vs {name_b} (Black)...")
    aw_w, aw_d, aw_b = play_match(policy_a, policy_b, args.num_games, args.max_steps, device)
    print(f"  White({name_a}) wins: {aw_w}, Draws: {aw_d}, Black({name_b}) wins: {aw_b}")

    print(f"Playing {args.num_games} games: {name_b} (White) vs {name_a} (Black)...")
    bw_w, bw_d, bw_b = play_match(policy_b, policy_a, args.num_games, args.max_steps, device)
    print(f"  White({name_b}) wins: {bw_w}, Draws: {bw_d}, Black({name_a}) wins: {bw_b}")

    # Aggregate from player A's perspective
    total_games = args.num_games * 2
    a_wins = aw_w + bw_b    # A wins as white + A wins as black
    draws = aw_d + bw_d
    a_losses = aw_b + bw_w  # B wins as black + B wins as white

    # Win rate for A (draws count as 0.5)
    a_score = (a_wins + 0.5 * draws) / total_games

    # Elo estimation (baseline = 1000)
    baseline_elo = 1000
    a_elo = elo_from_winrate(a_score, baseline_elo)

    print()
    print("=" * 60)
    print(f"{'RESULTS':^60}")
    print("=" * 60)
    print(f"  {'Player':<20} {'W':>6} {'D':>6} {'L':>6} {'Score':>8} {'Elo':>8}")
    print(f"  {'-'*56}")
    print(f"  {name_a:<20} {a_wins:>6} {draws:>6} {a_losses:>6} "
          f"{a_score:>7.1%} {a_elo:>8.0f}")
    print(f"  {name_b:<20} {a_losses:>6} {draws:>6} {a_wins:>6} "
          f"{1 - a_score:>7.1%} {2 * baseline_elo - a_elo:>8.0f}")
    print("=" * 60)
    print(f"  Total games: {total_games}")
    print()


if __name__ == '__main__':
    main()
