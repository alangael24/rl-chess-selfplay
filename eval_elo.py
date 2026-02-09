"""Elo evaluation for feed-forward Chess self-play checkpoints.

This version matches the 1-agent-per-game topology:
- each environment step has one action slot
- the policy controlling the move is selected by learner_turn bit in observation
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_env import Chess, NUM_ACTIONS
from train_chess import OBS_LEARNER_TURN, Policy


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def elo_from_winrate(winrate, base_elo=1000):
    if winrate <= 0.0:
        return base_elo - 400
    if winrate >= 1.0:
        return base_elo + 400
    return base_elo - 400.0 * np.log10(1.0 / winrate - 1.0)


class RandomPolicy:
    def forward_eval(self, obs, state=None):
        batch = obs.shape[0]
        logits = torch.randn(batch, NUM_ACTIONS, device=obs.device)
        value = torch.zeros(batch, 1, device=obs.device)
        return logits, value


def _load_state_dict(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "policy_state_dict" in ckpt:
        return ckpt["policy_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_policy(checkpoint_path, hidden_size=256, device="cpu"):
    policy = Policy(env=None, hidden_size=hidden_size)
    raw_sd = _load_state_dict(checkpoint_path, device)

    cleaned = {}
    for k, v in raw_sd.items():
        if k.startswith("policy."):
            cleaned[k[len("policy."):]] = v
        else:
            cleaned[k] = v

    policy_keys = set(policy.state_dict().keys())
    filtered = {k: v for k, v in cleaned.items() if k in policy_keys}

    if not filtered:
        print("WARNING: No matching keys found in checkpoint for current Policy.")
        print(f"Checkpoint keys (first 10): {list(cleaned.keys())[:10]}")
    else:
        missing = policy_keys - set(filtered.keys())
        if missing:
            print(f"WARNING: {len(missing)} missing keys (strict=False).")
        policy.load_state_dict(filtered, strict=False)

    policy.to(device)
    policy.eval()
    return policy


def play_match(policy_learner, policy_opponent, num_games=100, max_steps=512, device="cpu"):
    """Run games where learner side uses policy_learner and opponent side policy_opponent.

    Returns: (learner_wins, draws, opponent_wins)
    """
    env = Chess(num_envs=num_games, max_steps=max_steps)
    env.reset(seed=42)

    results = np.zeros(num_games, dtype=np.int32)  # 0=ongoing, 1=learner, 2=draw, 3=opponent

    for _ in range(max_steps):
        obs = torch.as_tensor(env.observations, dtype=torch.float32, device=device)
        learner_turn = obs[:, OBS_LEARNER_TURN] > 0.5
        opponent_turn = ~learner_turn

        actions = torch.zeros(num_games, dtype=torch.int64, device=device)
        with torch.no_grad():
            if learner_turn.any():
                logits, _ = policy_learner.forward_eval(obs[learner_turn])
                actions[learner_turn] = torch.argmax(logits, dim=-1)
            if opponent_turn.any():
                logits, _ = policy_opponent.forward_eval(obs[opponent_turn])
                actions[opponent_turn] = torch.argmax(logits, dim=-1)

        env.step(actions.cpu().numpy().astype(np.int32))

        for g in range(num_games):
            if results[g] != 0:
                continue
            if env.terminals[g]:
                wr = env.rewards[g]
                if wr > 0:
                    results[g] = 1
                elif wr < 0:
                    results[g] = 3
                else:
                    results[g] = 2

        if np.all(results != 0):
            break

    results[results == 0] = 2
    env.close()

    lw = int(np.sum(results == 1))
    d = int(np.sum(results == 2))
    ow = int(np.sum(results == 3))
    return lw, d, ow


def main():
    parser = argparse.ArgumentParser(description="Elo evaluation for feed-forward chess checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--baseline", type=str, default="random", help='Baseline: "random" or path to .pt file')
    parser.add_argument("--num-games", type=int, default=100, help="Number of games per learner/opponent run")
    parser.add_argument("--max-steps", type=int, default=512, help="Max steps per game")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    args = parser.parse_args()

    device = args.device

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        policy_a = load_policy(args.checkpoint, args.hidden_size, device)
        name_a = os.path.basename(args.checkpoint)
    else:
        print("No checkpoint provided, using random policy as player A")
        policy_a = RandomPolicy()
        name_a = "Random"

    if args.baseline == "random":
        policy_b = RandomPolicy()
        name_b = "Random"
    else:
        print(f"Loading baseline: {args.baseline}")
        policy_b = load_policy(args.baseline, args.hidden_size, device)
        name_b = os.path.basename(args.baseline)

    print(f"\nMatch: {name_a} vs {name_b}")
    print(f"Games per run: {args.num_games}, max steps: {args.max_steps}\n")

    print(f"Run 1: learner={name_a}, opponent={name_b}")
    a_lw, a_d, a_ow = play_match(policy_a, policy_b, args.num_games, args.max_steps, device)
    print(f"  learner wins: {a_lw}, draws: {a_d}, opponent wins: {a_ow}")

    print(f"Run 2: learner={name_b}, opponent={name_a}")
    b_lw, b_d, b_ow = play_match(policy_b, policy_a, args.num_games, args.max_steps, device)
    print(f"  learner wins: {b_lw}, draws: {b_d}, opponent wins: {b_ow}")

    total_games = args.num_games * 2
    a_wins = a_lw + b_ow
    draws = a_d + b_d
    a_losses = a_ow + b_lw
    a_score = (a_wins + 0.5 * draws) / total_games

    baseline_elo = 1000
    a_elo = elo_from_winrate(a_score, baseline_elo)

    print()
    print("=" * 60)
    print(f"{'RESULTS':^60}")
    print("=" * 60)
    print(f"  {'Player':<20} {'W':>6} {'D':>6} {'L':>6} {'Score':>8} {'Elo':>8}")
    print(f"  {'-' * 56}")
    print(f"  {name_a:<20} {a_wins:>6} {draws:>6} {a_losses:>6} {a_score:>7.1%} {a_elo:>8.0f}")
    print(f"  {name_b:<20} {a_losses:>6} {draws:>6} {a_wins:>6} {1 - a_score:>7.1%} {2 * baseline_elo - a_elo:>8.0f}")
    print("=" * 60)
    print(f"  Total games: {total_games}")
    print()


if __name__ == "__main__":
    main()

