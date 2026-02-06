"""Chess Self-Play Training with PufferLib.

Both White and Black agents share the same policy network.
The policy learns to play chess through self-play.

Two ways to train:
  1. PufferLib CLI (recommended):
     python install_cli.py  # one-time setup
     puffer train puffer_chess
     puffer train puffer_chess --train.learning-rate 0.001 --env.num-envs 4096

  2. Standalone:
     python train_chess.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS

BOARD_SIZE = 64
NUM_PIECE_TYPES = 13  # 0=empty, 1-6=white pieces, 7-12=black pieces


class ResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        return F.relu(self.ln2(self.fc2(F.relu(self.ln1(self.fc1(x))))) + x)


class Policy(nn.Module):
    def __init__(self, env, hidden_size=256, num_blocks=2):
        super().__init__()

        self.piece_embedding = nn.Embedding(NUM_PIECE_TYPES, 32)

        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        board_feat_size = BOARD_SIZE * 32
        self.input_fc = pufferlib.pytorch.layer_init(
            nn.Linear(board_feat_size + 64, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

    def forward_eval(self, x, state=None):
        batch_size = x.shape[0]
        board = x[:, :BOARD_SIZE].long()
        meta = x[:, BOARD_SIZE:BOARD_SIZE + 8].float()
        action_mask = x[:, BOARD_SIZE + 8:] > 0

        board = torch.clamp(board, 0, NUM_PIECE_TYPES - 1)
        board_emb = self.piece_embedding(board)
        board_flat = board_emb.reshape(batch_size, -1)

        meta_feat = self.meta_encoder(meta)

        combined = torch.cat([board_flat, meta_feat], dim=1)
        x = F.relu(self.input_ln(self.input_fc(combined)))

        for block in self.blocks:
            x = block(x)

        logits = self.actor(x)
        masked_logits = logits.masked_fill(~action_mask, -1e9)

        # Safety fallback for unexpected all-zero masks.
        all_masked = ~action_mask.any(dim=1)
        if all_masked.any():
            masked_logits[all_masked] = logits[all_masked]

        return masked_logits, self.critic(x)

    def forward(self, x, state=None):
        return self.forward_eval(x, state)


if __name__ == "__main__":
    print("=" * 60)
    print("CHESS SELF-PLAY TRAINING (standalone)")
    print("Tip: use 'puffer train puffer_chess' for CLI mode")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    args = pufferl.load_config('default')
    args['train']['env'] = 'chess'
    args['train']['total_timesteps'] = 1_000_000_000
    args['train']['torch_deterministic'] = False
    args['train']['precision'] = 'bfloat16'
    args['train']['learning_rate'] = 2.5e-4
    args['train']['ent_coef'] = 0.001
    args['train']['batch_size'] = 65536
    args['train']['minibatch_size'] = 16384
    args['train']['bptt_horizon'] = 16
    args['train']['update_epochs'] = 2
    args['train']['checkpoint_interval'] = 500
    args['train']['gamma'] = 0.99
    args['train']['gae_lambda'] = 0.95
    args['train']['clip_coef'] = 0.2

    NUM_GAMES = 2048
    NUM_AGENTS = NUM_GAMES * 2

    vecenv = pufferlib.vector.make(
        Chess,
        env_kwargs={
            'num_envs': NUM_GAMES,
            'max_steps': 256,
            'illegal_move_penalty': -0.1,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(vecenv, hidden_size=256, num_blocks=2).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Games: {NUM_GAMES}")
    print(f"  Agents: {NUM_AGENTS}")
    print("=" * 60)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'chess_selfplay.pt')
    trainer.close()
