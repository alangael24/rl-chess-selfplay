"""Chess Self-Play Training with PufferLib.

1-agent-per-game self-play: each step, the agent controls whoever's turn it is.
learner_color alternates each reset. Rewards are signed from learner's perspective.

Two-phase action system (97 actions):
  Phase 0: Pick a piece (action 0-63 = board square)
  Phase 1: Pick destination (0-63) or promotion (64-95)
  Action 96: PASS (legacy, never valid)

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
import pufferlib.models

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS

BOARD_SIZE = 64
NUM_PIECE_TYPES = 13  # 0=empty, 1-6=white pieces, 7-12=black pieces

# Observation layout offsets
OBS_BOARD = 0        # 64 bytes
OBS_SIDE = 64        # 2 bytes (one-hot: is_my_turn)
OBS_CASTLING = 66    # 4 bytes
OBS_EP = 70          # 1 byte
OBS_PHASE = 71       # 2 bytes (one-hot: phase 0 or 1)
OBS_SELECTED = 73    # 64 bytes (one-hot selected piece plane)
OBS_VALID_PIECES = 137  # 64 bytes
OBS_VALID_DESTS = 201   # 64 bytes
OBS_VALID_PROMOS = 265  # 32 bytes
OBS_SELF_CHECK = 297    # 1 byte
OBS_OPP_CHECK = 298     # 1 byte
OBS_RULE50 = 299        # 1 byte
OBS_PASS_VALID = 300    # 1 byte


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, c)
        self.gn2 = nn.GroupNorm(8, c)

    def forward(self, x):
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return F.relu(y + x)


class Policy(nn.Module):
    """Chess policy following PufferLib's encode/decode pattern.

    Implements encode_observations() and decode_actions() so it can be
    wrapped by pufferlib.models.LSTMWrapper for recurrence.
    Action masking data is stored during encode_observations and used
    in decode_actions.
    """
    def __init__(self, env, hidden_size=256, num_blocks=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # CNN input: 13 board channels + 3 spatial channels
        conv_in = NUM_PIECE_TYPES + 3
        conv_channels = 64
        self.board_stem = nn.Conv2d(
            conv_in, conv_channels, kernel_size=3, padding=1, bias=False)
        self.board_gn = nn.GroupNorm(8, conv_channels)
        self.board_blocks = nn.ModuleList([
            ResidualBlock(conv_channels) for _ in range(num_blocks)
        ])
        self.board_proj = pufferlib.pytorch.layer_init(
            nn.Linear(conv_channels * 8 * 8, hidden_size))

        # Scalar features: side(2) + castling(4) + ep(1) + phase(2) + check(2) + rule50(1) + pass_valid(1) + promos(32) = 45
        self.scalar_encoder = nn.Sequential(
            nn.Linear(45, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.fusion_fc = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size + 64, hidden_size))
        self.fusion_ln = nn.LayerNorm(hidden_size)

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

        # Action mask stored during encode_observations for use in decode_actions
        self._action_mask = None

    def encode_observations(self, x, state=None):
        batch_size = x.shape[0]

        # Parse observation
        board = x[:, OBS_BOARD:OBS_BOARD + 64].long()
        side = x[:, OBS_SIDE:OBS_SIDE + 2].float() / 255.0
        castling = x[:, OBS_CASTLING:OBS_CASTLING + 4].float() / 255.0
        ep = x[:, OBS_EP:OBS_EP + 1].float() / 255.0
        phase = x[:, OBS_PHASE:OBS_PHASE + 2].float() / 255.0
        selected = x[:, OBS_SELECTED:OBS_SELECTED + 64].float() / 255.0
        valid_pieces = x[:, OBS_VALID_PIECES:OBS_VALID_PIECES + 64].float() / 255.0
        valid_dests = x[:, OBS_VALID_DESTS:OBS_VALID_DESTS + 64].float() / 255.0
        valid_promos = x[:, OBS_VALID_PROMOS:OBS_VALID_PROMOS + 32].float() / 255.0
        self_check = x[:, OBS_SELF_CHECK:OBS_SELF_CHECK + 1].float() / 255.0
        opp_check = x[:, OBS_OPP_CHECK:OBS_OPP_CHECK + 1].float() / 255.0
        rule50 = x[:, OBS_RULE50:OBS_RULE50 + 1].float() / 255.0
        pass_valid = x[:, OBS_PASS_VALID:OBS_PASS_VALID + 1].float() / 255.0

        # Board one-hot -> 13 channels 8x8
        board = torch.clamp(board, 0, NUM_PIECE_TYPES - 1)
        board_oh = F.one_hot(board, num_classes=NUM_PIECE_TYPES).float()
        board_oh = board_oh.view(batch_size, 8, 8, NUM_PIECE_TYPES).permute(0, 3, 1, 2)

        # Spatial channels
        vp_plane = valid_pieces.view(batch_size, 1, 8, 8)
        vd_plane = valid_dests.view(batch_size, 1, 8, 8)
        sp_plane = selected.view(batch_size, 1, 8, 8)

        spatial = torch.cat([board_oh, vp_plane, vd_plane, sp_plane], dim=1)

        board_x = F.relu(self.board_gn(self.board_stem(spatial)))
        for block in self.board_blocks:
            board_x = block(board_x)

        board_feat = board_x.reshape(batch_size, -1)
        board_feat = F.relu(self.board_proj(board_feat))

        # Scalar features
        scalars = torch.cat([side, castling, ep, phase, self_check, opp_check, rule50, pass_valid, valid_promos], dim=1)
        scalar_feat = self.scalar_encoder(scalars)

        combined = torch.cat([board_feat, scalar_feat], dim=1)
        hidden = F.relu(self.fusion_ln(self.fusion_fc(combined)))

        # Build and store action mask for decode_actions
        is_phase0 = phase[:, 0:1] > 0.5
        mask = torch.zeros(batch_size, NUM_ACTIONS, device=x.device, dtype=torch.bool)
        vp_bool = valid_pieces > 0.5
        vd_bool = valid_dests > 0.5
        mask[:, :64] = torch.where(is_phase0, vp_bool, vd_bool)
        promo_bool = valid_promos > 0.5
        mask[:, 64:96] = torch.where(is_phase0, torch.zeros_like(promo_bool), promo_bool)
        pass_bool = pass_valid.squeeze(-1) > 0.5
        mask[:, 96] = pass_bool
        self._action_mask = mask

        return hidden

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        value = self.critic(hidden)

        # Apply action mask stored from encode_observations
        if self._action_mask is not None:
            mask = self._action_mask
            # Handle batch size mismatch (training reshapes B*T)
            if mask.shape[0] != logits.shape[0]:
                mask = None
            else:
                masked_logits = logits.masked_fill(~mask, -1e9)
                all_masked = ~mask.any(dim=1)
                if all_masked.any():
                    masked_logits[all_masked] = logits[all_masked]
                logits = masked_logits

        return logits, value

    def forward_eval(self, x, state=None):
        hidden = self.encode_observations(x, state=state)
        logits, value = self.decode_actions(hidden)
        return logits, value

    def forward(self, x, state=None):
        return self.forward_eval(x, state)


class ChessLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256):
        super().__init__(env, policy, input_size, hidden_size)


if __name__ == "__main__":
    print("=" * 60)
    print("CHESS SELF-PLAY TRAINING (standalone)")
    print("Two-phase action system: 97 actions")
    print("Tip: use 'puffer train puffer_chess' for CLI mode")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    from pufferlib import pufferl
    args = pufferl.load_config('default')
    args['train']['env'] = 'chess'
    args['train']['total_timesteps'] = 1_000_000_000
    args['train']['torch_deterministic'] = False
    args['train']['precision'] = 'bfloat16'
    args['train']['learning_rate'] = 1e-4
    args['train']['ent_coef'] = 0.005
    args['train']['batch_size'] = 524288
    args['train']['minibatch_size'] = 131072
    args['train']['bptt_horizon'] = 128
    args['train']['update_epochs'] = 1
    args['train']['checkpoint_interval'] = 500
    args['train']['gamma'] = 0.997
    args['train']['gae_lambda'] = 0.95
    args['train']['clip_coef'] = 0.15
    args['train']['max_grad_norm'] = 1.0
    args['train']['anneal_lr'] = True
    args['train']['use_rnn'] = True

    NUM_GAMES = 1024
    NUM_AGENTS = NUM_GAMES  # 1 agent per game
    EVAL_EVERY = 10

    vecenv = pufferlib.vector.make(
        Chess,
        env_kwargs={
            'num_envs': NUM_GAMES,
            'max_steps': 256,
            'illegal_move_penalty': -0.1,
            'reward_invalid_piece': -0.01,
            'reward_invalid_move': -0.01,
            'reward_valid_piece': 0.0,
            'reward_valid_move': 0.0,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )
    vecenv.agents_per_batch = NUM_AGENTS

    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    base_policy = Policy(vecenv, hidden_size=256, num_blocks=2)
    policy = ChessLSTM(vecenv, base_policy, input_size=256, hidden_size=256).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Games: {NUM_GAMES}")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  Eval every: {EVAL_EVERY} epochs")
    print("=" * 60)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    try:
        while trainer.epoch < trainer.total_epochs:
            if trainer.epoch == 0 or (trainer.epoch % EVAL_EVERY == 0):
                trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'chess_selfplay.pt')
    trainer.close()
