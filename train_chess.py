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
import argparse

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


class Policy(nn.Module):
    """Lightweight fighter-style policy for higher throughput."""
    def __init__(self, env, hidden_size=256, cnn_channels=64, embed_dim=16, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # 13 board one-hot + selected + valid_pieces + valid_dests = 16 channels.
        spatial_in = NUM_PIECE_TYPES + 3
        self.spatial_cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(spatial_in, cnn_channels, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 8x8 -> 4x4 -> 2x2 => cnn_channels * 4.
        spatial_flat = cnn_channels * 4

        self.side_embed = nn.Embedding(2, embed_dim)
        self.castle_embed = nn.Embedding(16, embed_dim)
        self.ep_embed = nn.Embedding(65, embed_dim)  # 0-7 files + 64 as "none"
        self.phase_embed = nn.Embedding(2, embed_dim)

        # scalar: self_check + opp_check + rule50 + pass_valid + valid_promos(32) = 36
        self.scalar_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(36, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        total_features = spatial_flat + 4 * embed_dim + hidden_size
        self.fusion_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(total_features, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

        self._action_mask = None

    def encode_observations(self, x, state=None):
        batch_size = x.shape[0]

        board = x[:, OBS_BOARD:OBS_BOARD + 64].long()
        phase = x[:, OBS_PHASE:OBS_PHASE + 2].float() / 255.0
        selected = x[:, OBS_SELECTED:OBS_SELECTED + 64].float() / 255.0
        valid_pieces = x[:, OBS_VALID_PIECES:OBS_VALID_PIECES + 64].float() / 255.0
        valid_dests = x[:, OBS_VALID_DESTS:OBS_VALID_DESTS + 64].float() / 255.0
        valid_promos = x[:, OBS_VALID_PROMOS:OBS_VALID_PROMOS + 32].float() / 255.0
        self_check = x[:, OBS_SELF_CHECK:OBS_SELF_CHECK + 1].float() / 255.0
        opp_check = x[:, OBS_OPP_CHECK:OBS_OPP_CHECK + 1].float() / 255.0
        rule50 = x[:, OBS_RULE50:OBS_RULE50 + 1].float() / 255.0
        pass_valid = x[:, OBS_PASS_VALID:OBS_PASS_VALID + 1].float() / 255.0

        board = torch.clamp(board, 0, NUM_PIECE_TYPES - 1)
        board_oh = F.one_hot(board, num_classes=NUM_PIECE_TYPES).float()
        board_oh = board_oh.view(batch_size, 8, 8, NUM_PIECE_TYPES).permute(0, 3, 1, 2)

        # Spatial channels
        vp_plane = valid_pieces.view(batch_size, 1, 8, 8)
        vd_plane = valid_dests.view(batch_size, 1, 8, 8)
        sp_plane = selected.view(batch_size, 1, 8, 8)

        spatial = torch.cat([board_oh, sp_plane, vp_plane, vd_plane], dim=1)
        spatial_feat = self.spatial_cnn(spatial)

        side_idx = x[:, OBS_SIDE:OBS_SIDE + 2].argmax(dim=1).long()
        castling_bits = (x[:, OBS_CASTLING:OBS_CASTLING + 4] > 0).long()
        castling_idx = (
            castling_bits[:, 0]
            + 2 * castling_bits[:, 1]
            + 4 * castling_bits[:, 2]
            + 8 * castling_bits[:, 3]
        ).long()
        ep_raw = x[:, OBS_EP].long()
        ep_idx = torch.where(
            ep_raw == 255,
            torch.full_like(ep_raw, 64),
            torch.clamp(ep_raw, 0, 7),
        )
        phase_idx = x[:, OBS_PHASE:OBS_PHASE + 2].argmax(dim=1).long()

        side_feat = self.side_embed(side_idx)
        castle_feat = self.castle_embed(castling_idx)
        ep_feat = self.ep_embed(ep_idx)
        phase_feat = self.phase_embed(phase_idx)

        scalar_in = torch.cat([self_check, opp_check, rule50, pass_valid, valid_promos], dim=1)
        scalar_feat = self.scalar_encoder(scalar_in)

        combined = torch.cat([spatial_feat, side_feat, castle_feat, ep_feat, phase_feat, scalar_feat], dim=1)
        hidden = self.fusion_fc(combined)

        is_phase0 = phase[:, 0:1] > 0.5
        mask = torch.zeros(batch_size, NUM_ACTIONS, device=x.device, dtype=torch.bool)
        vp_bool = valid_pieces > 0.5
        vd_bool = valid_dests > 0.5
        mask[:, :64] = torch.where(is_phase0, vp_bool, vd_bool)
        promo_bool = valid_promos > 0.5
        mask[:, 64:96] = torch.where(is_phase0, torch.zeros_like(promo_bool), promo_bool)
        # PASS is legacy and should remain invalid in 1-agent topology.
        mask[:, 96] = False
        self._action_mask = mask

        return hidden

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        value = self.critic(hidden)

        if self._action_mask is not None:
            mask = self._action_mask
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


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def parse_curriculum_schedule(spec):
    """Parse schedule like: '0:0.9,50:0.6,100:0.3,150:0.1,200:0.0'."""
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        epoch_s, pct_s = item.split(':', 1)
        out.append((int(epoch_s.strip()), _clamp01(float(pct_s.strip()))))
    out.sort(key=lambda t: t[0])
    return out


def build_default_curriculum_schedule(start_pct):
    start_pct = _clamp01(start_pct)
    return [
        (0, start_pct),
        (50, min(start_pct, 0.60)),
        (100, min(start_pct, 0.30)),
        (150, min(start_pct, 0.10)),
        (200, 0.0),
    ]


class FenCurriculumController:
    def __init__(self, vecenv, schedule):
        self.vecenv = vecenv
        self.schedule = schedule
        self.last_pct = None

    def target_pct(self, epoch):
        if not self.schedule:
            return None
        pct = self.schedule[0][1]
        for at_epoch, at_pct in self.schedule:
            if epoch >= at_epoch:
                pct = at_pct
            else:
                break
        return pct

    def maybe_apply(self, epoch, force=False):
        pct = self.target_pct(epoch)
        if pct is None:
            return
        if (not force) and self.last_pct is not None and abs(self.last_pct - pct) < 1e-9:
            return
        self.vecenv.set_fen_curric_pct(pct)
        self.last_pct = pct
        print(f"[curriculum] epoch={epoch} fen_curric_pct={pct:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--num-games', type=int, default=1024)
    parser.add_argument('--fen-file', type=str, default=None,
                        help='Path to FEN curriculum file')
    parser.add_argument('--fen-curric-pct', type=float, default=0.0,
                        help='Initial probability of resetting from curriculum FENs')
    parser.add_argument('--fen-curric-schedule', type=str, default='',
                        help='Decay schedule epoch:pct pairs, e.g. 0:0.9,50:0.6,100:0.3,150:0.1,200:0.0')
    parser.add_argument('--no-curriculum-decay', action='store_true',
                        help='Keep fen_curric_pct fixed (ignore schedule)')
    cli_args = parser.parse_args()

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

    NUM_GAMES = cli_args.num_games
    NUM_AGENTS = NUM_GAMES  # 1 agent per game
    EVAL_EVERY = 10

    fen_curric_pct = _clamp01(cli_args.fen_curric_pct)

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
            'fen_curric_pct': fen_curric_pct,
            'fen_file': cli_args.fen_file,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )
    vecenv.agents_per_batch = NUM_AGENTS

    curriculum = None
    if cli_args.fen_file:
        schedule = []
        if not cli_args.no_curriculum_decay:
            schedule = parse_curriculum_schedule(cli_args.fen_curric_schedule)
            if not schedule:
                start_pct = fen_curric_pct if fen_curric_pct > 0.0 else 0.9
                schedule = build_default_curriculum_schedule(start_pct)
        elif fen_curric_pct > 0.0:
            schedule = [(0, fen_curric_pct)]

        if schedule:
            curriculum = FenCurriculumController(vecenv, schedule)
            curriculum.maybe_apply(epoch=0, force=True)
            print(f"  Curriculum FEN file: {cli_args.fen_file}")
            print(f"  Curriculum schedule: {schedule}")

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
            if curriculum is not None:
                curriculum.maybe_apply(trainer.epoch)
            if trainer.epoch == 0 or (trainer.epoch % EVAL_EVERY == 0):
                trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'chess_selfplay.pt')
    trainer.close()
