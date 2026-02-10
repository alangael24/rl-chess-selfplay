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
import sys
import argparse
import os

import pufferlib
import pufferlib.pytorch

from chess_env import Chess, OBS_SIZE, NUM_ACTIONS

ACCUM_SIZE = 256
OBS_PHASE = ACCUM_SIZE
OBS_LEARNER_TURN = ACCUM_SIZE + 2


class Policy(nn.Module):
    """Feed-forward MLP policy over incremental accumulator observations."""
    def __init__(self, env, hidden_size=256, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        in_features = ACCUM_SIZE + 3  # accumulator + phase one-hot + learner_turn
        # NNUE-style feed-forward stack (no recurrence): 4 dense layers.
        self.backbone = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(in_features, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

        self._phase0 = None

    def encode_observations(self, x, state=None):
        accum = x[:, :ACCUM_SIZE].float()
        phase = x[:, OBS_PHASE:OBS_PHASE + 2].float()
        learner_turn = x[:, OBS_LEARNER_TURN:OBS_LEARNER_TURN + 1].float()
        model_in = torch.cat([accum, phase, learner_turn], dim=1)
        hidden = self.backbone(model_in)
        self._phase0 = phase[:, 0:1] > 0.5
        return hidden

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        value = self.critic(hidden)

        if self._phase0 is not None and self._phase0.shape[0] == logits.shape[0]:
            # Keep PASS invalid (legacy action never valid in 1-agent mode).
            logits = logits.clone()
            logits[:, 96] = -1e9
            # Promotions are impossible in phase 0.
            promo_logits = logits[:, 64:96]
            phase0 = self._phase0.expand_as(promo_logits)
            logits[:, 64:96] = torch.where(
                phase0,
                torch.full_like(promo_logits, -1e9),
                promo_logits,
            )

        return logits, value

    def forward_eval(self, x, state=None):
        hidden = self.encode_observations(x, state=state)
        logits, value = self.decode_actions(hidden)
        return logits, value

    def forward(self, x, state=None):
        return self.forward_eval(x, state)


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def resolve_fen_file(cli_fen_file):
    """Pick an explicit FEN file or a local default curriculum file."""
    if cli_fen_file:
        if os.path.exists(cli_fen_file):
            return cli_fen_file
        raise FileNotFoundError(f"FEN curriculum file not found: {cli_fen_file}")

    default_candidates = (
        "curriculum_mixed.txt",
        "curriculum_mate13.txt",
        "curriculum_mates.txt",
        "curriculum_train_checked.txt",
    )
    for candidate in default_candidates:
        if os.path.exists(candidate):
            return candidate
    return None


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
                        help='Path to FEN curriculum file. If omitted, auto-picks a local curriculum_*.txt when present')
    parser.add_argument('--fen-curric-pct', type=float, default=-1.0,
                        help='Initial probability of resetting from curriculum FENs. Default: 0.9 if a FEN file is active, else 0.0')
    parser.add_argument('--fen-curric-schedule', type=str, default='',
                        help='Decay schedule epoch:pct pairs, e.g. 0:0.9,50:0.6,100:0.3,150:0.1,200:0.0')
    parser.add_argument('--no-curriculum-decay', action='store_true',
                        help='Keep fen_curric_pct fixed (ignore schedule)')
    parser.add_argument('--reward-draw', type=float, default=-0.03,
                        help='Reward for draw outcomes (repetition/stalemate/50-move/insufficient)')
    parser.add_argument('--reward-truncation', type=float, default=-0.10,
                        help='Reward when max_steps truncation is reached')
    parser.add_argument('--reward-repetition', type=float, default=-0.01,
                        help='Per-move penalty on repeated positions (occurrence >= 2)')
    parser.add_argument('--reward-material', type=float, default=0.005,
                        help='Material delta shaping scale')
    parser.add_argument('--reward-position', type=float, default=0.002,
                        help='Positional delta shaping scale (PST-based)')
    parser.add_argument('--reward-capture-bonus', type=float, default=0.002,
                        help='Extra reward bonus for captures')
    parser.add_argument('--reward-check-bonus', type=float, default=0.001,
                        help='Extra reward bonus when giving check')
    parser.add_argument('--reward-castling', type=float, default=0.005,
                        help='One-time castling bonus')
    parser.add_argument('--reward-see-hanging', type=float, default=-0.005,
                        help='Penalty scale for hanging-move SEE scores (must be <= 0)')
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
    args['train']['use_rnn'] = False

    NUM_GAMES = cli_args.num_games
    NUM_AGENTS = NUM_GAMES  # 1 agent per game
    EVAL_EVERY = 10

    fen_file = resolve_fen_file(cli_args.fen_file)
    if cli_args.fen_curric_pct < 0.0:
        fen_curric_pct = 0.9 if fen_file else 0.0
    else:
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
            'reward_capture_bonus': cli_args.reward_capture_bonus,
            'reward_check_bonus': cli_args.reward_check_bonus,
            'reward_repetition': cli_args.reward_repetition,
            'reward_material': cli_args.reward_material,
            'reward_position': cli_args.reward_position,
            'reward_castling': cli_args.reward_castling,
            'reward_draw': cli_args.reward_draw,
            'reward_truncation': cli_args.reward_truncation,
            'reward_see_hanging': cli_args.reward_see_hanging,
            'fen_curric_pct': fen_curric_pct,
            'fen_file': fen_file,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )
    vecenv.agents_per_batch = NUM_AGENTS

    curriculum = None
    if fen_file:
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
            print(f"  Curriculum FEN file: {fen_file}")
            print(f"  Curriculum schedule: {schedule}")

    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(vecenv, hidden_size=256).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Games: {NUM_GAMES}")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  Eval every: {EVAL_EVERY} epochs")
    print(f"  reward_draw={cli_args.reward_draw} reward_truncation={cli_args.reward_truncation} reward_repetition={cli_args.reward_repetition}")
    print(f"  shaping: material={cli_args.reward_material} position={cli_args.reward_position} capture={cli_args.reward_capture_bonus} check={cli_args.reward_check_bonus} castling={cli_args.reward_castling} see={cli_args.reward_see_hanging}")
    if fen_file:
        print(f"  Curriculum active: fen_file={fen_file} fen_curric_pct={fen_curric_pct:.3f}")
    else:
        print("  Curriculum active: no (no FEN file found/provided)")
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
