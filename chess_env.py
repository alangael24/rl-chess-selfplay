"""PufferLib PufferEnv wrapper for Chess self-play.

1-agent-per-game topology:
  - num_agents = num_envs (1 agent per game)
  - Each step, the agent controls whoever's turn it is (White or Black)
  - learner_color alternates each reset for symmetric self-play
  - Rewards are signed: positive when learner benefits, negative when opponent does

Observation:
  - Incremental HalfKP-like accumulator (256 float dims)
  - Phase one-hot (2 dims)
  - Learner-turn bit (1 dim)

Two-phase action system (97 actions):
  Phase 0: Pick a piece (action 0-63 = board square)
  Phase 1: Pick destination (0-63) or promotion (64-95)
  Action 96: PASS (legacy, never valid in 1-agent mode)

Follows the patterns from PufferLib Ocean environments (Connect4, Go).
"""

import numpy as np
import gymnasium

import pufferlib
from csrc import binding

ACCUM_SIZE = 256
OBS_META = 3
OBS_SIZE = ACCUM_SIZE + OBS_META
NUM_ACTIONS = 97  # 64 squares + 32 promotions + 1 pass


class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=128, render_mode=None, report_interval=128,
                 max_steps=256, illegal_move_penalty=-0.1,
                 reward_invalid_piece=-0.01, reward_invalid_move=-0.01,
                 reward_valid_piece=0.0, reward_valid_move=0.0,
                 reward_capture_bonus=0.0, reward_check_bonus=0.0,
                 reward_repetition=0.0, reward_material=0.0,
                 reward_position=0.0, reward_castling=0.0,
                 reward_draw=0.0, reward_see_hanging=0.0,
                 enable_50_move_rule=1,
                 enable_threefold_repetition=1,
                 fen_file=None, fen_curric_pct=0.0,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        # Required by newer PufferLib buffer setup path.
        self.selfplay = 0
        # 1 agent per game: agent controls whoever's turn it is
        self.num_agents = num_envs
        self.agents_per_batch = self.num_agents

        super().__init__(buf=buf)

        init_kwargs = dict(
            max_steps=max_steps,
            illegal_move_penalty=illegal_move_penalty,
            reward_invalid_piece=reward_invalid_piece,
            reward_invalid_move=reward_invalid_move,
            reward_valid_piece=reward_valid_piece,
            reward_valid_move=reward_valid_move,
            reward_capture_bonus=reward_capture_bonus,
            reward_check_bonus=reward_check_bonus,
            reward_repetition=reward_repetition,
            reward_material=reward_material,
            reward_position=reward_position,
            reward_castling=reward_castling,
            reward_draw=reward_draw,
            reward_see_hanging=reward_see_hanging,
            enable_50_move_rule=enable_50_move_rule,
            enable_threefold_repetition=enable_threefold_repetition,
            fen_curric_pct=fen_curric_pct,
            num_games=num_envs,
        )
        if fen_file is not None:
            init_kwargs['fen_file'] = fen_file

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents,  # number of agent slots
            seed,
            **init_kwargs,
        )
        self.fen_curric_pct = float(fen_curric_pct)
        self.fen_file = fen_file

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get('episode_length', 0) > 0:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)

    def load_fens(self, fen_file):
        """Replace current FEN curriculum list at runtime."""
        loaded = int(binding.vec_load_fens(self.c_envs, fen_file))
        self.fen_file = fen_file
        return loaded

    def set_fen_curric_pct(self, pct):
        """Update curriculum sampling probability at runtime (0.0 to 1.0)."""
        pct = float(max(0.0, min(1.0, pct)))
        binding.vec_set_fen_pct(self.c_envs, pct)
        self.fen_curric_pct = pct

    def load_qpolicy(self, qpolicy_path):
        """Load native NNUE int8/int16 weights for C-side search + inference."""
        return bool(binding.vec_load_qpolicy(self.c_envs, qpolicy_path))

    def infer_actions_qpolicy(self, out_values=False):
        """Infer actions using C-side NNUE value search."""
        actions = np.zeros(self.num_agents, dtype=np.int32)
        if out_values:
            values = np.zeros(self.num_agents, dtype=np.float32)
            binding.vec_infer_actions(self.c_envs, actions, values)
            return actions, values
        binding.vec_infer_actions(self.c_envs, actions)
        return actions

    def step_qpolicy(self):
        """Run NNUE search + step fully in C (no Python model forward)."""
        binding.vec_step_qpolicy(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get('episode_length', 0) > 0:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)


def test_performance(timeout=10, num_envs=512):
    import time
    env = Chess(num_envs=num_envs)
    env.reset()
    tick = 0
    num_agents = num_envs  # 1 agent per game
    atn_cache = 1024
    actions = np.random.randint(0, NUM_ACTIONS, (atn_cache, num_agents))

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    elapsed = time.time() - start
    sps = num_agents * tick / elapsed
    print(f'SPS: {sps:,.0f}  ({num_envs} games, {num_agents} agents, {tick} ticks in {elapsed:.1f}s)')


if __name__ == '__main__':
    test_performance()
