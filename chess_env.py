"""PufferLib PufferEnv wrapper for Chess self-play.

Self-play design:
  - num_agents = num_envs * 2 (2 players per game: White and Black)
  - Each game has 2 agent slots: even index = White, odd index = Black
  - Both agents share the same policy network
  - The C binding pairs consecutive agent slots into shared games

Two-phase action system (97 actions):
  Phase 0: Pick a piece (action 0-63 = board square)
  Phase 1: Pick destination (0-63) or promotion (64-95)
  Action 96: PASS (valid when it's NOT this player's turn)

Follows the patterns from PufferLib Ocean environments (Connect4, Go).
"""

import numpy as np
import gymnasium

import pufferlib
from csrc import binding

OBS_SIZE = 301    # 64 board + 2 side + 4 castling + 1 ep + 2 phase + 64 selected
                  # + 64 valid_pieces + 64 valid_dests + 32 valid_promos
                  # + 1 self_check + 1 opp_check + 1 rule50 + 1 pass_valid
NUM_ACTIONS = 97  # 64 squares + 32 promotions + 1 pass


class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=128, render_mode=None, report_interval=128,
                 max_steps=256, illegal_move_penalty=-0.1,
                 reward_invalid_piece=-0.01, reward_invalid_move=-0.01,
                 reward_valid_piece=0.0, reward_valid_move=0.0,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        # 2 players per game: agent 2*i = White, agent 2*i+1 = Black
        self.num_agents = num_envs * 2

        super().__init__(buf=buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents,  # number of agent slots
            seed,
            max_steps=max_steps,
            illegal_move_penalty=illegal_move_penalty,
            reward_invalid_piece=reward_invalid_piece,
            reward_invalid_move=reward_invalid_move,
            reward_valid_piece=reward_valid_piece,
            reward_valid_move=reward_valid_move,
            num_games=num_envs,
        )

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


def test_performance(timeout=10, num_envs=512):
    import time
    env = Chess(num_envs=num_envs)
    env.reset()
    tick = 0
    num_agents = num_envs * 2
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
