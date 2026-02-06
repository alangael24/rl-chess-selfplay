"""PufferLib PufferEnv wrapper for Chess self-play.

Self-play design:
  - num_agents = num_envs * 2 (2 players per game: White and Black)
  - Each game has 2 agent slots: even index = White, odd index = Black
  - Both agents share the same policy network
  - The C binding pairs consecutive agent slots into shared games

Follows the patterns from PufferLib Ocean environments (Connect4, Go).
"""

import numpy as np
import gymnasium

import pufferlib
from csrc import binding

OBS_SIZE = 72   # 64 board squares + 8 metadata bytes
NUM_ACTIONS = 4096  # 64 * 64 (from_square * 64 + to_square)


class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=128, render_mode=None, report_interval=128,
                 max_steps=256, illegal_move_penalty=-0.1,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        # 2 players per game: agent 2*i = White, agent 2*i+1 = Black
        self.num_agents = num_envs * 2

        super().__init__(buf=buf)

        # The C binding creates num_agents Env structs.
        # Internally, consecutive pairs (2*i, 2*i+1) share the same ChessGame.
        # We pass num_agents (not num_envs) so env_binding.h creates the right
        # number of agent slots, each with its own obs/reward/terminal pointers.
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents,  # number of agent slots
            seed,
            max_steps=max_steps,
            illegal_move_penalty=illegal_move_penalty,
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
