#!/usr/bin/env python3
"""Benchmark fully C-side NNUE search stepping.

Usage:
  python tools/benchmark_qpolicy_c.py --qpolicy model.qpol --num-envs 512 --seconds 5
"""

import argparse
import os
import sys
import time

# Allow running from git clone root without installing as a package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chess_env import Chess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpolicy", required=True, help="Path to native NNUE .qpol file")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()

    env = Chess(num_envs=args.num_envs, max_steps=args.max_steps)
    env.reset(seed=123)
    env.load_qpolicy(args.qpolicy)

    ticks = 0
    start = time.time()
    while time.time() - start < args.seconds:
        env.step_qpolicy()
        ticks += 1

    elapsed = time.time() - start
    sps = env.num_agents * ticks / elapsed
    print(f"SPS={sps:,.0f}  envs={env.num_agents} ticks={ticks} elapsed={elapsed:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
