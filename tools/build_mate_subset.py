#!/usr/bin/env python3
"""Build a mate-in-N subset from a FEN-only file using Stockfish.

Example:
  python3 tools/build_mate_subset.py \
    --input curriculum_train_checked.txt \
    --output curriculum_mate13.txt \
    --engine /home/alanga/.local/bin/stockfish \
    --depth 12 \
    --max-mate 3
"""

import argparse

import chess
import chess.engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--min-mate", type=int, default=1)
    parser.add_argument("--max-mate", type=int, default=3)
    args = parser.parse_args()

    kept = 0
    total = 0
    seen = set()

    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        with open(args.input, "r", encoding="utf-8") as fin, open(
            args.output, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                fen = line.strip()
                if not fen or fen in seen:
                    continue
                seen.add(fen)
                total += 1

                try:
                    board = chess.Board(fen)
                except Exception:
                    continue

                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                score = info.get("score")
                if score is None:
                    continue
                mate = score.pov(board.turn).mate()
                if mate is None:
                    continue
                if args.min_mate <= mate <= args.max_mate:
                    fout.write(fen + "\n")
                    kept += 1

    print(f"total={total} kept={kept} output={args.output}")


if __name__ == "__main__":
    main()

