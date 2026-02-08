#!/usr/bin/env python3
"""Build mixed curriculum:
50% mate in 1, 30% mate in 2, 20% rook/queen endgames (no immediate mate).

Input is a FEN-only file.
"""

import argparse
import random

import chess
import chess.engine


def is_rq_endgame(board):
    """Kings + rooks/queens/pawns only, capped material count."""
    non_king = 0
    rq = 0
    for p in board.piece_map().values():
        if p.piece_type == chess.KING:
            continue
        non_king += 1
        if p.piece_type in (chess.ROOK, chess.QUEEN):
            rq += 1
        elif p.piece_type == chess.PAWN:
            pass
        else:
            return False
    return rq >= 1 and non_king <= 6


def classify(fen, engine, depth):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info.get("score")
    if score is None:
        return None

    pov = score.pov(board.turn)
    mate = pov.mate()
    cp = pov.score(mate_score=100000)

    if mate == 1:
        return "mate1"
    if mate == 2:
        return "mate2"

    # Rook/queen endgame bucket: no immediate mate, still clearly favorable.
    if is_rq_endgame(board):
        if mate is not None:
            if mate > 2:
                return "rq_end"
            return None
        if cp is not None and cp >= 300:
            return "rq_end"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="FEN-only source file")
    parser.add_argument("--output", required=True, help="Mixed curriculum output")
    parser.add_argument("--engine", required=True, help="Stockfish path")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    buckets = {"mate1": [], "mate2": [], "rq_end": []}

    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        for line in open(args.input, "r", encoding="utf-8"):
            fen = line.strip()
            if not fen:
                continue
            try:
                kind = classify(fen, engine, args.depth)
            except Exception:
                continue
            if kind in buckets:
                buckets[kind].append(fen)

    # Exact 5:3:2 ratio without duplicates.
    units = min(len(buckets["mate1"]) // 5, len(buckets["mate2"]) // 3, len(buckets["rq_end"]) // 2)
    if units <= 0:
        raise RuntimeError(
            f"Insufficient candidates for 50/30/20 ratio: "
            f"mate1={len(buckets['mate1'])}, mate2={len(buckets['mate2'])}, rq_end={len(buckets['rq_end'])}"
        )

    n1, n2, n3 = 5 * units, 3 * units, 2 * units

    out = []
    out += random.sample(buckets["mate1"], n1)
    out += random.sample(buckets["mate2"], n2)
    out += random.sample(buckets["rq_end"], n3)
    random.shuffle(out)

    with open(args.output, "w", encoding="utf-8") as f:
        for fen in out:
            f.write(fen + "\n")

    print(
        f"saved={len(out)} "
        f"mate1={n1} mate2={n2} rq_end={n3} "
        f"(candidates: mate1={len(buckets['mate1'])}, mate2={len(buckets['mate2'])}, rq_end={len(buckets['rq_end'])})"
    )


if __name__ == "__main__":
    main()

