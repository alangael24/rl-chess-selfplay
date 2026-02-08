#!/usr/bin/env python3
"""Validate and balance curriculum FENs for chess training.

Usage:
  python tools/validate_curriculum_fens.py \
    --input curriculum_raw.txt \
    --output curriculum_mates.txt \
    --engine /usr/games/stockfish \
    --depth 12 \
    --min-cp 500 \
    --max-mate 5

If --engine is omitted, the script only checks FEN legality and balancing.
"""

import argparse
from dataclasses import dataclass

chess = None


@dataclass
class FenEval:
    fen: str
    side_to_move: bool  # True=White, False=Black
    accepted: bool
    reason: str


def read_fens(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def is_legal_fen(fen):
    try:
        board = chess.Board(fen)
    except Exception:
        return False, None
    # Board constructor already validates core FEN semantics.
    return True, board


def is_strongly_winning(board, info, min_cp, max_mate):
    score = info.get("score", None)
    if score is None:
        return False, "no_score"

    pov = score.pov(board.turn)
    mate = pov.mate()
    if mate is not None:
        if mate > 0 and mate <= max_mate:
            return True, f"mate_in_{mate}"
        return False, f"mate={mate}"

    cp = pov.score(mate_score=100000)
    if cp is None:
        return False, "cp_none"
    if cp >= min_cp:
        return True, f"cp={cp}"
    return False, f"cp={cp}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", default=None,
                        help="Path to UCI engine (e.g., stockfish). Optional.")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--min-cp", type=int, default=500)
    parser.add_argument("--max-mate", type=int, default=5)
    parser.add_argument("--no-balance", action="store_true",
                        help="Keep all accepted positions, do not balance by side-to-move.")
    args = parser.parse_args()

    global chess
    try:
        import chess as _chess
        import chess.engine  # noqa: F401
        chess = _chess
    except Exception as exc:
        raise RuntimeError(
            "python-chess is required for this tool. Install with: pip install python-chess"
        ) from exc

    raw = read_fens(args.input)
    accepted = []
    rejected = []

    engine = None
    if args.engine:
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)

    try:
        for fen in raw:
            ok, board = is_legal_fen(fen)
            if not ok:
                rejected.append(FenEval(fen, True, False, "illegal_fen"))
                continue

            if engine is None:
                accepted.append(FenEval(fen, board.turn, True, "legal_only"))
                continue

            info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
            good, reason = is_strongly_winning(board, info, args.min_cp, args.max_mate)
            if good:
                accepted.append(FenEval(fen, board.turn, True, reason))
            else:
                rejected.append(FenEval(fen, board.turn, False, reason))
    finally:
        if engine is not None:
            engine.quit()

    if not args.no_balance:
        whites = [x for x in accepted if x.side_to_move]
        blacks = [x for x in accepted if not x.side_to_move]
        keep = min(len(whites), len(blacks))
        balanced = whites[:keep] + blacks[:keep]
    else:
        balanced = accepted

    with open(args.output, "w", encoding="utf-8") as f:
        for row in balanced:
            f.write(row.fen + "\n")

    print(f"input={len(raw)} accepted={len(accepted)} balanced={len(balanced)} rejected={len(rejected)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
