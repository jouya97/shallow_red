"""Run a deterministic held-out three-piece Syzygy guidance pilot.

Example:
    uv run python scripts/syzygy_pilot.py \
        --tablebase artifacts/tablebases/syzygy-3 --positions-per-class 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.agents.tablebase import SyzygyLosingAgent

_MATERIAL: tuple[tuple[str, chess.PieceType], ...] = (
    ("KQvK", chess.QUEEN),
    ("KRvK", chess.ROOK),
    ("KBvK", chess.BISHOP),
    ("KNvK", chess.KNIGHT),
    ("KPvK", chess.PAWN),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate exact standard-WDL guidance on held-out positions."
    )
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--positions-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.positions_per_class <= 0:
        raise ValueError("--positions-per-class must be positive")
    boards = _held_out_positions(
        arguments.positions_per_class,
        seed=arguments.seed,
    )
    fallback = HeuristicAgent()
    covered = 0
    improved = 0
    equal = 0
    worse = 0
    selected_wdl: Counter[int] = Counter()
    fallback_wdl: Counter[int] = Counter()
    material_coverage: Counter[str] = Counter()
    automatic_terminal: Counter[str] = Counter()
    dtz_complete = 0

    with SyzygyLosingAgent(arguments.tablebase, fallback) as agent:
        if not agent.tablebase_available:
            raise ValueError(f"no Syzygy tables found in {arguments.tablebase}")
        for material, board in boards:
            if board.is_game_over(claim_draw=False):
                automatic_terminal[material] += 1
                continue
            context = MoveContext(
                game_id=f"syzygy-pilot-{material}-{covered:06d}",
                ply=board.ply(),
                seed=arguments.seed,
                target_color=board.turn,
            )
            scores = agent.evaluate_moves(board, context)
            if scores is None:
                continue
            covered += 1
            material_coverage[material] += 1
            score_by_move = {score.move: score for score in scores}
            selected = agent.select_move(board, context)
            fallback_move = fallback.select_move(board.copy(stack=True), context)
            exact_wdl = score_by_move[selected].opponent_wdl
            baseline_wdl = score_by_move[fallback_move].opponent_wdl
            selected_wdl[exact_wdl] += 1
            fallback_wdl[baseline_wdl] += 1
            if exact_wdl > baseline_wdl:
                improved += 1
            elif exact_wdl == baseline_wdl:
                equal += 1
            else:
                worse += 1
            if all(score.opponent_dtz is not None for score in scores):
                dtz_complete += 1

    result = {
        "seed": arguments.seed,
        "positions_generated": len(boards),
        "positions_covered": covered,
        "material_coverage": dict(sorted(material_coverage.items())),
        "automatic_terminal_positions": dict(sorted(automatic_terminal.items())),
        "selected_opponent_wdl": _string_keys(selected_wdl),
        "fallback_opponent_wdl": _string_keys(fallback_wdl),
        "selected_wdl_vs_fallback": {
            "improved": improved,
            "equal": equal,
            "worse": worse,
        },
        "positions_with_complete_dtz": dtz_complete,
        "semantics": (
            "WDL is exact ordinary chess from the opponent side after the "
            "target move; positive values favor the target being checkmated."
        ),
        "limitation": (
            "This is not an inverted-game solution and assumes an opponent "
            "pursues its standard-chess win; DTZ is not mate distance."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _held_out_positions(
    positions_per_class: int,
    *,
    seed: int,
) -> tuple[tuple[str, chess.Board], ...]:
    positions: list[tuple[str, chess.Board]] = []
    for material_name, piece_type in _MATERIAL:
        seen: set[str] = set()
        attempt = 0
        while len(seen) < positions_per_class:
            if attempt >= positions_per_class * 10_000:
                raise RuntimeError(
                    f"could not generate {positions_per_class} "
                    f"{material_name} positions"
                )
            digest = hashlib.sha256(
                f"syzygy-pilot-v1\0{seed}\0{material_name}\0{attempt}".encode()
            ).digest()
            attempt += 1
            target_color = bool(digest[0] & 1)
            target_has_extra = bool(digest[0] & 2)
            squares = tuple(digest[index] % 64 for index in range(1, 4))
            if len(set(squares)) != 3:
                continue
            target_king, opponent_king, extra_square = squares
            if chess.square_distance(target_king, opponent_king) <= 1:
                continue
            if piece_type == chess.PAWN and chess.square_rank(extra_square) in (0, 7):
                continue
            board = chess.Board.empty()
            board.turn = target_color
            board.set_piece_at(target_king, chess.Piece(chess.KING, target_color))
            board.set_piece_at(
                opponent_king,
                chess.Piece(chess.KING, not target_color),
            )
            extra_color = target_color if target_has_extra else not target_color
            board.set_piece_at(extra_square, chess.Piece(piece_type, extra_color))
            board.clear_stack()
            if not board.is_valid():
                continue
            fen = board.fen(en_passant="fen")
            if fen in seen:
                continue
            seen.add(fen)
            positions.append((material_name, board))
    return tuple(positions)


def _string_keys(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


if __name__ == "__main__":
    raise SystemExit(main())
