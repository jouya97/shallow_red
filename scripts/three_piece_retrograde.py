"""Exhaustively solve a symmetry-reduced non-pawn three-piece class.

Examples:
    uv run python scripts/three_piece_retrograde.py KQvK --extra-owner opponent
    uv run python scripts/three_piece_retrograde.py KBvK --extra-owner both
"""

from __future__ import annotations

import argparse
import json

import chess

from worst_chess.objective.three_piece import (
    solve_pawn_three_piece_class,
    solve_three_piece_class,
)

_MATERIAL: dict[str, chess.PieceType] = {
    "KQvK": chess.QUEEN,
    "KRvK": chess.ROOK,
    "KBvK": chess.BISHOP,
    "KNvK": chess.KNIGHT,
    "KPvK": chess.PAWN,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve exact history-free forced selfmate for a non-pawn "
            "three-piece material class. White is always the selfmate target."
        )
    )
    parser.add_argument("material", choices=tuple(_MATERIAL))
    parser.add_argument(
        "--extra-owner",
        choices=("target", "opponent", "both"),
        default="both",
        help="side that owns the non-king piece (default: both, separately)",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    ownership = (
        (True, False)
        if arguments.extra_owner == "both"
        else (arguments.extra_owner == "target",)
    )
    classes = []
    for extra_is_target in ownership:
        piece_type = _MATERIAL[arguments.material]
        result = (
            solve_pawn_three_piece_class(extra_is_target=extra_is_target)
            if piece_type == chess.PAWN
            else solve_three_piece_class(
                piece_type,
                extra_is_target=extra_is_target,
            )
        )
        finite_plies = tuple(
            plies for plies in result.solution.plies if plies is not None
        )
        classes.append(
            {
                "extra_owner": "target" if extra_is_target else "opponent",
                "legal_symmetry_reduced_states": len(result.states),
                "target_checkmate_terminals": result.successful_terminals,
                "forced_selfmate_states_including_terminals": (
                    result.solution.forced_count
                ),
                "forced_selfmate_nonterminal_states": (
                    result.forced_nonterminal_count
                ),
                "maximum_forced_selfmate_plies": (
                    max(finite_plies) if finite_plies else None
                ),
            }
        )
    output = {
        "material": arguments.material,
        "target": "white",
        "classes": classes,
        "optimization": (
            "target minimizes exact plies to its checkmate; resisting opponent "
            "avoids it when possible and otherwise maximizes exact plies"
        ),
        "draw_scope": {
            "stalemate": "terminal non-success",
            "dead_position": "terminal non-success",
            "cycles": "non-forced draw/failure",
            "omitted": [
                "50-move optional claim",
                "75-move automatic draw",
                "threefold optional claim",
                "fivefold automatic repetition",
            ],
        },
        "promotion_scope": (
            "KPvK promotions link to independently exhaustive KQ/KR/KB/KN-v-K "
            "solutions; non-pawn classes are closed under capture to bare kings."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
