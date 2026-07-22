from __future__ import annotations

import chess
import pytest

from worst_chess.objective.three_piece import (
    canonical_pawn_placement,
    canonical_placement,
    solve_three_piece_class,
    state_board,
)


def test_canonical_placement_identifies_board_symmetries() -> None:
    placement = (chess.A1, chess.H8, chess.B1)
    rotation_180 = (chess.H8, chess.A1, chess.G8)

    assert canonical_placement(*placement) == canonical_placement(*rotation_180)


def test_pawn_canonicalization_uses_file_reflection_not_rank_reflection() -> None:
    placement = (chess.A2, chess.H7, chess.B3)
    file_reflection = (chess.H2, chess.A7, chess.G3)
    rank_reflection = (chess.A7, chess.H2, chess.B6)

    assert canonical_pawn_placement(*placement) == canonical_pawn_placement(
        *file_reflection
    )
    assert canonical_pawn_placement(*placement) != canonical_pawn_placement(
        *rank_reflection
    )


def test_kbv_k_has_no_forced_selfmate_for_either_owner() -> None:
    for extra_is_target in (True, False):
        result = solve_three_piece_class(
            chess.BISHOP,
            extra_is_target=extra_is_target,
        )

        assert result.states
        assert result.successful_terminals == 0
        assert result.solution.forced_count == 0
        assert result.forced_nonterminal_count == 0
        assert all(
            state_board(
                state,
                piece_type=chess.BISHOP,
                extra_is_target=extra_is_target,
            ).is_valid()
            for state in result.states
        )


def test_pawn_class_requires_promotion_closed_graph() -> None:
    with pytest.raises(ValueError, match="promotion-closure"):
        solve_three_piece_class(chess.PAWN, extra_is_target=False)
