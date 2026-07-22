from __future__ import annotations

import chess
import pytest

from worst_chess.objective.four_piece import (
    CsrGraph,
    FourPieceState,
    canonical_four_piece_state,
    four_piece_board,
    pack_four_piece_state,
    project_kbvkr,
    unpack_four_piece_state,
)
from worst_chess.objective.retrograde import solve_forced_selfmate


def test_four_piece_canonicalization_preserves_d4_orbits() -> None:
    original = canonical_four_piece_state(
        target_king=chess.A1,
        target_bishop=chess.B3,
        opponent_king=chess.H8,
        opponent_rook=chess.G6,
        target_turn=True,
    )
    rotated = canonical_four_piece_state(
        target_king=chess.H8,
        target_bishop=chess.G6,
        opponent_king=chess.A1,
        opponent_rook=chess.B3,
        target_turn=True,
    )

    assert original == rotated


def test_four_piece_pack_round_trip_and_board_material() -> None:
    state = FourPieceState(chess.A1, chess.C1, chess.H8, chess.H1, False)

    assert unpack_four_piece_state(pack_four_piece_state(state)) == state
    board = four_piece_board(state)
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.C1) == chess.Piece(chess.BISHOP, chess.WHITE)
    assert board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.BLACK)


def test_csr_graph_works_with_existing_retrograde_solver() -> None:
    from array import array

    graph = CsrGraph(
        array("Q", (0, 0, 1, 3)),
        array("I", (0, 0, 1)),
    )

    solution = solve_forced_selfmate(graph, (True, True, False), {0})

    assert solution.forced_selfmate == (True, True, True)
    assert solution.plies == (0, 1, 2)


def test_projection_is_deterministic_and_ram_gated() -> None:
    first = project_kbvkr(sample_size=200, seed=17)
    second = project_kbvkr(sample_size=200, seed=17)

    assert first.legal_samples == second.legal_samples
    assert first.projected_legal_states == second.projected_legal_states
    assert first.projected_edges == second.projected_edges
    assert first.projected_peak_ram_bytes == second.projected_peak_ram_bytes
    assert 0 < first.legal_samples <= first.sample_size
    assert first.projected_legal_states_high_95 <= (
        first.raw_symmetry_reduced_states
    )


def test_projection_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="positive"):
        project_kbvkr(sample_size=0)
