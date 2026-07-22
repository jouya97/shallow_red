from __future__ import annotations

import chess
import pytest

from worst_chess.objective.proof_search import (
    ProofSearchConfig,
    ProofStatus,
    prove_forced_selfmate,
    prove_forced_selfmate_after_move,
)

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w KQ - 7 6"


def test_proves_composed_selfmate_in_one_against_every_reply() -> None:
    board = chess.Board(SELF_MATE_IN_ONE)

    result = prove_forced_selfmate(
        board,
        chess.WHITE,
        ProofSearchConfig(max_plies=2, node_budget=10_000),
    )

    assert result.status is ProofStatus.PROVEN
    assert result.plies == 2
    assert result.first_move == chess.Move.from_uci("f3h4")
    assert tuple(move.uci() for move in result.principal_variation) == (
        "f3h4",
        "d8h4",
    )


def test_refutes_same_position_inside_too_short_a_horizon() -> None:
    result = prove_forced_selfmate(
        chess.Board(SELF_MATE_IN_ONE),
        chess.WHITE,
        ProofSearchConfig(max_plies=1, node_budget=10_000),
    )

    assert result.status is ProofStatus.REFUTED
    assert result.plies is None


def test_proves_a_designated_selfmating_first_move() -> None:
    board = chess.Board(SELF_MATE_IN_ONE)

    result = prove_forced_selfmate_after_move(
        board,
        chess.WHITE,
        chess.Move.from_uci("f3h4"),
        ProofSearchConfig(max_plies=2, node_budget=10_000),
    )

    assert result.status is ProofStatus.PROVEN
    assert result.plies == 2
    assert tuple(move.uci() for move in result.principal_variation) == (
        "f3h4",
        "d8h4",
    )


def test_refutes_a_designated_non_selfmating_first_move() -> None:
    board = chess.Board(SELF_MATE_IN_ONE)

    result = prove_forced_selfmate_after_move(
        board,
        chess.WHITE,
        chess.Move.from_uci("d5c7"),
        ProofSearchConfig(max_plies=2, node_budget=10_000),
    )

    assert result.status is ProofStatus.REFUTED


def test_reports_unknown_instead_of_false_when_budget_expires() -> None:
    result = prove_forced_selfmate(
        chess.Board(),
        chess.WHITE,
        ProofSearchConfig(max_plies=4, node_budget=1),
    )

    assert result.status is ProofStatus.UNKNOWN
    assert result.nodes == 1


def test_terminal_success_requires_the_designated_target_to_be_mated() -> None:
    white_mated = chess.Board(
        "rnb1kbnr/pppp1ppp/8/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )

    assert prove_forced_selfmate(white_mated, chess.WHITE).status is ProofStatus.PROVEN
    assert prove_forced_selfmate(white_mated, chess.BLACK).status is ProofStatus.REFUTED


def test_validates_limits_and_target_color() -> None:
    with pytest.raises(ValueError, match="max_plies"):
        ProofSearchConfig(max_plies=-1)
    with pytest.raises(ValueError, match="node_budget"):
        ProofSearchConfig(node_budget=0)
    with pytest.raises(TypeError, match="target_color"):
        prove_forced_selfmate(chess.Board(), 1)  # type: ignore[arg-type]
