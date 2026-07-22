from __future__ import annotations

import chess
import pytest

from worst_chess.chess.outcomes import DrawPolicy, adjudicate


def test_nonterminal_position_has_no_result_or_utility() -> None:
    result = adjudicate(chess.Board(), chess.WHITE, DrawPolicy.NEVER_CLAIM)

    assert not result.terminal
    assert result.winner is None
    assert result.termination is None
    assert result.utility is None
    assert not result.target_was_checkmated
    assert not result.target_won


def test_target_checkmate_is_positive_utility() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)

    result = adjudicate(board, chess.WHITE, DrawPolicy.NEVER_CLAIM)

    assert result.terminal
    assert result.winner == chess.BLACK
    assert result.termination is chess.Termination.CHECKMATE
    assert result.utility == 1.0
    assert result.target_was_checkmated
    assert not result.target_won


def test_target_checkmating_opponent_is_negative_utility() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)

    result = adjudicate(board, chess.BLACK, DrawPolicy.NEVER_CLAIM)

    assert result.terminal
    assert result.winner == chess.BLACK
    assert result.utility == -1.0
    assert not result.target_was_checkmated
    assert result.target_won


@pytest.mark.parametrize(
    ("fen", "termination"),
    [
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", chess.Termination.STALEMATE),
        (
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            chess.Termination.INSUFFICIENT_MATERIAL,
        ),
    ],
)
def test_automatic_position_draws(fen: str, termination: chess.Termination) -> None:
    for policy in DrawPolicy:
        result = adjudicate(chess.Board(fen), chess.WHITE, policy)
        assert result.terminal
        assert result.winner is None
        assert result.termination is termination
        assert result.utility == 0.0


def test_claimable_fifty_move_draw_depends_on_policy() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 100 51")

    never = adjudicate(board, chess.WHITE, DrawPolicy.NEVER_CLAIM)
    claim = adjudicate(board, chess.WHITE, DrawPolicy.CLAIM_AVAILABLE)

    assert not never.terminal
    assert claim.terminal
    assert claim.termination is chess.Termination.FIFTY_MOVES
    assert claim.utility == 0.0


def test_seventyfive_move_draw_is_automatic() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 150 76")

    for policy in DrawPolicy:
        result = adjudicate(board, chess.WHITE, policy)
        assert result.terminal
        assert result.termination is chess.Termination.SEVENTYFIVE_MOVES
        assert result.utility == 0.0


def test_checkmate_precedes_seventyfive_move_rule() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 150 76")

    result = adjudicate(board, chess.BLACK, DrawPolicy.NEVER_CLAIM)

    assert result.termination is chess.Termination.CHECKMATE
    assert result.utility == 1.0


def _repetition_board(cycles: int) -> chess.Board:
    board = chess.Board()
    for _ in range(cycles):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
    return board


def test_claimable_threefold_repetition_depends_on_policy() -> None:
    board = _repetition_board(2)

    never = adjudicate(board, chess.WHITE, DrawPolicy.NEVER_CLAIM)
    claim = adjudicate(board, chess.WHITE, DrawPolicy.CLAIM_AVAILABLE)

    assert not never.terminal
    assert claim.terminal
    assert claim.termination is chess.Termination.THREEFOLD_REPETITION


def test_fivefold_repetition_is_automatic() -> None:
    board = _repetition_board(4)

    result = adjudicate(board, chess.WHITE, DrawPolicy.NEVER_CLAIM)

    assert result.terminal
    assert result.termination is chess.Termination.FIVEFOLD_REPETITION
    assert result.utility == 0.0


def test_adjudication_does_not_mutate_fen_or_move_stack() -> None:
    board = _repetition_board(2)
    fen = board.fen()
    move_stack = tuple(board.move_stack)

    adjudicate(board, chess.BLACK, DrawPolicy.CLAIM_AVAILABLE)

    assert board.fen() == fen
    assert tuple(board.move_stack) == move_stack


def test_draw_policy_must_be_explicit_enum() -> None:
    with pytest.raises(TypeError, match="DrawPolicy"):
        adjudicate(chess.Board(), chess.WHITE, "never_claim")  # type: ignore[arg-type]

