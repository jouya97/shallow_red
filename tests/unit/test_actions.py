from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from worst_chess.chess.actions import (
    ACTION_PLANES,
    ACTION_SPACE_SIZE,
    ActionEncodingError,
    decode_action,
    encode_move,
    legal_action_mask,
)


def assert_all_legal_moves_round_trip(board: chess.Board) -> None:
    original_fen = board.fen()
    original_stack = tuple(board.move_stack)
    moves = list(board.legal_moves)
    actions = [encode_move(board, move) for move in moves]

    assert len(actions) == len(set(actions))
    assert all(
        decode_action(board, action) == move
        for action, move in zip(actions, moves, strict=True)
    )
    assert board.fen() == original_fen
    assert tuple(board.move_stack) == original_stack


def test_action_space_has_4672_fixed_actions() -> None:
    assert ACTION_PLANES == 73
    assert ACTION_SPACE_SIZE == 4672


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ("e1g1", "e1c1"),
        ),
        ("8/8/8/3pP3/8/8/8/K6k w - d6 0 1", ("e5d6",)),
        (
            "r1r4k/1P6/8/8/8/8/8/7K w - - 0 1",
            tuple(
                f"b7{destination}{piece}"
                for destination in ("a8", "b8", "c8")
                for piece in ("q", "r", "b", "n")
            ),
        ),
        (
            "7k/8/8/8/8/8/1p6/R1R4K b - - 0 1",
            tuple(
                f"b2{destination}{piece}"
                for destination in ("a1", "b1", "c1")
                for piece in ("q", "r", "b", "n")
            ),
        ),
    ],
)
def test_special_legal_moves_round_trip(fen: str, moves: tuple[str, ...]) -> None:
    board = chess.Board(fen)
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, uci
        assert decode_action(board, encode_move(board, move)) == move


def test_all_legal_moves_round_trip_across_random_games() -> None:
    random_source = random.Random(20260721)
    board = chess.Board()
    for _ in range(600):
        assert_all_legal_moves_round_trip(board)
        if board.is_game_over():
            board.reset()
        else:
            board.push(random_source.choice(list(board.legal_moves)))


def test_legal_action_mask_is_boolean_and_exact() -> None:
    board = chess.Board()
    mask = legal_action_mask(board)

    assert mask.shape == (4672,)
    assert mask.dtype == np.bool_
    assert np.count_nonzero(mask) == 20
    assert set(np.flatnonzero(mask)) == {
        encode_move(board, move) for move in board.legal_moves
    }
    for action in np.flatnonzero(mask):
        assert decode_action(board, action) in board.legal_moves


@pytest.mark.parametrize("action", [-1, 4672, 100_000, 1.25, True])
def test_invalid_action_is_rejected(action: object) -> None:
    with pytest.raises(ActionEncodingError):
        decode_action(chess.Board(), action)  # type: ignore[arg-type]


def test_off_board_or_illegal_action_is_rejected() -> None:
    # a1 north-west is off the board.
    off_board_action = chess.A1 * ACTION_PLANES + 7 * 7
    with pytest.raises(ActionEncodingError, match="off board"):
        decode_action(chess.Board(), off_board_action)

    # a1 north is on the board but there is no piece on a1 initially.
    with pytest.raises(ActionEncodingError, match="legal move"):
        decode_action(chess.Board(), chess.A1 * ACTION_PLANES)


def test_illegal_move_is_rejected_without_mutating_board() -> None:
    board = chess.Board()
    fen = board.fen()
    with pytest.raises(ActionEncodingError, match="not legal"):
        encode_move(board, chess.Move.from_uci("e2e5"))
    assert board.fen() == fen
