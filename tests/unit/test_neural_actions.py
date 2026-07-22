from __future__ import annotations

import chess
import pytest

from worst_chess.chess.actions import encode_move
from worst_chess.chess.neural_actions import (
    ABSOLUTE_ACTION_ORIENTATION,
    PERSPECTIVE_ACTION_ORIENTATION,
    decode_neural_action,
    encode_neural_move,
    neural_legal_action_mask,
)


def test_perspective_actions_align_mirrored_white_and_black_positions() -> None:
    white = chess.Board()
    black = white.mirror()
    white_move = chess.Move.from_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")

    white_action = encode_neural_move(
        white, white_move, PERSPECTIVE_ACTION_ORIENTATION
    )
    black_action = encode_neural_move(
        black, black_move, PERSPECTIVE_ACTION_ORIENTATION
    )

    assert white_action == black_action
    assert decode_neural_action(
        white, white_action, PERSPECTIVE_ACTION_ORIENTATION
    ) == white_move
    assert decode_neural_action(
        black, black_action, PERSPECTIVE_ACTION_ORIENTATION
    ) == black_move
    assert (
        neural_legal_action_mask(white, PERSPECTIVE_ACTION_ORIENTATION)
        == neural_legal_action_mask(black, PERSPECTIVE_ACTION_ORIENTATION)
    ).all()


@pytest.mark.parametrize(
    "board",
    [
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("8/5P2/8/8/8/8/2p5/4K2k b - - 0 1"),
        chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
    ],
)
def test_perspective_round_trip_covers_black_special_moves(
    board: chess.Board,
) -> None:
    for move in board.legal_moves:
        action = encode_neural_move(
            board, move, PERSPECTIVE_ACTION_ORIENTATION
        )
        assert (
            decode_neural_action(
                board, action, PERSPECTIVE_ACTION_ORIENTATION
            )
            == move
        )


def test_absolute_orientation_preserves_canonical_action_indices() -> None:
    board = chess.Board().mirror()
    move = chess.Move.from_uci("e7e5")

    assert encode_neural_move(board, move) == encode_move(board, move)
    assert decode_neural_action(
        board,
        encode_move(board, move),
        ABSOLUTE_ACTION_ORIENTATION,
    ) == move


def test_neural_actions_reject_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        neural_legal_action_mask(chess.Board(), "diagonal")
