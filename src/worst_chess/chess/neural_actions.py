"""Optional side-to-move action coordinates for neural policy heads.

The canonical dataset and rules action space remains absolute.  Neural models
may instead mirror Black's board and moves vertically, matching the existing
perspective-normalized observation.  White actions are unchanged.
"""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

from worst_chess.chess.actions import (
    decode_action,
    encode_move,
    legal_action_mask,
)

ABSOLUTE_ACTION_ORIENTATION = "absolute"
PERSPECTIVE_ACTION_ORIENTATION = "perspective_vertical_mirror"
ACTION_ORIENTATION_METADATA_KEY = "policy_action_orientation"


def validate_action_orientation(orientation: str) -> str:
    """Validate and return a supported neural action convention."""

    if orientation not in {
        ABSOLUTE_ACTION_ORIENTATION,
        PERSPECTIVE_ACTION_ORIENTATION,
    }:
        raise ValueError(f"unsupported neural action orientation: {orientation!r}")
    return orientation


def encode_neural_move(
    board: chess.Board,
    move: chess.Move,
    orientation: str = ABSOLUTE_ACTION_ORIENTATION,
) -> int:
    """Encode a legal move in the selected neural-policy coordinates."""

    validate_action_orientation(orientation)
    if orientation == ABSOLUTE_ACTION_ORIENTATION or board.turn == chess.WHITE:
        return encode_move(board, move)
    mirrored_board = board.mirror()
    return encode_move(mirrored_board, _mirror_move(move))


def decode_neural_action(
    board: chess.Board,
    action: int,
    orientation: str = ABSOLUTE_ACTION_ORIENTATION,
) -> chess.Move:
    """Decode a neural-policy action to a legal move on ``board``."""

    validate_action_orientation(orientation)
    if orientation == ABSOLUTE_ACTION_ORIENTATION or board.turn == chess.WHITE:
        return decode_action(board, action)
    mirrored_move = decode_action(board.mirror(), action)
    move = _mirror_move(mirrored_move)
    if move not in board.legal_moves:  # Defensive symmetry boundary.
        raise ValueError(f"mirrored neural action is not legal: {action}")
    return move


def neural_legal_action_mask(
    board: chess.Board,
    orientation: str = ABSOLUTE_ACTION_ORIENTATION,
) -> NDArray[np.bool_]:
    """Return the legal mask in the selected neural-policy coordinates."""

    validate_action_orientation(orientation)
    if orientation == ABSOLUTE_ACTION_ORIENTATION or board.turn == chess.WHITE:
        return legal_action_mask(board)
    return legal_action_mask(board.mirror())


def canonical_to_neural_action_map(
    board: chess.Board,
    orientation: str = ABSOLUTE_ACTION_ORIENTATION,
) -> dict[int, int]:
    """Map every canonical legal action to its neural-policy action."""

    validate_action_orientation(orientation)
    legal_moves = tuple(board.legal_moves)
    if orientation == ABSOLUTE_ACTION_ORIENTATION or board.turn == chess.WHITE:
        actions = (encode_move(board, move) for move in legal_moves)
        return {action: action for action in actions}
    mirrored_board = board.mirror()
    return {
        encode_move(board, move): encode_move(mirrored_board, _mirror_move(move))
        for move in legal_moves
    }


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


__all__ = [
    "ABSOLUTE_ACTION_ORIENTATION",
    "ACTION_ORIENTATION_METADATA_KEY",
    "PERSPECTIVE_ACTION_ORIENTATION",
    "canonical_to_neural_action_map",
    "decode_neural_action",
    "encode_neural_move",
    "neural_legal_action_mask",
    "validate_action_orientation",
]
