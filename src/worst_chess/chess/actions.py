"""Fixed AlphaZero-style action encoding for orthodox chess.

The action space has one set of 73 move planes for each of the 64 origin
squares.  Plane ordering is fixed and independent of the side to move:

* 0..55: N, NE, E, SE, S, SW, W, NW sliding moves, one to seven squares;
* 56..63: NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW knight moves;
* 64..72: left, straight, right underpromotions (relative to the moving
  pawn), with knight, bishop, and rook as the inner ordering.

Queen promotions use their ordinary one-square sliding plane.  Castling and
en-passant likewise need no special planes.  Public conversion functions only
accept legal moves/actions, which makes a successfully decoded action safe to
pass to :meth:`chess.Board.push`.
"""

from __future__ import annotations

from numbers import Integral

import chess
import numpy as np
from numpy.typing import NDArray

ACTION_PLANES = 73
ACTION_SPACE_SIZE = len(chess.SQUARES) * ACTION_PLANES
"""Number of actions in the fixed policy head (64 * 73 = 4,672)."""

# Common aliases used by model/environment code.
NUM_ACTION_PLANES = ACTION_PLANES
NUM_ACTIONS = ACTION_SPACE_SIZE

_SLIDING_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # north
    (1, 1),  # north-east
    (1, 0),  # east
    (1, -1),  # south-east
    (0, -1),  # south
    (-1, -1),  # south-west
    (-1, 0),  # west
    (-1, 1),  # north-west
)
_KNIGHT_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_UNDERPROMOTION_PIECES: tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)


class ActionEncodingError(ValueError):
    """Raised when a move or action cannot represent a legal chess move."""


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal move without mutating ``board``.

    Args:
        board: Orthodox chess position in which ``move`` is to be played.
        move: A move contained in ``board.legal_moves``.

    Raises:
        ActionEncodingError: If the move is illegal or outside this action
            representation.
    """

    if move not in board.legal_moves:
        raise ActionEncodingError(f"move is not legal in the position: {move.uci()}")

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    if move.promotion in _UNDERPROMOTION_PIECES:
        forward = 1 if board.turn == chess.WHITE else -1
        if rank_delta != forward or file_delta not in (-1, 0, 1):
            raise ActionEncodingError(f"invalid underpromotion geometry: {move.uci()}")
        relative_file_delta = file_delta if board.turn == chess.WHITE else -file_delta
        direction_index = relative_file_delta + 1
        piece_index = _UNDERPROMOTION_PIECES.index(move.promotion)
        plane = 64 + direction_index * 3 + piece_index
        return move.from_square * ACTION_PLANES + plane

    try:
        knight_index = _KNIGHT_DIRECTIONS.index((file_delta, rank_delta))
    except ValueError:
        knight_index = -1
    if knight_index >= 0:
        plane = 56 + knight_index
        return move.from_square * ACTION_PLANES + plane

    distance = max(abs(file_delta), abs(rank_delta))
    if distance < 1 or distance > 7:
        raise ActionEncodingError(f"invalid sliding move geometry: {move.uci()}")
    if not (
        file_delta == 0
        or rank_delta == 0
        or abs(file_delta) == abs(rank_delta)
    ):
        raise ActionEncodingError(f"move has no AlphaZero action plane: {move.uci()}")

    direction = (file_delta // distance, rank_delta // distance)
    try:
        direction_index = _SLIDING_DIRECTIONS.index(direction)
    # Defensive: geometry above should make this impossible.
    except ValueError as error:
        raise ActionEncodingError(
            f"move has no sliding direction: {move.uci()}"
        ) from error
    plane = direction_index * 7 + distance - 1
    return move.from_square * ACTION_PLANES + plane


def decode_action(board: chess.Board, action: int) -> chess.Move:
    """Decode ``action`` to a legal move without mutating ``board``.

    Invalid indices, off-board destinations, and actions that do not denote a
    currently legal move all raise :class:`ActionEncodingError`.
    """

    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ActionEncodingError(f"action must be an integer, got {type(action)!r}")
    action = int(action)
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ActionEncodingError(
            f"action must be in [0, {ACTION_SPACE_SIZE}), got {action}"
        )

    from_square, plane = divmod(action, ACTION_PLANES)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    promotion: chess.PieceType | None = None

    if plane < 56:
        direction_index, distance_offset = divmod(plane, 7)
        file_step, rank_step = _SLIDING_DIRECTIONS[direction_index]
        distance = distance_offset + 1
        to_file = from_file + file_step * distance
        to_rank = from_rank + rank_step * distance
    elif plane < 64:
        file_delta, rank_delta = _KNIGHT_DIRECTIONS[plane - 56]
        to_file = from_file + file_delta
        to_rank = from_rank + rank_delta
    else:
        direction_index, piece_index = divmod(plane - 64, 3)
        relative_file_delta = direction_index - 1
        forward = 1 if board.turn == chess.WHITE else -1
        file_delta = (
            relative_file_delta
            if board.turn == chess.WHITE
            else -relative_file_delta
        )
        to_file = from_file + file_delta
        to_rank = from_rank + forward
        promotion = _UNDERPROMOTION_PIECES[piece_index]

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ActionEncodingError(f"action destination is off board: {action}")

    to_square = chess.square(to_file, to_rank)
    if promotion is None:
        piece = board.piece_at(from_square)
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and to_rank in (0, 7)
        ):
            promotion = chess.QUEEN

    move = chess.Move(from_square, to_square, promotion=promotion)
    if move not in board.legal_moves:
        raise ActionEncodingError(f"action does not denote a legal move: {action}")
    return move


def legal_action_mask(board: chess.Board) -> NDArray[np.bool_]:
    """Return a boolean mask with exactly the legal actions set to true."""

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[encode_move(board, move)] = True
    return mask


# Readable synonyms for callers that prefer action-first naming.
move_to_action = encode_move
action_to_move = decode_action


__all__ = [
    "ACTION_PLANES",
    "ACTION_SPACE_SIZE",
    "NUM_ACTION_PLANES",
    "NUM_ACTIONS",
    "ActionEncodingError",
    "action_to_move",
    "decode_action",
    "encode_move",
    "legal_action_mask",
    "move_to_action",
]
