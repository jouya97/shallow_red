"""Compact, perspective-normalized neural-network observations."""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

PIECE_PLANES = 12
SIDE_TO_MOVE_PLANE = 12
OWN_KINGSIDE_CASTLING_PLANE = 13
OWN_QUEENSIDE_CASTLING_PLANE = 14
OPPONENT_KINGSIDE_CASTLING_PLANE = 15
OPPONENT_QUEENSIDE_CASTLING_PLANE = 16
EN_PASSANT_PLANE = 17
HALFMOVE_CLOCK_PLANE = 18
REPETITION_TWICE_PLANE = 19
REPETITION_THREE_TIMES_PLANE = 20
OBSERVATION_PLANES = 21
OBSERVATION_SHAPE = (OBSERVATION_PLANES, 8, 8)


def _oriented_square(square: chess.Square, perspective: chess.Color) -> chess.Square:
    """Keep files stable while placing the perspective's home rank at the bottom."""

    return square if perspective == chess.WHITE else chess.square_mirror(square)


def _fill_flag(observation: NDArray[np.float32], plane: int, flag: bool) -> None:
    if flag:
        observation[plane].fill(1.0)


def encode_observation(
    board: chess.Board, perspective: chess.Color
) -> NDArray[np.float32]:
    """Encode ``board`` without mutating it.

    Planes 0-5 contain the perspective's pawn through king, planes 6-11 contain
    the opponent's pieces, followed by side-to-move, castling, en-passant,
    halfmove-clock, and repetition state. Black positions are mirrored by rank
    so that the perspective's home rank is always row zero.
    """

    observation = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        owner_offset = 0 if piece.color == perspective else 6
        plane = owner_offset + piece.piece_type - 1
        oriented = _oriented_square(square, perspective)
        rank = chess.square_rank(oriented)
        file = chess.square_file(oriented)
        observation[plane, rank, file] = 1.0

    _fill_flag(observation, SIDE_TO_MOVE_PLANE, board.turn == perspective)
    _fill_flag(
        observation,
        OWN_KINGSIDE_CASTLING_PLANE,
        board.has_kingside_castling_rights(perspective),
    )
    _fill_flag(
        observation,
        OWN_QUEENSIDE_CASTLING_PLANE,
        board.has_queenside_castling_rights(perspective),
    )
    _fill_flag(
        observation,
        OPPONENT_KINGSIDE_CASTLING_PLANE,
        board.has_kingside_castling_rights(not perspective),
    )
    _fill_flag(
        observation,
        OPPONENT_QUEENSIDE_CASTLING_PLANE,
        board.has_queenside_castling_rights(not perspective),
    )

    if board.ep_square is not None:
        ep_square = _oriented_square(board.ep_square, perspective)
        observation[
            EN_PASSANT_PLANE,
            chess.square_rank(ep_square),
            chess.square_file(ep_square),
        ] = 1.0

    observation[HALFMOVE_CLOCK_PLANE].fill(min(board.halfmove_clock, 150) / 150.0)
    _fill_flag(observation, REPETITION_TWICE_PLANE, board.is_repetition(2))
    _fill_flag(observation, REPETITION_THREE_TIMES_PLANE, board.is_repetition(3))
    return observation

