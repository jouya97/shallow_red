import chess
import numpy as np

from worst_chess.chess.observations import (
    EN_PASSANT_PLANE,
    HALFMOVE_CLOCK_PLANE,
    OBSERVATION_SHAPE,
    OWN_KINGSIDE_CASTLING_PLANE,
    SIDE_TO_MOVE_PLANE,
    encode_observation,
)


def test_initial_observation_from_white_perspective() -> None:
    board = chess.Board()
    before = board.fen()

    observation = encode_observation(board, chess.WHITE)

    assert observation.shape == OBSERVATION_SHAPE
    assert observation.dtype == np.float32
    assert observation[0, 1, :].sum() == 8  # Own pawns on rank two.
    assert observation[6, 6, :].sum() == 8  # Opponent pawns on rank seven.
    assert observation[SIDE_TO_MOVE_PLANE].all()
    assert observation[OWN_KINGSIDE_CASTLING_PLANE].all()
    assert board.fen() == before


def test_black_perspective_is_rank_mirrored_and_side_normalized() -> None:
    board = chess.Board()
    board.turn = chess.BLACK

    observation = encode_observation(board, chess.BLACK)

    assert observation[0, 1, :].sum() == 8
    assert observation[6, 6, :].sum() == 8
    assert observation[SIDE_TO_MOVE_PLANE].all()


def test_en_passant_and_halfmove_planes() -> None:
    board = chess.Board("8/8/8/8/3pP3/8/8/4K2k b - e3 30 40")

    observation = encode_observation(board, chess.WHITE)

    assert observation[EN_PASSANT_PLANE, 2, 4] == 1.0
    assert observation[EN_PASSANT_PLANE].sum() == 1.0
    assert np.allclose(observation[HALFMOVE_CLOCK_PLANE], 0.2)

