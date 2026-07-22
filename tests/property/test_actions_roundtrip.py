from __future__ import annotations

import random

import chess
from hypothesis import given
from hypothesis import strategies as st

from worst_chess.chess.actions import decode_action, encode_move, legal_action_mask


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_every_legal_move_has_one_decodable_masked_action(seed: int) -> None:
    random_source = random.Random(seed)
    board = chess.Board()
    for _ in range(random_source.randrange(0, 180)):
        if board.is_game_over():
            break
        board.push(random_source.choice(list(board.legal_moves)))

    original_fen = board.fen()
    original_stack = tuple(board.move_stack)
    mask = legal_action_mask(board)
    legal_moves = list(board.legal_moves)
    actions = [encode_move(board, move) for move in legal_moves]

    assert len(actions) == len(set(actions))
    assert mask.sum() == len(legal_moves)
    assert all(mask[action] for action in actions)
    assert {decode_action(board, action) for action in actions} == set(legal_moves)
    assert board.fen() == original_fen
    assert tuple(board.move_stack) == original_stack

