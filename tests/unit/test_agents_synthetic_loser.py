from __future__ import annotations

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.agents.synthetic_loser import (
    ExploringLoserAgent,
    build_synthetic_loser_league,
)


class FixedAgent:
    name = "fixed"

    def __init__(self, move: chess.Move) -> None:
        self.move = move

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return self.move


def _context(target_color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("synthetic-test", 0, 17, target_color)


def test_exploration_is_reproducible_legal_and_nonmutating() -> None:
    board = chess.Board()
    agent = ExploringLoserAgent(
        FixedAgent(chess.Move.from_uci("e2e4")),
        exploration_probability=1.0,
        salt="test",
    )
    original = board.fen()

    first = agent.select_move(board, _context())
    second = agent.select_move(board, _context())

    assert first == second
    assert first in board.legal_moves
    assert board.fen() == original


def test_exploration_rejects_an_avoidable_win() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    mating = chess.Move.from_uci("f7g7")
    agent = ExploringLoserAgent(
        FixedAgent(mating),
        exploration_probability=0.0,
        salt="avoid-win",
    )

    selected = agent.select_move(board, _context())
    after = board.copy(stack=False)
    after.push(selected)

    assert selected != mating
    assert not after.is_checkmate()


def test_synthetic_league_acts_for_outer_opponent() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    league = build_synthetic_loser_league()
    context = _context(chess.WHITE)

    move = league.select_move(board, context)

    assert move in board.legal_moves
    with pytest.raises(Exception, match="non-target"):
        league.select_move(chess.Board(), context)
