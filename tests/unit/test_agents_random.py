from __future__ import annotations

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.random import RandomAgent


def _context(**overrides: object) -> MoveContext:
    values: dict[str, object] = {
        "game_id": "game-alpha",
        "ply": 7,
        "seed": 1234,
        "target_color": chess.WHITE,
    }
    values.update(overrides)
    return MoveContext(**values)  # type: ignore[arg-type]


def test_random_agent_is_stable_and_legal_without_mutating_board() -> None:
    board = chess.Board()
    original_fen = board.fen()
    agent = RandomAgent()

    first = agent.select_move(board, _context())
    second = agent.select_move(board, _context())

    assert first == chess.Move.from_uci("g2g4")
    assert second == first
    assert first in board.legal_moves
    assert board.fen() == original_fen


def test_random_agent_raises_on_terminal_position() -> None:
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")

    with pytest.raises(AgentError, match="terminal"):
        RandomAgent().select_move(board, _context(target_color=chess.BLACK))
