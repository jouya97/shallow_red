from __future__ import annotations

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.resistant import ResistantOpponentAgent


def _context(target_color: chess.Color = chess.BLACK) -> MoveContext:
    return MoveContext(
        game_id="resistant",
        ply=0,
        seed=17,
        target_color=target_color,
    )


def test_resistant_opponent_avoids_immediate_mate_when_alternative_exists() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    mating_move = chess.Move.from_uci("f7g7")
    after_mate = board.copy(stack=False)
    after_mate.push(mating_move)
    assert after_mate.is_checkmate()

    selected = ResistantOpponentAgent().select_move(board, _context())
    after_selected = board.copy(stack=False)
    after_selected.push(selected)

    assert selected in board.legal_moves
    assert selected != mating_move
    assert not after_selected.is_checkmate()


def test_resistant_opponent_values_an_ordinary_material_gain() -> None:
    board = chess.Board("7k/8/8/8/3q4/8/3R4/7K w - - 0 1")
    agent = ResistantOpponentAgent()

    capture = agent.score_move(board, chess.Move.from_uci("d2d4"), chess.BLACK)
    quiet = agent.score_move(board, chess.Move.from_uci("d2a2"), chess.BLACK)

    assert capture > quiet


def test_resistant_opponent_prefers_target_king_safety() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    agent = ResistantOpponentAgent()

    checking = agent.score_move(board, chess.Move.from_uci("a1a8"), chess.BLACK)
    quiet = agent.score_move(board, chess.Move.from_uci("a1a2"), chess.BLACK)

    assert checking < quiet


def test_resistant_opponent_is_deterministic_legal_and_non_mutating() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    original_fen = board.fen()
    agent = ResistantOpponentAgent()
    context = _context(chess.WHITE)

    first = agent.select_move(board, context)
    second = agent.select_move(board, context)

    assert first == second
    assert first in board.legal_moves
    assert board.fen() == original_fen


def test_resistant_opponent_requires_non_target_turn() -> None:
    board = chess.Board()
    agent = ResistantOpponentAgent()

    with pytest.raises(AgentError, match="non-target color"):
        agent.select_move(board, _context(chess.WHITE))
    with pytest.raises(AgentError, match="target color"):
        agent.score_move(board, chess.Move.from_uci("e2e4"), chess.WHITE)


def test_resistant_opponent_rejects_illegal_and_terminal_moves() -> None:
    agent = ResistantOpponentAgent()
    board = chess.Board()
    with pytest.raises(AgentError, match="illegal"):
        agent.score_move(board, chess.Move.from_uci("a1a8"), chess.BLACK)

    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    with pytest.raises(AgentError, match="terminal"):
        agent.select_move(terminal, _context(chess.WHITE))
