from __future__ import annotations

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.heuristic import HeuristicAgent


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext(game_id="heuristic", ply=0, seed=1, target_color=color)


def test_heuristic_prefers_offering_queen_to_capturing_rook() -> None:
    board = chess.Board("7k/8/8/8/3r4/8/3Q4/6K1 w - - 0 1")
    original_fen = board.fen()
    agent = HeuristicAgent()

    selected = agent.select_move(board, _context())

    # The selected queen move puts it directly on the rook's fourth rank.
    assert selected == chess.Move.from_uci("d2b4")
    after = board.copy(stack=False)
    after.push(selected)
    assert chess.Move.from_uci("d4b4") in after.legal_moves
    assert board.fen() == original_fen


def test_heuristic_penalizes_taking_enemy_material() -> None:
    board = chess.Board("7k/8/8/8/3r4/8/3Q4/6K1 w - - 0 1")
    agent = HeuristicAgent()

    sacrifice = agent.score_move(board, chess.Move.from_uci("d2b4"), chess.WHITE)
    capture = agent.score_move(board, chess.Move.from_uci("d2d4"), chess.WHITE)

    assert sacrifice > capture


def test_heuristic_rewards_exposing_king_ring() -> None:
    # Moving the g-pawn removes one of the king's remaining pawn-shield pieces.
    board = chess.Board("7k/8/8/8/8/8/P5PP/6K1 w - - 0 1")
    agent = HeuristicAgent()

    expose = agent.score_move(board, chess.Move.from_uci("g2g3"), chess.WHITE)
    quiet = agent.score_move(board, chess.Move.from_uci("a2a3"), chess.WHITE)

    assert expose > quiet


def test_heuristic_rejects_illegal_and_terminal_moves() -> None:
    agent = HeuristicAgent()
    board = chess.Board()
    with pytest.raises(AgentError, match="illegal"):
        agent.score_move(board, chess.Move.from_uci("a1a8"), chess.WHITE)

    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    with pytest.raises(AgentError, match="terminal"):
        agent.select_move(terminal, _context(chess.BLACK))
