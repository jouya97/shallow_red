from __future__ import annotations

import os

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.agents.stockfish import ReverseStockfishAgent

STOCKFISH = os.environ.get("WORST_CHESS_STOCKFISH")
pytestmark = pytest.mark.skipif(
    not STOCKFISH,
    reason="set WORST_CHESS_STOCKFISH to an executable Stockfish binary",
)


def test_reverse_stockfish_real_engine_returns_legal_move() -> None:
    assert STOCKFISH is not None
    board = chess.Board()
    original_fen = board.fen()
    context = MoveContext(
        game_id="stockfish-integration",
        ply=0,
        seed=0,
        target_color=chess.WHITE,
    )

    with ReverseStockfishAgent(STOCKFISH, nodes=100) as agent:
        move = agent.select_move(board, context)

    assert move in board.legal_moves
    assert board.fen() == original_fen
