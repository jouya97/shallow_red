from __future__ import annotations

from dataclasses import dataclass

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.weak import (
    CaptureFirstOpponentAgent,
    MaterialOpponentAgent,
    NoisyOpponentAgent,
)


def _context(target_color: chess.Color = chess.BLACK) -> MoveContext:
    return MoveContext("weak-opponent", 0, 41, target_color)


@dataclass
class RecordingOpponent:
    calls: int = 0

    @property
    def name(self) -> str:
        return "recording_opponent"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        self.calls += 1
        move = min(board.legal_moves, key=chess.Move.uci)
        board.push(move)
        return move


@pytest.mark.parametrize(
    "agent",
    [MaterialOpponentAgent(), CaptureFirstOpponentAgent()],
)
def test_purposeful_weak_opponents_take_immediate_checkmate(agent: object) -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    original = board.fen()

    move = agent.select_move(board, _context())  # type: ignore[attr-defined]
    result = board.copy(stack=False)
    result.push(move)

    assert result.is_checkmate()
    assert board.fen() == original


def test_material_opponent_prefers_winning_material() -> None:
    board = chess.Board("7k/8/8/8/3q4/8/3R4/7K w - - 0 1")
    agent = MaterialOpponentAgent()

    capture = agent.score_move(board, chess.Move.from_uci("d2d4"), chess.BLACK)
    quiet = agent.score_move(board, chess.Move.from_uci("d2a2"), chess.BLACK)

    assert capture > quiet


def test_capture_first_opponent_takes_highest_value_available_piece() -> None:
    board = chess.Board("7k/8/8/8/q2r4/8/R2Q4/7K w - - 0 1")

    move = CaptureFirstOpponentAgent().select_move(board, _context())

    assert move == chess.Move.from_uci("a2a4")


def test_noisy_opponent_probability_zero_delegates_on_a_copy() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    original = board.fen()
    base = RecordingOpponent()
    agent = NoisyOpponentAgent(base, random_move_probability=0.0)

    move = agent.select_move(board, _context(chess.WHITE))

    assert move in board.legal_moves
    assert base.calls == 1
    assert board.fen() == original


def test_noisy_opponent_probability_one_is_deterministic_and_skips_base() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    base = RecordingOpponent()
    agent = NoisyOpponentAgent(
        base,
        random_move_probability=1.0,
        salt="test",
    )

    first = agent.select_move(board, _context(chess.WHITE))
    second = agent.select_move(board, _context(chess.WHITE))

    assert first == second
    assert first in board.legal_moves
    assert base.calls == 0


def test_weak_opponents_validate_role_and_probability() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NoisyOpponentAgent(MaterialOpponentAgent(), random_move_probability=1.1)
    with pytest.raises(AgentError, match="non-target"):
        MaterialOpponentAgent().select_move(chess.Board(), _context(chess.WHITE))
