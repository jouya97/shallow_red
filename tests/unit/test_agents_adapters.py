from __future__ import annotations

from dataclasses import dataclass

import chess
import pytest

from worst_chess.agents.adapters import SelfishLoserOpponentAgent
from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.evaluation.match import MatchConfig, play_match


@dataclass
class RecordingLoser:
    received: MoveContext | None = None

    @property
    def name(self) -> str:
        return "recording_loser"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        self.received = context
        assert context.target_color == board.turn
        move = min(board.legal_moves, key=chess.Move.uci)
        board.push(move)
        return move


class IllegalLoser:
    @property
    def name(self) -> str:
        return "illegal_loser"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci("a1a8")


def test_selfish_loser_retargets_context_to_acting_opponent() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    original = board.fen()
    loser = RecordingLoser()
    adapter = SelfishLoserOpponentAgent(loser)
    outer = MoveContext("two-losers", 11, 73, chess.WHITE)

    move = adapter.select_move(board, outer)

    assert move in board.legal_moves
    assert board.fen() == original
    assert loser.received == MoveContext("two-losers", 11, 73, chess.BLACK)
    assert adapter.name == "selfish_loser_opponent[recording_loser]"


def test_selfish_loser_adapter_runs_in_existing_match_harness() -> None:
    result = play_match(
        HeuristicAgent(),
        SelfishLoserOpponentAgent(HeuristicAgent()),
        MatchConfig(
            game_id="selfish-vs-selfish",
            seed=5,
            target_color=chess.WHITE,
            max_plies=12,
        ),
    )

    assert result.protocol_failure is None
    assert result.black_agent == "selfish_loser_opponent[heuristic_loser]"
    assert result.plies


def test_selfish_loser_adapter_requires_outer_opponent_role() -> None:
    adapter = SelfishLoserOpponentAgent(RecordingLoser())

    with pytest.raises(AgentError, match="non-target color"):
        adapter.select_move(
            chess.Board(),
            MoveContext("wrong-role", 0, 1, chess.WHITE),
        )


def test_selfish_loser_adapter_rejects_illegal_wrapped_move() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    adapter = SelfishLoserOpponentAgent(IllegalLoser())

    with pytest.raises(AgentError, match="returned illegal"):
        adapter.select_move(
            board,
            MoveContext("illegal", 1, 1, chess.WHITE),
        )


def test_selfish_loser_adapter_rejects_terminal_position() -> None:
    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    adapter = SelfishLoserOpponentAgent(RecordingLoser())

    with pytest.raises(AgentError, match="terminal"):
        adapter.select_move(
            terminal,
            MoveContext("terminal", 0, 1, chess.WHITE),
        )
