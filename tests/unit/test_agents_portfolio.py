from __future__ import annotations

from dataclasses import dataclass

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.portfolio import RegimeSwitchingOpponentAgent


@dataclass
class EdgeMoveAgent:
    label: str
    last: bool
    calls: int = 0

    @property
    def name(self) -> str:
        return self.label

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        self.calls += 1
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        move = legal[-1] if self.last else legal[0]
        board.push(move)
        return move


class IllegalAgent:
    @property
    def name(self) -> str:
        return "illegal"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci("a1a8")


def _context(*, ply: int = 1, seed: int = 29) -> MoveContext:
    return MoveContext("portfolio-game", ply, seed, chess.WHITE)


def test_portfolio_keeps_member_stable_within_regime_and_varies_across_suite() -> None:
    agent = RegimeSwitchingOpponentAgent(
        [EdgeMoveAgent("first", False), EdgeMoveAgent("last", True)],
        regime_plies=8,
        salt="test",
    )

    assert agent.member_name_for(_context(ply=0)) == agent.member_name_for(
        _context(ply=7)
    )
    selected = {
        agent.member_name_for(MoveContext(f"game-{game}", 1, game, chess.WHITE))
        for game in range(32)
    }
    assert selected == {"first", "last"}


def test_portfolio_is_deterministic_legal_and_delegates_on_copy() -> None:
    first = EdgeMoveAgent("first", False)
    last = EdgeMoveAgent("last", True)
    agent = RegimeSwitchingOpponentAgent([first, last], regime_plies=4)
    board = chess.Board()
    board.push_uci("e2e4")
    original = board.fen()

    selected_once = agent.select_move(board, _context())
    selected_twice = agent.select_move(board, _context())

    assert selected_once == selected_twice
    assert selected_once in board.legal_moves
    assert first.calls + last.calls == 2
    assert board.fen() == original


def test_portfolio_weights_change_the_selected_member() -> None:
    first = EdgeMoveAgent("first", False)
    last = EdgeMoveAgent("last", True)
    contexts = [MoveContext(f"game-{index}", 1, 3, chess.WHITE) for index in range(50)]
    balanced = RegimeSwitchingOpponentAgent([first, last], salt="weighted")
    biased = RegimeSwitchingOpponentAgent(
        [first, last], weights=[1, 100], salt="weighted"
    )

    balanced_last = sum(
        balanced.member_name_for(context) == "last" for context in contexts
    )
    biased_last = sum(
        biased.member_name_for(context) == "last" for context in contexts
    )

    assert biased_last > balanced_last


def test_portfolio_validates_configuration_role_terminal_and_member_move() -> None:
    member = EdgeMoveAgent("member", False)
    with pytest.raises(ValueError, match="at least two"):
        RegimeSwitchingOpponentAgent([member])
    with pytest.raises(ValueError, match="one entry"):
        RegimeSwitchingOpponentAgent([member, member], weights=[1])
    with pytest.raises(ValueError, match="positive integers"):
        RegimeSwitchingOpponentAgent([member, member], weights=[1, 0])
    with pytest.raises(ValueError, match="regime_plies"):
        RegimeSwitchingOpponentAgent([member, member], regime_plies=0)

    agent = RegimeSwitchingOpponentAgent([member, member])
    with pytest.raises(AgentError, match="non-target"):
        agent.select_move(chess.Board(), MoveContext("game", 0, 1, chess.WHITE))

    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    with pytest.raises(AgentError, match="terminal"):
        agent.select_move(terminal, MoveContext("game", 0, 1, chess.WHITE))

    board = chess.Board()
    board.push_uci("e2e4")
    illegal = RegimeSwitchingOpponentAgent([IllegalAgent(), IllegalAgent()])
    with pytest.raises(AgentError, match="returned illegal"):
        illegal.select_move(board, _context())
