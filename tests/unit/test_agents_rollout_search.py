from __future__ import annotations

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import PolicyMove
from worst_chess.agents.rollout_search import NeuralShortlistRolloutAgent
from worst_chess.training.rollout_teacher import RolloutConfig


class MutatingFakePolicy:
    name = "mutating-fake"

    def __init__(self, root_moves: tuple[chess.Move, ...]) -> None:
        self.root_moves = root_moves

    def rank_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        top_k: int | None = None,
    ) -> tuple[PolicyMove, ...]:
        del context
        limit = len(self.root_moves) if top_k is None else top_k
        ranked = tuple(
            PolicyMove(move, index, float(-index))
            for index, move in enumerate(self.root_moves[:limit])
        )
        board.push(min(board.legal_moves, key=chess.Move.uci))
        return ranked

    def select_move(
        self, board: chess.Board, context: MoveContext
    ) -> chess.Move:
        del context
        move = min(board.legal_moves, key=chess.Move.uci)
        board.push(move)
        return move


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("shortlist-rollout", 2, 97, color)


def _mating_opponent(
    board: chess.Board, context: MoveContext
) -> chess.Move:
    assert board.turn != context.target_color
    for move in sorted(board.legal_moves, key=chess.Move.uci):
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return move
    return min(board.legal_moves, key=chess.Move.uci)


def test_shortlist_rollout_finds_selfmate_and_does_not_mutate() -> None:
    board = chess.Board()
    board.push_uci("f2f3")
    board.push_uci("e7e5")
    policy = MutatingFakePolicy(
        (
            chess.Move.from_uci("a2a3"),
            chess.Move.from_uci("g2g4"),
            chess.Move.from_uci("h2h4"),
        )
    )
    agent = NeuralShortlistRolloutAgent(  # type: ignore[arg-type]
        policy,
        top_k=2,
        config=RolloutConfig(rollouts=3, max_plies=2, seed=11),
        opponent=_mating_opponent,
    )
    original_fen = board.fen(en_passant="fen")
    original_stack = tuple(board.move_stack)

    first = agent.select_move(board, _context())
    second = agent.select_move(board, _context())
    summaries = agent.evaluate_shortlist(board, _context())

    assert first == second == chess.Move.from_uci("g2g4")
    assert {item.move.uci() for item in summaries} == {"a2a3", "g2g4"}
    assert all(item.move in board.legal_moves for item in summaries)
    assert board.fen(en_passant="fen") == original_fen
    assert tuple(board.move_stack) == original_stack


def test_shortlist_rollout_uses_uci_tie_break_and_validates_inputs() -> None:
    board = chess.Board()
    policy = MutatingFakePolicy(
        (
            chess.Move.from_uci("h2h4"),
            chess.Move.from_uci("a2a3"),
        )
    )
    agent = NeuralShortlistRolloutAgent(  # type: ignore[arg-type]
        policy,
        top_k=2,
        config=RolloutConfig(rollouts=1, max_plies=1),
    )

    assert agent.select_move(board, _context()) == chess.Move.from_uci("a2a3")
    with pytest.raises(ValueError, match="positive integer"):
        NeuralShortlistRolloutAgent(policy, top_k=0)  # type: ignore[arg-type]
    with pytest.raises(AgentError, match="target color"):
        agent.select_move(board, _context(chess.BLACK))


def test_shortlist_rollout_rejects_illegal_policy_candidate() -> None:
    policy = MutatingFakePolicy((chess.Move.from_uci("a1a8"),))
    agent = NeuralShortlistRolloutAgent(  # type: ignore[arg-type]
        policy,
        top_k=1,
        config=RolloutConfig(rollouts=1, max_plies=1),
    )

    with pytest.raises(AgentError, match="illegal"):
        agent.select_move(chess.Board(), _context())
