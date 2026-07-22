from __future__ import annotations

import math

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import PolicyMove
from worst_chess.agents.opponent_model import (
    OpportunisticHybridAgent,
    RandomReplySearchAgent,
    RandomReplyWeights,
    StalemateAwareRandomReplySearchAgent,
)
from worst_chess.agents.stockfish import ReverseMoveScore


class FakePolicy:
    def __init__(self, moves: list[chess.Move]) -> None:
        self.moves = moves

    def rank_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        top_k: int | None = None,
    ) -> tuple[PolicyMove, ...]:
        del board, context
        limit = len(self.moves) if top_k is None else top_k
        return tuple(
            PolicyMove(move, index, float(-index))
            for index, move in enumerate(self.moves[:limit])
        )


class FakeEvaluator:
    name = "fake_reverse"

    def __init__(self, wanted: chess.Move) -> None:
        self.wanted = wanted
        self.calls = 0

    def evaluate_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        root_moves: list[chess.Move] | tuple[chess.Move, ...] | None = None,
    ) -> tuple[ReverseMoveScore, ...]:
        del board, context
        self.calls += 1
        assert root_moves is not None
        ranked = sorted(root_moves, key=lambda move: move != self.wanted)
        return tuple(
            ReverseMoveScore(move, 0.0, float(-index), 1000, 0)
            for index, move in enumerate(ranked)
        )


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("random-reply", 0, 7, color)


def test_random_reply_search_prefers_move_with_immediate_mating_reply() -> None:
    board = chess.Board("3n4/r1k1b3/p1p5/PpPp4/6p1/2KP2qb/6N1/5r2 w - - 0 70")
    agent = RandomReplySearchAgent()
    original = board.fen()

    opportunity = agent.score_move(
        board, chess.Move.from_uci("c3d4"), chess.WHITE
    )
    quiet = agent.score_move(board, chess.Move.from_uci("c3b4"), chess.WHITE)
    selected = agent.select_move(board, _context())

    assert opportunity > quiet
    assert selected == chess.Move.from_uci("c3d4")
    assert board.fen() == original


def test_random_reply_search_is_deterministic_and_legal() -> None:
    board = chess.Board()
    agent = RandomReplySearchAgent(top_k=4)

    first = agent.select_move(board, _context())
    second = agent.select_move(board, _context())

    assert first == second
    assert first in board.legal_moves


def test_random_reply_search_rejects_bad_role_move_and_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        RandomReplySearchAgent(top_k=0)
    with pytest.raises(AgentError, match="target color"):
        RandomReplySearchAgent().select_move(chess.Board(), _context(chess.BLACK))
    with pytest.raises(AgentError, match="illegal"):
        RandomReplySearchAgent().score_move(
            chess.Board(),
            chess.Move.from_uci("a1a8"),
            chess.WHITE,
        )


def test_random_reply_teacher_scores_target_checkmate_finitely() -> None:
    board = chess.Board("7k/5KQ1/8/8/8/8/8/8 w - - 0 1")

    scores = RandomReplySearchAgent().score_moves(board, _context())

    target_mate = scores[chess.Move.from_uci("g7g8")]
    assert math.isfinite(target_mate)
    assert target_mate == RandomReplyWeights().target_checkmate


def test_stalemate_aware_search_validates_threshold_and_acts_legally() -> None:
    with pytest.raises(ValueError, match="low_material_threshold"):
        StalemateAwareRandomReplySearchAgent(low_material_threshold=0)
    with pytest.raises(ValueError, match="pressure_scale"):
        StalemateAwareRandomReplySearchAgent(pressure_scale=0)
    with pytest.raises(ValueError, match="pressure_min_material"):
        StalemateAwareRandomReplySearchAgent(pressure_min_material=-1)

    board = chess.Board("8/8/8/6k1/8/5K2/6P1/8 w - - 0 1")
    agent = StalemateAwareRandomReplySearchAgent()

    move = agent.select_move(board, _context())

    assert move in board.legal_moves
    assert agent.weights.low_material_threshold == 1_000
    assert agent.pressure_scale == 1.0


def test_stalemate_aware_pressure_scales_king_attack_features() -> None:
    baseline = StalemateAwareRandomReplySearchAgent()
    pressured = StalemateAwareRandomReplySearchAgent(
        pressure_scale=4.0,
        pressure_min_material=2_000,
    )

    assert (
        pressured.weights.checking_reply_probability
        == baseline.weights.checking_reply_probability
    )
    assert (
        pressured.weights.target_ring_attack
        == baseline.weights.target_ring_attack
    )
    assert pressured.weights.captured_target_value == (
        baseline.weights.captured_target_value
    )
    assert pressured.weights.pressure_scale == 4.0
    assert pressured.weights.pressure_min_material == 2_000


def test_opportunistic_hybrid_takes_mating_reply_without_reverse_search() -> None:
    board = chess.Board("3n4/r1k1b3/p1p5/PpPp4/6p1/2KP2qb/6N1/5r2 w - - 0 70")
    mating = chess.Move.from_uci("c3d4")
    quiet = chess.Move.from_uci("c3b4")
    evaluator = FakeEvaluator(quiet)
    agent = OpportunisticHybridAgent(
        FakePolicy([quiet, mating]),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        policy_top_k=2,
        reverse_top_k=2,
    )

    selected = agent.select_move(board, _context())

    assert selected == mating
    assert evaluator.calls == 0


def test_opportunistic_hybrid_falls_back_to_reverse_search() -> None:
    board = chess.Board()
    first = chess.Move.from_uci("e2e4")
    wanted = chess.Move.from_uci("d2d4")
    evaluator = FakeEvaluator(wanted)
    agent = OpportunisticHybridAgent(
        FakePolicy([first, wanted]),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        policy_top_k=2,
        reverse_top_k=2,
    )

    selected = agent.select_move(board, _context())

    assert selected == wanted
    assert evaluator.calls == 1
