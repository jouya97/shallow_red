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
    SampledExpectimaxConfig,
    StalemateAwareRandomReplySearchAgent,
    TwoTurnRandomReplyAgent,
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


class ConstantContinuation:
    def select_move(
        self, board: chess.Board, context: MoveContext
    ) -> chess.Move:
        assert board.turn == context.target_color
        return min(board.legal_moves, key=chess.Move.uci)

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
        *,
        policy_logit: float = 0.0,
    ) -> float:
        del board, move, target_color, policy_logit
        return 0.0


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


def test_two_turn_search_is_legal_deterministic_and_nonmutating() -> None:
    board = chess.Board()
    policy = FakePolicy(
        [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    )
    agent = TwoTurnRandomReplyAgent(
        policy,  # type: ignore[arg-type]
        top_k=2,
        config=SampledExpectimaxConfig(reply_samples=3, seed=91),
        continuation=ConstantContinuation(),  # type: ignore[arg-type]
    )
    original_fen = board.fen()
    original_stack = list(board.move_stack)

    first = agent.select_move(board, _context())
    second = agent.select_move(board, _context())

    assert first == second == chess.Move.from_uci("d2d4")
    assert first in board.legal_moves
    assert board.fen() == original_fen
    assert board.move_stack == original_stack


def test_two_turn_search_target_win_is_dominated_by_ongoing_line() -> None:
    board = chess.Board("7k/5KQ1/8/8/8/8/8/8 w - - 0 1")
    target_win = chess.Move.from_uci("g7g8")
    ongoing = chess.Move.from_uci("g7g6")
    policy = FakePolicy([target_win, ongoing])
    agent = TwoTurnRandomReplyAgent(
        policy,  # type: ignore[arg-type]
        top_k=2,
        config=SampledExpectimaxConfig(reply_samples=1),
        continuation=ConstantContinuation(),  # type: ignore[arg-type]
    )

    scores = agent.score_candidates(board, _context())

    assert scores[target_win] == agent._TARGET_WIN_VALUE
    assert scores[ongoing] > scores[target_win]
    assert agent.select_move(board, _context()) == ongoing


def test_two_turn_search_validates_configuration_and_role() -> None:
    with pytest.raises(ValueError, match="reply_samples"):
        SampledExpectimaxConfig(reply_samples=0)
    with pytest.raises(ValueError, match="not exceed"):
        SampledExpectimaxConfig(reply_samples=1_025)
    with pytest.raises(ValueError, match="top_k"):
        TwoTurnRandomReplyAgent(FakePolicy([]), top_k=0)  # type: ignore[arg-type]
    with pytest.raises(AgentError, match="target color"):
        TwoTurnRandomReplyAgent(
            FakePolicy([chess.Move.from_uci("e2e4")]),  # type: ignore[arg-type]
            continuation=ConstantContinuation(),  # type: ignore[arg-type]
        ).select_move(chess.Board(), _context(chess.BLACK))
    assert (
        TwoTurnRandomReplyAgent._SELF_MATE_VALUE
        > TwoTurnRandomReplyAgent._DRAW_VALUE
        > TwoTurnRandomReplyAgent._TARGET_WIN_VALUE
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
    with pytest.raises(ValueError, match="cycle_penalty"):
        StalemateAwareRandomReplySearchAgent(cycle_penalty=-1)
    with pytest.raises(ValueError, match="cycle_penalty"):
        StalemateAwareRandomReplySearchAgent(cycle_penalty=math.inf)

    board = chess.Board("8/8/8/6k1/8/5K2/6P1/8 w - - 0 1")
    agent = StalemateAwareRandomReplySearchAgent()

    move = agent.select_move(board, _context())

    assert move in board.legal_moves
    assert agent.weights.low_material_threshold == 1_000
    assert agent.pressure_scale == 1.0


def test_stalemate_aware_cycle_penalty_preserves_default_and_avoids_repeat() -> None:
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1"):
        board.push_uci(move)
    repeating = chess.Move.from_uci("f6g8")
    alternative = chess.Move.from_uci("f6h5")
    repeated = board.copy(stack=True)
    repeated.push(repeating)
    assert repeated.is_repetition(2)

    baseline = StalemateAwareRandomReplySearchAgent()
    zero = StalemateAwareRandomReplySearchAgent(cycle_penalty=0.0)
    penalized = StalemateAwareRandomReplySearchAgent(cycle_penalty=1e12)

    baseline_repeat = baseline.score_move(board, repeating, chess.BLACK)
    assert zero.score_move(board, repeating, chess.BLACK) == baseline_repeat
    assert penalized.score_move(board, repeating, chess.BLACK) == (
        baseline_repeat - 1e12
    )
    assert penalized.score_move(board, alternative, chess.BLACK) == (
        baseline.score_move(board, alternative, chess.BLACK)
    )
    assert penalized.select_move(board, _context(chess.BLACK)) != repeating
    assert zero.name == baseline.name
    assert "cycle_penalty_1e+12" in penalized.name


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


def test_tactical_mate_override_reaches_outside_neural_top_k() -> None:
    board = chess.Board("3n4/r1k1b3/p1p5/PpPp4/6p1/2KP2qb/6N1/5r2 w - - 0 70")
    quiet = chess.Move.from_uci("c3b4")
    mating = chess.Move.from_uci("c3d4")
    policy = FakePolicy([quiet])
    baseline = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
    )
    overridden = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
        tactical_mate_override=True,
    )
    original_fen = board.fen()
    original_stack = list(board.move_stack)

    assert baseline.select_move(board, _context()) == quiet
    assert overridden.select_move(board, _context()) == mating
    assert overridden.select_move(board, _context()) == mating
    assert mating in board.legal_moves
    assert board.fen() == original_fen
    assert board.move_stack == original_stack
    assert "tactical_mate_override" in overridden.name


def test_tactical_mate_override_falls_back_exactly_without_opportunity() -> None:
    board = chess.Board()
    policy = FakePolicy(
        [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    )
    baseline = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
    )
    overridden = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
        tactical_mate_override=True,
    )

    assert overridden.select_move(board, _context()) == baseline.select_move(
        board, _context()
    )


def test_tactical_mate_override_breaks_probability_ties_by_uci() -> None:
    board = chess.Board(
        "5N1k/3rp3/1p1p3r/pR3n2/2PP3P/1K1B4/3B4/q7 w - - 8 62"
    )
    first = chess.Move.from_uci("d2h6")
    second = chess.Move.from_uci("d3b1")
    agent = StalemateAwareRandomReplySearchAgent(
        tactical_mate_override=True
    )

    first_probability = agent._immediate_mate_probability(
        board, first, chess.WHITE
    )
    second_probability = agent._immediate_mate_probability(
        board, second, chess.WHITE
    )

    assert first_probability == second_probability > 0.0
    assert first.uci() < second.uci()
    assert agent.select_move(board, _context()) == first


def test_forced_mate_override_reaches_only_certain_mate_outside_top_k() -> None:
    board = chess.Board(
        "6r1/8/7B/N1R5/3k4/3b1K2/8/1Q4q1 w - - 0 1"
    )
    quiet = chess.Move.from_uci("a5b3")
    forced = chess.Move.from_uci("h6e3")
    policy = FakePolicy([quiet])
    agent = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
        forced_mate_override=True,
    )
    original = board.fen()

    assert agent._immediate_mate_probability(board, forced, chess.WHITE) == 1.0
    assert agent.select_move(board, _context()) == forced
    assert agent.select_move(board, _context()) == forced
    assert board.fen() == original
    assert "forced_mate_override" in agent.name


def test_forced_mate_override_ignores_noncertain_mate_and_falls_back() -> None:
    board = chess.Board("3n4/r1k1b3/p1p5/PpPp4/6p1/2KP2qb/6N1/5r2 w - - 0 70")
    quiet = chess.Move.from_uci("c3b4")
    possible = chess.Move.from_uci("c3d4")
    policy = FakePolicy([quiet])
    baseline = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
    )
    forced = StalemateAwareRandomReplySearchAgent(
        policy,  # type: ignore[arg-type]
        top_k=1,
        forced_mate_override=True,
    )

    probability = forced._immediate_mate_probability(
        board, possible, chess.WHITE
    )
    assert 0.0 < probability < 1.0
    assert forced.select_move(board, _context()) == baseline.select_move(
        board, _context()
    )


def test_mate_override_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        StalemateAwareRandomReplySearchAgent(
            tactical_mate_override=True,
            forced_mate_override=True,
        )


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
