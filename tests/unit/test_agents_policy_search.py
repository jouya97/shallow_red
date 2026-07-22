from __future__ import annotations

import chess
import torch
from torch import Tensor, nn

from worst_chess.agents.base import MoveContext
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.policy_search import PolicyGuidedReverseSearchAgent
from worst_chess.agents.stockfish import ReverseMoveScore
from worst_chess.chess.actions import ACTION_SPACE_SIZE, encode_move
from worst_chess.training.model import ModelConfig, PolicyValueNetwork


class FixedPolicy(PolicyValueNetwork):
    def __init__(self, preferred_action: int | None) -> None:
        nn.Module.__init__(self)
        self.config = ModelConfig(channels=1, residual_blocks=1)
        self.preferred_action = preferred_action

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.zeros(
            observations.shape[0], ACTION_SPACE_SIZE, device=observations.device
        )
        if self.preferred_action is not None:
            logits[:, self.preferred_action] = 100.0
        return logits, torch.zeros(observations.shape[0], 1, device=observations.device)


class FakeEvaluator:
    name = "fake_reverse"

    def __init__(self, wanted: chess.Move) -> None:
        self.wanted = wanted
        self.received: list[chess.Move] = []

    def evaluate_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        root_moves: list[chess.Move] | tuple[chess.Move, ...] | None = None,
    ) -> tuple[ReverseMoveScore, ...]:
        del board, context
        assert root_moves is not None
        self.received = list(root_moves)
        ranked = sorted(root_moves, key=lambda move: move != self.wanted)
        return tuple(
            ReverseMoveScore(move, 0.0, float(-index), 1000, 0)
            for index, move in enumerate(ranked)
        )


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("policy-search", 0, 1, color)


def test_policy_guided_search_only_evaluates_top_k_and_uses_search_winner() -> None:
    board = chess.Board()
    policy_first = chess.Move.from_uci("e2e4")
    search_winner = chess.Move.from_uci("d2d4")
    model = FixedPolicy(encode_move(board, policy_first))
    # Give d2d4 the second-highest distinct logit.
    original_forward = model.forward

    def forward(observations):  # type: ignore[no-untyped-def]
        logits, value = original_forward(observations)
        logits[:, encode_move(board, search_winner)] = 90.0
        return logits, value

    model.forward = forward  # type: ignore[method-assign]
    evaluator = FakeEvaluator(search_winner)
    agent = PolicyGuidedReverseSearchAgent(
        NeuralAgent(model), evaluator, top_k=2  # type: ignore[arg-type]
    )
    original_fen = board.fen()

    selected = agent.select_move(board, _context())

    assert selected == search_winner
    assert evaluator.received == [policy_first, search_winner]
    assert board.fen() == original_fen


def test_policy_guided_search_rejects_target_mismatch() -> None:
    board = chess.Board()
    evaluator = FakeEvaluator(next(iter(board.legal_moves)))
    model: PolicyValueNetwork = FixedPolicy(None)
    agent = PolicyGuidedReverseSearchAgent(
        NeuralAgent(model), evaluator  # type: ignore[arg-type]
    )

    try:
        agent.select_move(board, _context(chess.BLACK))
    except Exception as error:
        assert "target color" in str(error)
    else:
        raise AssertionError("expected target-color mismatch")


def test_policy_guided_search_rejects_invalid_top_k() -> None:
    board = chess.Board()
    evaluator = FakeEvaluator(next(iter(board.legal_moves)))
    model = NeuralAgent(FixedPolicy(None))
    try:
        PolicyGuidedReverseSearchAgent(
            model, evaluator, top_k=0  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected invalid top_k")
