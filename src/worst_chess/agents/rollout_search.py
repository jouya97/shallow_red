"""Inference-time stochastic lookahead against a uniform-random opponent."""

from __future__ import annotations

import chess

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.random import RandomAgent
from worst_chess.training.rollout_teacher import (
    LexicographicRolloutScorer,
    RolloutConfig,
    RolloutMoveScore,
    SelectorLike,
)


class NeuralShortlistRolloutAgent:
    """Roll out only the neural policy's most plausible root actions.

    Future target turns use the frozen neural policy directly and opponent
    turns use deterministic context-seeded uniform random moves.  Each root
    candidate receives common-random-number rollouts from
    :class:`LexicographicRolloutScorer`, so the choice retains the exact
    priority ``(selfmates, -selfmate plies, -target wins, -truncations)``.
    """

    def __init__(
        self,
        policy: NeuralAgent,
        *,
        top_k: int = 4,
        config: RolloutConfig | None = None,
        target_continuation: SelectorLike | None = None,
        opponent: SelectorLike | None = None,
    ) -> None:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.policy = policy
        self.top_k = top_k
        self.config = config or RolloutConfig(rollouts=2, max_plies=80)
        self.scorer = LexicographicRolloutScorer(
            target_continuation or policy,
            opponent or RandomAgent(),
            self.config,
        )

    @property
    def name(self) -> str:
        return (
            f"neural_shortlist_rollout_k{self.top_k}_"
            f"r{self.config.rollouts}_h{self.config.max_plies}_"
            "uniform_random"
        )

    def evaluate_shortlist(
        self,
        board: chess.Board,
        context: MoveContext,
    ) -> tuple[RolloutMoveScore, ...]:
        """Return rollout summaries for the frozen neural top-k shortlist."""

        if board.turn != context.target_color:
            raise AgentError(
                "NeuralShortlistRolloutAgent must act for the target color"
            )
        if board.is_game_over(claim_draw=False):
            raise AgentError(
                "NeuralShortlistRolloutAgent cannot move from a terminal position"
            )
        original_fen = board.fen(en_passant="fen")
        original_stack = tuple(board.move_stack)
        ranked = self.policy.rank_moves(
            board.copy(stack=True),
            context,
            top_k=self.top_k,
        )
        summaries = self.scorer.evaluate_candidates(
            board,
            context,
            (candidate.move for candidate in ranked),
        )
        if (
            board.fen(en_passant="fen") != original_fen
            or tuple(board.move_stack) != original_stack
        ):
            raise RuntimeError("shortlist rollout search mutated the input board")
        return summaries

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        """Choose the lexicographically strongest rollout result."""

        summaries = self.evaluate_shortlist(board, context)
        # Summaries are UCI-ordered, and max() retains the first exact tie.
        return max(summaries, key=lambda item: item.ranking_score).move


__all__ = ["NeuralShortlistRolloutAgent"]
