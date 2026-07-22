"""Neural top-k policy guidance followed by fixed-budget reverse search."""

from __future__ import annotations

import chess

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.stockfish import ReverseStockfishAgent


class PolicyGuidedReverseSearchAgent:
    """Search only the neural policy's most promising legal root moves.

    The neural policy cheaply reduces the branching factor; Stockfish then
    evaluates every retained root independently under the same fixed budget.
    The wrapper does not own either component's lifecycle, allowing callers to
    share or explicitly close engine resources.
    """

    def __init__(
        self,
        policy: NeuralAgent,
        evaluator: ReverseStockfishAgent,
        *,
        top_k: int = 8,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.policy = policy
        self.evaluator = evaluator
        self.top_k = top_k

    @property
    def name(self) -> str:
        return (
            f"policy_guided_top_{self.top_k}_"
            f"{self.evaluator.name}"
        )

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError(
                "PolicyGuidedReverseSearchAgent must act for the target color"
            )
        candidates = self.policy.rank_moves(board, context, top_k=self.top_k)
        if not candidates:
            raise AgentError(
                "PolicyGuidedReverseSearchAgent cannot move from a terminal position"
            )
        evaluations = self.evaluator.evaluate_moves(
            board,
            context,
            root_moves=[candidate.move for candidate in candidates],
        )
        return evaluations[0].move


__all__ = ["PolicyGuidedReverseSearchAgent"]
