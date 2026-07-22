"""Role adapters for composing agents with the target-relative match API."""

from __future__ import annotations

from dataclasses import replace

import chess

from worst_chess.agents.base import Agent, AgentError, MoveContext


class SelfishLoserOpponentAgent:
    """Let the non-target player use a policy that tries to lose for itself.

    The match harness supplies the same ``MoveContext.target_color`` to both
    seats: it always names the player whose outcome the match will score.  A
    losing policy acting in the opponent seat would therefore optimize the
    wrong color if called directly.  This adapter changes only
    ``target_color`` in the wrapped policy's context to the acting color.

    The original board is protected from a mutating wrapped policy, and the
    returned move is checked against the original legal-move set.  Game ID,
    ply, and seed are preserved, so deterministic policies remain reproducible.
    """

    def __init__(self, loser: Agent) -> None:
        self.loser = loser

    @property
    def name(self) -> str:
        return f"selfish_loser_opponent[{self.loser.name}]"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn == context.target_color:
            raise AgentError(
                f"{self.name} must act for the non-target color in the outer match"
            )
        legal = tuple(board.legal_moves)
        if not legal:
            raise AgentError(f"{self.name} cannot move from a terminal position")

        selfish_context = replace(context, target_color=board.turn)
        move = self.loser.select_move(board.copy(stack=True), selfish_context)
        if not isinstance(move, chess.Move) or move not in legal:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(
                f"wrapped losing agent {self.loser.name} returned illegal move {detail}"
            )
        return move


__all__ = ["SelfishLoserOpponentAgent"]
