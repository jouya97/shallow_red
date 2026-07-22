"""Reproducible regime switching across purposeful opponent policies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import chess

from worst_chess.agents.base import Agent, AgentError, MoveContext


class RegimeSwitchingOpponentAgent:
    """Use one member policy for each deterministic multi-ply regime.

    A losing policy can overfit the exact tactical preferences of any single
    shallow opponent.  This wrapper makes that failure visible by switching
    among two or more purposeful policies.  Selection is derived from the game
    context and a salt, not the board position, so the target cannot steer the
    active policy by reaching a specially hashed FEN.  A policy remains active
    for ``regime_plies`` half-moves to retain some strategic coherence.

    Unlike :class:`NoisyOpponentAgent`, this class never substitutes a uniform
    random move.  It invokes exactly one member per decision, making its cost
    equal to the selected member's cost.  Identical contexts always select the
    same member, independent of process hash randomization.
    """

    def __init__(
        self,
        members: Sequence[Agent],
        *,
        weights: Sequence[int] | None = None,
        regime_plies: int = 8,
        salt: str = "default",
    ) -> None:
        if len(members) < 2:
            raise ValueError("portfolio requires at least two member agents")
        if regime_plies <= 0:
            raise ValueError("regime_plies must be positive")
        if not salt:
            raise ValueError("salt must not be empty")

        resolved_weights = (
            tuple(weights) if weights is not None else (1,) * len(members)
        )
        if len(resolved_weights) != len(members):
            raise ValueError("weights must have one entry per member agent")
        if any(weight <= 0 for weight in resolved_weights):
            raise ValueError("portfolio weights must be positive integers")

        self.members = tuple(members)
        self.weights = resolved_weights
        self.regime_plies = regime_plies
        self.salt = salt

    @property
    def name(self) -> str:
        roster = ",".join(member.name for member in self.members)
        return f"regime_portfolio[{roster}]"

    def member_name_for(self, context: MoveContext) -> str:
        """Return the selected member name for audit and report metadata."""

        return self._member_for(context).name

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn == context.target_color:
            raise AgentError(f"{self.name} must act for the non-target color")
        legal = tuple(board.legal_moves)
        if not legal:
            raise AgentError(f"{self.name} cannot move from a terminal position")

        member = self._member_for(context)
        move = member.select_move(board.copy(stack=True), context)
        if not isinstance(move, chess.Move) or move not in legal:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(
                f"portfolio member {member.name} returned illegal move {detail}"
            )
        return move

    def _member_for(self, context: MoveContext) -> Agent:
        regime = context.ply // self.regime_plies
        payload = (
            f"worst-chess-regime-portfolio-v1\0{self.salt}\0{context.seed}\0"
            f"{context.game_id}\0{regime}"
        ).encode()
        sample = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        choice = sample % sum(self.weights)

        cumulative = 0
        for member, weight in zip(self.members, self.weights, strict=True):
            cumulative += weight
            if choice < cumulative:
                return member
        raise AssertionError("positive portfolio weights must select a member")


__all__ = ["RegimeSwitchingOpponentAgent"]
