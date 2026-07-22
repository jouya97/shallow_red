"""Terminal adjudication from the designated loser's point of view."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess

from worst_chess.objective.rewards import terminal_utility


class DrawPolicy(str, Enum):
    """Whether an available optional draw claim ends the game."""

    NEVER_CLAIM = "never_claim"
    CLAIM_AVAILABLE = "claim_available"


@dataclass(frozen=True)
class TargetOutcome:
    """Adjudication result from one designated target color's perspective."""

    terminal: bool
    winner: chess.Color | None
    termination: chess.Termination | None
    utility: float | None
    target_was_checkmated: bool
    target_won: bool


def adjudicate(
    board: chess.Board,
    target_color: chess.Color,
    draw_policy: DrawPolicy = DrawPolicy.NEVER_CLAIM,
) -> TargetOutcome:
    """Adjudicate ``board`` without mutating it.

    Automatic outcomes (including checkmate, stalemate, dead position,
    fivefold repetition, and the 75-move rule) are terminal under either
    policy.  Threefold-repetition and 50-move claims are terminal only under
    :attr:`DrawPolicy.CLAIM_AVAILABLE`.
    """

    if not isinstance(draw_policy, DrawPolicy):
        raise TypeError(f"draw_policy must be DrawPolicy, got {draw_policy!r}")

    # Claim detection may replay the move stack internally.  A full copy gives
    # callers a stronger non-mutation guarantee than relying on that internal
    # implementation detail being perfectly restored.
    probe = board.copy(stack=True)
    outcome = probe.outcome(
        claim_draw=draw_policy is DrawPolicy.CLAIM_AVAILABLE
    )
    if outcome is None:
        return TargetOutcome(
            terminal=False,
            winner=None,
            termination=None,
            utility=None,
            target_was_checkmated=False,
            target_won=False,
        )

    target_was_checkmated = (
        outcome.termination is chess.Termination.CHECKMATE
        and outcome.winner is not None
        and outcome.winner != target_color
    )
    target_won = outcome.winner == target_color
    return TargetOutcome(
        terminal=True,
        winner=outcome.winner,
        termination=outcome.termination,
        utility=terminal_utility(outcome.winner, target_color),
        target_was_checkmated=target_was_checkmated,
        target_won=target_won,
    )


__all__ = ["DrawPolicy", "TargetOutcome", "adjudicate"]
