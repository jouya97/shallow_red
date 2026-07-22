"""Canonical sparse terminal utility for the designated losing side."""

from __future__ import annotations

import chess

TARGET_CHECKMATED_UTILITY = 1.0
"""Utility when the designated target's king is checkmated."""

DRAW_UTILITY = 0.0
"""Utility for any drawn result."""

TARGET_WIN_UTILITY = -1.0
"""Utility when the designated target checkmates its opponent."""


def terminal_utility(
    winner: chess.Color | None,
    target_color: chess.Color,
) -> float:
    """Return sparse terminal utility from ``target_color``'s perspective.

    In orthodox chess, a decisive board outcome is checkmate.  A winner of
    ``None`` therefore denotes a draw, the opponent winning means the target
    was checkmated, and the target winning means it accidentally checkmated
    its opponent.
    """

    if winner is None:
        return DRAW_UTILITY
    if winner == target_color:
        return TARGET_WIN_UTILITY
    return TARGET_CHECKMATED_UTILITY


__all__ = [
    "DRAW_UTILITY",
    "TARGET_CHECKMATED_UTILITY",
    "TARGET_WIN_UTILITY",
    "terminal_utility",
]
