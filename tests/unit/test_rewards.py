from __future__ import annotations

import chess
import pytest

from worst_chess.objective.rewards import (
    DRAW_UTILITY,
    TARGET_CHECKMATED_UTILITY,
    TARGET_WIN_UTILITY,
    terminal_utility,
)


def test_reward_constants_define_reverse_chess_objective() -> None:
    assert TARGET_CHECKMATED_UTILITY == 1.0
    assert DRAW_UTILITY == 0.0
    assert TARGET_WIN_UTILITY == -1.0


@pytest.mark.parametrize("target_color", [chess.WHITE, chess.BLACK])
def test_draw_has_zero_utility(target_color: chess.Color) -> None:
    assert terminal_utility(None, target_color) == 0.0


@pytest.mark.parametrize("target_color", [chess.WHITE, chess.BLACK])
def test_target_winning_is_bad_and_opponent_winning_is_good(
    target_color: chess.Color,
) -> None:
    assert terminal_utility(target_color, target_color) == -1.0
    assert terminal_utility(not target_color, target_color) == 1.0

