from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.generate_fuzzer_frontier import (  # noqa: E402
    FrontierConfig,
    generate_frontier,
    target_aligned_plies,
)


@pytest.mark.parametrize(
    ("minimum", "color", "expected"),
    [
        (40, chess.WHITE, 40),
        (40, chess.BLACK, 41),
        (41, chess.WHITE, 42),
        (41, chess.BLACK, 41),
    ],
)
def test_target_aligned_plies_leaves_requested_color_on_move(
    minimum: int,
    color: chess.Color,
    expected: int,
) -> None:
    assert target_aligned_plies(minimum, color) == expected


def test_target_aligned_plies_rejects_nonpositive_minimum() -> None:
    with pytest.raises(ValueError, match="positive"):
        target_aligned_plies(0, chess.WHITE)


def test_generate_frontier_rejects_negative_start_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        generate_frontier(
            config=FrontierConfig(
                checkpoint=tmp_path / "unused.pt",
                warmup_plies=40,
                target_top_k=12,
                target_exploration=0.2,
                seed=1,
            ),
            positions=1,
            start_index=-1,
            workers=1,
        )
