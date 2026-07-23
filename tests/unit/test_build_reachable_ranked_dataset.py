from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.build_reachable_ranked_dataset import (  # noqa: E402
    build_reachable_positions,
)


def test_builds_deterministic_all_legal_move_records() -> None:
    records = [
        {
            "fen": chess.STARTING_FEN,
            "target_color": "white",
            "game_id": "game-1",
        }
    ]

    first = build_reachable_positions(records, seed=7, source_id="reachable")
    second = build_reachable_positions(records, seed=7, source_id="reachable")

    assert first == second
    assert len(first) == 1
    assert len(first[0].move_targets) == 20
    assert first[0].trajectory_id == "reachable/game-1"
    assert first[0].value_target is None


def test_rejects_wrong_turn_target() -> None:
    with pytest.raises(ValueError, match="target-to-move"):
        build_reachable_positions(
            [
                {
                    "fen": chess.STARTING_FEN,
                    "target_color": "black",
                }
            ],
            seed=1,
            source_id="reachable",
        )
