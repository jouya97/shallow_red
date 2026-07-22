from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.merge_retro_reports import merge_reports  # noqa: E402


def report(seed_index: int, fen: str, status: str, gain: int | None) -> dict:
    return {
        "seeds": [{"seed_index": seed_index, "generated_ancestors": 3}],
        "records": [
            {
                "fen": fen,
                "status": status,
                "distance_gain": gain,
            }
        ],
    }


def test_merges_statuses_and_orders_seed_indices() -> None:
    merged = merge_reports(
        [
            report(2, "fen-b", "refuted", None),
            report(0, "fen-a", "proven", 2),
        ]
    )

    assert merged["seed_indices"] == [0, 2]
    assert merged["summary"] == {
        "generated_ancestors": 6,
        "searched": 2,
        "proven": 1,
        "proven_extended": 1,
        "refuted": 1,
        "unknown": 0,
        "duplicate_fens": 0,
    }


def test_rejects_overlapping_shard_coverage() -> None:
    with pytest.raises(ValueError, match="duplicate seed"):
        merge_reports(
            [
                report(0, "fen-a", "proven", 2),
                report(0, "fen-b", "refuted", None),
            ]
        )
