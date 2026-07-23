from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.split_ranked_dataset import build_parser  # noqa: E402


def test_splitter_groups_matching_trajectory_suffixes_by_default() -> None:
    arguments = build_parser().parse_args(
        [
            "--input",
            "input.jsonl",
            "--output",
            "splits",
        ]
    )

    assert arguments.group_matching_suffixes is True
