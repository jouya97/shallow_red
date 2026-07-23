"""Create deterministic trajectory-grouped ranked dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worst_chess.training.ranked_dataset import (
    read_ranked_jsonl,
    split_ranked_by_trajectory,
    write_ranked_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--group-matching-suffixes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    positions = tuple(
        position
        for path in arguments.input
        for position in read_ranked_jsonl(path)
    )
    split = split_ranked_by_trajectory(
        positions,
        seed=arguments.seed,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
        group_matching_suffixes=arguments.group_matching_suffixes,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": arguments.output / "train.jsonl",
        "validation": arguments.output / "validation.jsonl",
        "test": arguments.output / "test.jsonl",
    }
    write_ranked_jsonl(paths["train"], split.train)
    write_ranked_jsonl(paths["validation"], split.validation)
    write_ranked_jsonl(paths["test"], split.test)
    train_families = {position.trajectory_id for position in split.train}
    validation_families = {
        position.trajectory_id for position in split.validation
    }
    test_families = {position.trajectory_id for position in split.test}
    if (
        train_families & validation_families
        or train_families & test_families
        or validation_families & test_families
    ):
        raise RuntimeError("trajectory family leaked across ranked splits")
    summary = {
        "positions": len(positions),
        "trajectory_families": len(
            {position.trajectory_id for position in positions}
        ),
        "train_positions": len(split.train),
        "validation_positions": len(split.validation),
        "test_positions": len(split.test),
        "train_families": len(train_families),
        "validation_families": len(validation_families),
        "test_families": len(test_families),
        "group_matching_suffixes": arguments.group_matching_suffixes,
        "seed": arguments.seed,
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
