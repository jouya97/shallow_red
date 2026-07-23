"""Evaluate a checkpoint on one or more explicit ranked datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from worst_chess.chess.neural_actions import (
    ACTION_ORIENTATION_METADATA_KEY,
    PERSPECTIVE_ACTION_ORIENTATION,
)
from worst_chess.training.model import load_checkpoint
from worst_chess.training.ranked_dataset import read_ranked_jsonl
from worst_chess.training.ranked_trainer import evaluate_ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank-temperature", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    model, metadata = load_checkpoint(arguments.checkpoint, device="cpu")
    positions = tuple(
        position
        for path in arguments.dataset
        for position in read_ranked_jsonl(path)
    )
    perspective_actions = (
        metadata.get(ACTION_ORIENTATION_METADATA_KEY)
        == PERSPECTIVE_ACTION_ORIENTATION
    )
    metrics = evaluate_ranked(
        model,
        positions,
        batch_size=arguments.batch_size,
        rank_temperature=arguments.rank_temperature,
        value_loss_weight=0.0,
        device=arguments.device,
        perspective_actions=perspective_actions,
    )
    report = {
        "checkpoint": str(arguments.checkpoint),
        "datasets": [str(path) for path in arguments.dataset],
        "positions": len(positions),
        "perspective_actions": perspective_actions,
        "metrics": asdict(metrics),
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
