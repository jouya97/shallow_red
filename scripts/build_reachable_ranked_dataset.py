"""Seed strict ranked records from reachable FEN candidate files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.agents.opponent_model import RandomReplySearchAgent
from worst_chess.training.ranked_dataset import (
    RankedPosition,
    rank_position,
    write_ranked_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    return parser


def build_reachable_positions(
    records: list[dict[str, Any]],
    *,
    seed: int,
    source_id: str,
) -> tuple[RankedPosition, ...]:
    scorer = RandomReplySearchAgent().score_moves
    positions: list[RankedPosition] = []
    for index, record in enumerate(records):
        fen = record.get("fen")
        target_name = record.get("target_color")
        if not isinstance(fen, str):
            raise ValueError(f"record {index} must contain a FEN string")
        if target_name not in {"white", "black"}:
            raise ValueError(f"record {index} target_color must be white or black")
        board = chess.Board(fen)
        target_color = target_name == "white"
        if not board.is_valid() or board.turn != target_color:
            raise ValueError(f"record {index} must be a valid target-to-move position")
        game_id = str(record.get("game_id", "unknown"))
        trajectory_id = f"{source_id}/{game_id}"
        context = MoveContext(
            game_id=trajectory_id,
            ply=board.ply(),
            seed=_stable_int(seed, index, fen, game_id),
            target_color=target_color,
        )
        positions.append(
            rank_position(
                board,
                target_color=target_color,
                scorer=scorer,
                context=context,
                source_id=source_id,
                trajectory_id=trajectory_id,
            )
        )
    return tuple(positions)


def _stable_int(*parts: object) -> int:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def main() -> int:
    arguments = build_parser().parse_args()
    records = [
        json.loads(line)
        for line in arguments.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if arguments.start < 0:
        raise ValueError("--start must be nonnegative")
    if arguments.count is not None and arguments.count < 1:
        raise ValueError("--count must be positive")
    stop = (
        len(records)
        if arguments.count is None
        else arguments.start + arguments.count
    )
    positions = build_reachable_positions(
        records[arguments.start:stop],
        seed=arguments.seed,
        source_id=f"reachable-{arguments.input.stem}-seed-{arguments.seed}",
    )
    write_ranked_jsonl(arguments.output, positions)
    print(f"positions: {len(positions)}")
    print(f"dataset: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
