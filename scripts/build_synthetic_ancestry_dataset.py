"""Build all-legal reranking seeds from confirmed synthetic ancestry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.training.ranked_dataset import (
    RankedPosition,
    rank_position,
    write_ranked_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--confirm-report", type=Path, required=True)
    parser.add_argument(
        "--win-tail-target-positions",
        type=int,
        default=4,
        help="Keep this many final target turns from each winning trajectory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def select_training_seeds(
    screen_report: dict[str, Any],
    confirm_report: dict[str, Any],
    *,
    win_tail_target_positions: int,
) -> list[dict[str, Any]]:
    """Select confirmed loss steering and terminal-win safety positions."""

    if win_tail_target_positions < 1:
        raise ValueError("win_tail_target_positions must be positive")
    screen_records = screen_report.get("records")
    confirm_records = confirm_report.get("records")
    if not isinstance(screen_records, list) or not isinstance(confirm_records, list):
        raise ValueError("screen and confirmation records must be lists")

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in confirm_records:
        source_outcome = record.get("source_outcome")
        if source_outcome not in {"loss", "win"}:
            raise ValueError("confirmation record has invalid source_outcome")
        candidates = record.get("candidates", [])
        best_selfmates = max(
            (candidate.get("selfmates", 0) for candidate in candidates),
            default=0,
        )
        if best_selfmates < 1:
            continue
        came_from_loss = source_outcome == "loss"
        selected.append(
            {
                **record,
                "seed_kind": (
                    "confirmed-loss-steering"
                    if came_from_loss
                    else "recovered-win-steering"
                ),
                # A loss supplies an observed state value.  A counterfactual
                # selfmate recovered from a winning trajectory supplies only
                # a policy preference, so do not invent a value label for it.
                "value_target": 1.0 if came_from_loss else None,
                "confirmed_best_selfmates": best_selfmates,
            }
        )
        seen.add((record["fen"], record["target_color"]))

    for record in screen_records:
        if record.get("source_outcome") != "win":
            continue
        if int(record.get("target_turns_before_end", 0)) > win_tail_target_positions:
            continue
        key = (record["fen"], record["target_color"])
        if key in seen:
            continue
        selected.append(
            {
                **record,
                "seed_kind": "observed-win-safety",
                "value_target": -1.0,
                "confirmed_best_selfmates": None,
            }
        )
        seen.add(key)
    return selected


def build_ranked_seeds(
    records: list[dict[str, Any]],
) -> tuple[RankedPosition, ...]:
    """Create valid placeholder rankings for the all-legal rollout command."""

    heuristic = HeuristicAgent()
    positions: list[RankedPosition] = []
    for index, record in enumerate(records):
        board = chess.Board(record["fen"])
        target_color = record["target_color"] == "white"
        if board.turn != target_color:
            raise ValueError("synthetic ancestry seed target must be on move")
        source_id = f"synthetic-ancestry-v1/{record['seed_kind']}"
        game_root_id = str(record["game_id"]).split("/", maxsplit=1)[0]
        trajectory_id = f"synthetic-ancestry/{game_root_id}"

        def scorer(
            scoring_board: chess.Board,
            context: MoveContext,
        ) -> dict[chess.Move, float]:
            return {
                move: float(
                    heuristic.score_move(
                        scoring_board,
                        move,
                        context.target_color,
                    )
                )
                for move in scoring_board.legal_moves
            }

        positions.append(
            rank_position(
                board,
                target_color=target_color,
                scorer=scorer,
                context=MoveContext(
                    game_id=trajectory_id,
                    ply=board.ply(),
                    seed=20260722 + index,
                    target_color=target_color,
                ),
                source_id=source_id,
                trajectory_id=trajectory_id,
                value_target=(
                    None
                    if record["value_target"] is None
                    else float(record["value_target"])
                ),
            )
        )
    return tuple(positions)


def main() -> int:
    arguments = build_parser().parse_args()
    screen_report = json.loads(arguments.screen_report.read_text(encoding="utf-8"))
    confirm_report = json.loads(arguments.confirm_report.read_text(encoding="utf-8"))
    records = select_training_seeds(
        screen_report,
        confirm_report,
        win_tail_target_positions=arguments.win_tail_target_positions,
    )
    positions = build_ranked_seeds(records)
    write_ranked_jsonl(arguments.output, positions)
    manifest = {
        "schema": "synthetic-ancestry-training-seeds-v1",
        "positions": len(records),
        "confirmed_loss_steering": sum(
            record["seed_kind"] == "confirmed-loss-steering" for record in records
        ),
        "recovered_win_steering": sum(
            record["seed_kind"] == "recovered-win-steering" for record in records
        ),
        "observed_win_safety": sum(
            record["seed_kind"] == "observed-win-safety" for record in records
        ),
        "records": records,
    }
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "positions": manifest["positions"],
                "confirmed_loss_steering": manifest["confirmed_loss_steering"],
                "observed_win_safety": manifest["observed_win_safety"],
                "recovered_win_steering": manifest["recovered_win_steering"],
            },
            sort_keys=True,
        )
    )
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
