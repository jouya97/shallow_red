"""Apply hard win avoidance to synthetic-ancestry safety examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.chess.actions import decode_action
from worst_chess.training.ranked_dataset import (
    RankedPosition,
    rank_position,
    read_ranked_jsonl,
    write_ranked_jsonl,
)
from worst_chess.training.rollout_teacher import RolloutConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--rollout-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--safety-first-all", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def decode_rollout_score(
    score: float,
    config: RolloutConfig,
) -> tuple[int, int, int, int, int]:
    """Invert the exact mixed-radix rollout score.

    Returns ``(selfmates, selfmate_plies_sum, target_wins, draws,
    truncations)``. The scorer guarantees integer-valued floats within the
    exact IEEE-754 range, so exhaustive inversion is small and unambiguous.
    """

    if not score.is_integer():
        raise ValueError("rollout score must be an integer-valued float")
    encoded = int(score)
    speed_radix = config.rollouts * config.max_plies + 1
    outcome_radix = config.rollouts + 1
    matches: list[tuple[int, int, int, int, int]] = []
    for selfmates in range(config.rollouts + 1):
        for plies_sum in range(selfmates, selfmates * config.max_plies + 1):
            for target_wins in range(config.rollouts - selfmates + 1):
                for truncations in range(config.rollouts - selfmates - target_wins + 1):
                    draws = config.rollouts - selfmates - target_wins - truncations
                    candidate = (
                        (selfmates * speed_radix - plies_sum) * outcome_radix
                        - target_wins
                    ) * outcome_radix - truncations
                    if candidate == encoded:
                        matches.append(
                            (
                                selfmates,
                                plies_sum,
                                target_wins,
                                draws,
                                truncations,
                            )
                        )
    if len(matches) != 1:
        raise ValueError(f"rollout score {score} decoded to {len(matches)} outcomes")
    return matches[0]


def safety_first_score(
    outcome: tuple[int, int, int, int, int],
    config: RolloutConfig,
) -> float:
    """Rank fewer target wins before loss reliability and speed."""

    selfmates, plies_sum, target_wins, _draws, truncations = outcome
    speed_radix = config.rollouts * config.max_plies + 1
    maximum_speed_component = config.rollouts * config.rollouts * config.max_plies
    safety_radix = maximum_speed_component + 1
    safe_rollouts = config.rollouts - target_wins
    selfmate_and_speed = selfmates * speed_radix - plies_sum
    score = (safe_rollouts * safety_radix + selfmate_and_speed) * (
        config.rollouts + 1
    ) - truncations
    return float(score)


def finalize_dataset(
    positions: tuple[RankedPosition, ...],
    manifest: dict[str, Any],
    *,
    config: RolloutConfig,
    safety_first_all: bool = False,
) -> tuple[tuple[RankedPosition, ...], dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    metadata = {(record["fen"], record["target_color"]): record for record in records}
    finalized: list[RankedPosition] = []
    report_records: list[dict[str, Any]] = []
    seen: set[tuple[str, chess.Color]] = set()
    for index, position in enumerate(positions):
        key = (position.fen, position.target_color)
        if key in seen:
            raise ValueError("ranked inputs contain a duplicate position")
        seen.add(key)
        color_name = "white" if position.target_color else "black"
        try:
            record = metadata[(position.fen, color_name)]
        except KeyError as error:
            raise ValueError("ranked position is missing from the manifest") from error
        board = position.board()
        move_scores = {
            decode_action(board, target.action): float(target.teacher_score)
            for target in position.move_targets
        }
        original_best = _best_moves(move_scores)
        is_observed_win_safety = (
            record.get("seed_kind") == "observed-win-safety"
        )
        uses_safety_order = safety_first_all or is_observed_win_safety
        if uses_safety_order:
            outcomes = {
                move: decode_rollout_score(score, config)
                for move, score in move_scores.items()
            }
            move_scores = {
                move: safety_first_score(outcome, config)
                for move, outcome in outcomes.items()
            }
        final_best = _best_moves(move_scores)

        def scorer(
            scoring_board: chess.Board,
            context: MoveContext,
            scores: dict[chess.Move, float] = move_scores,
        ) -> dict[chess.Move, float]:
            del scoring_board, context
            return scores

        finalized.append(
            rank_position(
                board,
                target_color=position.target_color,
                scorer=scorer,
                context=MoveContext(
                    game_id=position.trajectory_id,
                    ply=board.ply(),
                    seed=config.seed + index,
                    target_color=position.target_color,
                ),
                source_id=(
                    f"{position.source_id}/safety-first-v1"
                    if uses_safety_order
                    else position.source_id
                ),
                trajectory_id=position.trajectory_id,
                value_target=position.value_target,
            )
        )
        report_records.append(
            {
                "fen": position.fen,
                "seed_kind": record.get("seed_kind"),
                "uses_safety_order": uses_safety_order,
                "original_best_moves": sorted(move.uci() for move in original_best),
                "final_best_moves": sorted(move.uci() for move in final_best),
                "best_changed": original_best != final_best,
            }
        )
    return tuple(finalized), {
        "schema": "synthetic-ancestry-finalization-v1",
        "positions": len(finalized),
        "safety_first_all": safety_first_all,
        "safety_positions": sum(
            record["seed_kind"] == "observed-win-safety" for record in report_records
        ),
        "safety_best_changed": sum(
            record["seed_kind"] == "observed-win-safety" and record["best_changed"]
            for record in report_records
        ),
        "all_best_changed": sum(
            record["best_changed"] for record in report_records
        ),
        "records": report_records,
    }


def _best_moves(scores: dict[chess.Move, float]) -> set[chess.Move]:
    best = max(scores.values())
    return {move for move, score in scores.items() if score == best}


def main() -> int:
    arguments = build_parser().parse_args()
    positions: list[RankedPosition] = []
    for path in arguments.input:
        positions.extend(read_ranked_jsonl(path))
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    finalized, report = finalize_dataset(
        tuple(positions),
        manifest,
        config=RolloutConfig(
            rollouts=arguments.rollouts,
            max_plies=arguments.rollout_plies,
            seed=arguments.seed,
        ),
        safety_first_all=arguments.safety_first_all,
    )
    write_ranked_jsonl(arguments.output, finalized)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "positions": report["positions"],
                "all_best_changed": report["all_best_changed"],
                "safety_positions": report["safety_positions"],
                "safety_best_changed": report["safety_best_changed"],
            },
            sort_keys=True,
        )
    )
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
