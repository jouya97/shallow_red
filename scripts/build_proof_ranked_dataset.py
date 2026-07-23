"""Turn proven selfmate roots into honest all-legal-move policy labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.objective.proof_search import (
    ProofSearchConfig,
    prove_forced_selfmate_after_move,
)
from worst_chess.training.ranked_dataset import (
    RankedPosition,
    rank_position,
    write_ranked_jsonl,
)


@dataclass(frozen=True, slots=True)
class LabeledProofPosition:
    positions: tuple[RankedPosition, ...]
    report: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-report", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--node-budget", type=int, default=100_000)
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def label_proof_position(
    record: dict[str, Any],
    *,
    node_budget: int,
    mirror: bool,
) -> LabeledProofPosition:
    if node_budget < 1:
        raise ValueError("node_budget must be positive")
    board = chess.Board(record["fen"])
    target_color = record["target_color"] == "white"
    if board.turn != target_color:
        raise ValueError("proof target must be the side to move")
    horizon = record.get("forced_plies")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("proof record must have positive forced_plies")

    statuses: dict[chess.Move, str] = {}
    nodes = 0
    for move in sorted(board.legal_moves, key=chess.Move.uci):
        result = prove_forced_selfmate_after_move(
            board,
            target_color,
            move,
            ProofSearchConfig(max_plies=horizon, node_budget=node_budget),
        )
        statuses[move] = result.status.value
        nodes += result.nodes
    counts = Counter(statuses.values())
    if not counts["proven"]:
        raise ValueError("input claim was not reproduced by all-move labeling")

    root_id = record.get("root_source_id", record.get("source_id", "unknown"))
    trajectory_id = f"yacpdb-{root_id}"
    positions = [
        _ranked_position(
            board,
            statuses,
            source_id="forced-selfmate-proof-v1",
            trajectory_id=trajectory_id,
        )
    ]
    if mirror:
        mirrored_board = board.mirror()
        mirrored_statuses = {
            mirror_move(move): status for move, status in statuses.items()
        }
        positions.append(
            _ranked_position(
                mirrored_board,
                mirrored_statuses,
                source_id="forced-selfmate-proof-v1-mirrored",
                trajectory_id=trajectory_id,
            )
        )
    return LabeledProofPosition(
        positions=tuple(positions),
        report={
            "fen": record["fen"],
            "root_source_id": root_id,
            "horizon": horizon,
            "legal_moves": len(statuses),
            "proven": counts["proven"],
            "refuted": counts["refuted"],
            "unknown": counts["unknown"],
            "nodes": nodes,
        },
    )


def build_dataset(
    report: dict[str, Any],
    *,
    start: int,
    count: int | None,
    node_budget: int,
    mirror: bool,
) -> tuple[tuple[RankedPosition, ...], dict[str, Any]]:
    if start < 0:
        raise ValueError("start must not be negative")
    if count is not None and count < 1:
        raise ValueError("count must be positive")
    raw_records = report.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("proof report records must be a list")
    proven = [record for record in raw_records if record.get("status") == "proven"]
    stop = len(proven) if count is None else min(len(proven), start + count)
    selected = proven[start:stop]
    positions: list[RankedPosition] = []
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset, record in enumerate(selected):
        labeled = label_proof_position(
            record,
            node_budget=node_budget,
            mirror=mirror,
        )
        positions.extend(labeled.positions)
        reports.append(labeled.report)
        print(
            f"positions={offset + 1} legal={labeled.report['legal_moves']} "
            f"proven={labeled.report['proven']} "
            f"unknown={labeled.report['unknown']}",
            flush=True,
        )
    return tuple(positions), {
        "input_proven": len(proven),
        "start": start,
        "count": len(selected),
        "dataset_positions": len(positions),
        "node_budget": node_budget,
        "mirrored": mirror,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "legal_moves": sum(item["legal_moves"] for item in reports),
            "proven": sum(item["proven"] for item in reports),
            "refuted": sum(item["refuted"] for item in reports),
            "unknown": sum(item["unknown"] for item in reports),
            "nodes": sum(item["nodes"] for item in reports),
        },
        "records": reports,
    }


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def _ranked_position(
    board: chess.Board,
    statuses: dict[chess.Move, str],
    *,
    source_id: str,
    trajectory_id: str,
) -> RankedPosition:
    scores = {"proven": 2.0, "unknown": 1.0, "refuted": 0.0}

    def scorer(
        scoring_board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        del scoring_board, context
        return {move: scores[status] for move, status in statuses.items()}

    return rank_position(
        board,
        target_color=board.turn,
        scorer=scorer,
        context=MoveContext(
            game_id=trajectory_id,
            ply=board.ply(),
            seed=0,
            target_color=board.turn,
        ),
        source_id=source_id,
        trajectory_id=trajectory_id,
        value_target=1.0,
    )


def main() -> int:
    arguments = build_parser().parse_args()
    source_report = json.loads(arguments.proof_report.read_text(encoding="utf-8"))
    positions, report = build_dataset(
        source_report,
        start=arguments.start,
        count=arguments.count,
        node_budget=arguments.node_budget,
        mirror=arguments.mirror,
    )
    write_ranked_jsonl(arguments.output, positions)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
