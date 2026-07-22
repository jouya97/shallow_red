"""Extract reachable near-mate positions and run bounded selfmate proofs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from worst_chess.objective.proof_search import (
    ProofSearchConfig,
    ProofStatus,
    prove_forced_selfmate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--pgn", type=Path, nargs="+", required=True)
    extract.add_argument("--tail-target-positions", type=int, default=4)
    extract.add_argument("--output", type=Path, required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--input", type=Path, required=True)
    search.add_argument("--start", type=int, default=0)
    search.add_argument("--count", type=int)
    search.add_argument("--max-plies", type=int, nargs="+", default=(2, 4, 6))
    search.add_argument("--node-budget", type=int, default=100_000)
    search.add_argument("--output", type=Path, required=True)
    return parser


def extract_candidates(
    paths: list[Path],
    *,
    tail_target_positions: int,
) -> list[dict[str, Any]]:
    if tail_target_positions < 1:
        raise ValueError("tail_target_positions must be positive")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, bool]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            while game := chess.pgn.read_game(stream):
                target_header = game.headers.get("Target")
                if target_header not in {"white", "black"}:
                    continue
                target_color = target_header == "white"
                board = game.board()
                target_positions: list[tuple[str, int]] = []
                for move in game.mainline_moves():
                    if board.turn == target_color:
                        target_positions.append(
                            (board.fen(en_passant="fen"), board.ply())
                        )
                    board.push(move)
                if not (board.is_checkmate() and board.turn == target_color):
                    continue
                selected = target_positions[-tail_target_positions:]
                for fen, ply in selected:
                    key = (fen, target_color)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "fen": fen,
                            "target_color": "white" if target_color else "black",
                            "source": path.name,
                            "game_id": game.headers.get("Round", "unknown"),
                            "ply": ply,
                            "plies_before_observed_mate": board.ply() - ply,
                        }
                    )
    return candidates


def search_candidates(
    candidates: list[dict[str, Any]],
    *,
    start: int,
    count: int | None,
    horizons: tuple[int, ...],
    node_budget: int,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must not be negative")
    if count is not None and count < 1:
        raise ValueError("count must be positive")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("max-plies horizons must be positive")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("max-plies horizons must be unique and increasing")
    stop = len(candidates) if count is None else min(len(candidates), start + count)
    selected = candidates[start:stop]
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset, candidate in enumerate(selected):
        board = chess.Board(candidate["fen"])
        target_color = candidate["target_color"] == "white"
        attempts: list[dict[str, Any]] = []
        final_status = ProofStatus.REFUTED
        for horizon in horizons:
            result = prove_forced_selfmate(
                board,
                target_color,
                ProofSearchConfig(max_plies=horizon, node_budget=node_budget),
            )
            attempts.append(
                {
                    "max_plies": horizon,
                    "status": result.status.value,
                    "forced_plies": result.plies,
                    "nodes": result.nodes,
                    "cache_hits": result.cache_hits,
                    "principal_variation": [
                        move.uci() for move in result.principal_variation
                    ],
                }
            )
            final_status = result.status
            if result.status in {ProofStatus.PROVEN, ProofStatus.UNKNOWN}:
                break
        records.append(
            {
                "index": start + offset,
                **candidate,
                "status": final_status.value,
                "attempts": attempts,
            }
        )
        if (offset + 1) % 10 == 0:
            counts = Counter(record["status"] for record in records)
            print(
                f"positions={offset + 1} proven={counts['proven']} "
                f"refuted={counts['refuted']} unknown={counts['unknown']}",
                flush=True,
            )
    counts = Counter(record["status"] for record in records)
    return {
        "input_positions": len(candidates),
        "start": start,
        "count": len(records),
        "horizons": list(horizons),
        "node_budget": node_budget,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "proven": counts["proven"],
            "refuted": counts["refuted"],
            "unknown": counts["unknown"],
        },
        "records": records,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "extract":
        candidates = extract_candidates(
            arguments.pgn,
            tail_target_positions=arguments.tail_target_positions,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            "".join(
                json.dumps(candidate, sort_keys=True) + "\n"
                for candidate in candidates
            ),
            encoding="utf-8",
        )
        print(f"candidates: {len(candidates)}")
        print(f"output: {arguments.output}")
        return 0

    result = search_candidates(
        _read_jsonl(arguments.input),
        start=arguments.start,
        count=arguments.count,
        horizons=tuple(arguments.max_plies),
        node_budget=arguments.node_budget,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
