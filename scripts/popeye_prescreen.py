"""Use Popeye as a fast prescreen for short orthodox selfmates.

Popeye is substantially faster than the auditable Python AND/OR solver on
many composition-style positions.  This script only identifies candidates;
positive results must still be replayed through ``prove_forced_selfmate`` so
that Shallow Red's exact draw and history rules remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chess

_SOLUTION_RE = re.compile(r"(?m)^\s*1\.[A-Za-z0-9]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--popeye", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-moves", type=int, default=4)
    parser.add_argument("--seconds-per-attempt", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def popeye_input(board: chess.Board, target_color: chess.Color, moves: int) -> str:
    """Render a history-free orthodox target-turn position for Popeye."""

    if moves < 1:
        raise ValueError("moves must be positive")
    if board.turn != target_color:
        raise ValueError("the target must be on move")
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(
        chess.BLACK
    ):
        raise ValueError("castling rights are not supported by the prescreen")
    if board.ep_square is not None:
        raise ValueError("en-passant state is not supported by the prescreen")
    normalized = board if target_color == chess.WHITE else board.mirror()
    return (
        "begin\n"
        f"forsyth {normalized.board_fen()}\n"
        f"stipulation s#{moves}\n"
        "option NoBoard MaxSolutions 1 NoCastling a1h1a8h8\n"
        "end\n"
    )


def classify_output(output: str) -> str:
    """Classify one completed Popeye invocation conservatively."""

    if _SOLUTION_RE.search(output):
        return "found"
    if "solution finished" in output.lower():
        return "not_found"
    return "unknown"


def prescreen_candidates(
    candidates: list[dict[str, Any]],
    *,
    popeye: Path,
    start: int,
    count: int | None,
    max_moves: int,
    seconds_per_attempt: int,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must not be negative")
    if count is not None and count < 1:
        raise ValueError("count must be positive")
    if max_moves < 1 or seconds_per_attempt < 1:
        raise ValueError("search limits must be positive")
    if not popeye.is_file():
        raise FileNotFoundError(popeye)

    stop = len(candidates) if count is None else min(len(candidates), start + count)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset, candidate in enumerate(candidates[start:stop]):
        index = start + offset
        board = chess.Board(candidate["fen"])
        target_color = candidate["target_color"] == "white"
        record: dict[str, Any] = {"index": index, **candidate}
        try:
            inputs = [
                popeye_input(board, target_color, moves)
                for moves in range(1, max_moves + 1)
            ]
        except ValueError as error:
            record.update(status="unsupported", detail=str(error), attempts=[])
            records.append(record)
            continue

        attempts: list[dict[str, Any]] = []
        final_status = "not_found"
        for moves, content in enumerate(inputs, start=1):
            attempt_started = time.perf_counter()
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".inp"
            ) as stream:
                stream.write(content)
                stream.flush()
                try:
                    completed = subprocess.run(
                        [str(popeye), stream.name],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=seconds_per_attempt,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    attempts.append(
                        {
                            "moves": moves,
                            "status": "unknown",
                            "elapsed_seconds": time.perf_counter()
                            - attempt_started,
                            "detail": "process timeout",
                        }
                    )
                    final_status = "unknown"
                    break
            status = (
                classify_output(completed.stdout)
                if completed.returncode == 0
                else "error"
            )
            attempt = {
                "moves": moves,
                "status": status,
                "elapsed_seconds": time.perf_counter() - attempt_started,
            }
            if status in {"found", "error", "unknown"}:
                attempt["output"] = completed.stdout[-4_000:]
            attempts.append(attempt)
            final_status = status
            if status != "not_found":
                break
        record.update(status=final_status, attempts=attempts)
        records.append(record)
        if (offset + 1) % 10 == 0:
            counts = Counter(item["status"] for item in records)
            print(
                f"positions={offset + 1} found={counts['found']} "
                f"not_found={counts['not_found']} unknown={counts['unknown']} "
                f"unsupported={counts['unsupported']}",
                flush=True,
            )

    counts = Counter(record["status"] for record in records)
    return {
        "input_positions": len(candidates),
        "start": start,
        "count": len(records),
        "max_moves": max_moves,
        "seconds_per_attempt": seconds_per_attempt,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": dict(sorted(counts.items())),
        "records": records,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    candidates = [
        json.loads(line)
        for line in arguments.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = prescreen_candidates(
        candidates,
        popeye=arguments.popeye,
        start=arguments.start,
        count=arguments.count,
        max_moves=arguments.max_moves,
        seconds_per_attempt=arguments.seconds_per_attempt,
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
