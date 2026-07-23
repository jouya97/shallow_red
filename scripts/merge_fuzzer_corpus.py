"""Merge sharded selfmate-fuzzer outputs into one auditable corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess.pgn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_file(directory: Path, filename: str) -> Path:
    direct = directory / filename
    if direct.is_file():
        return direct
    matches = sorted(directory.glob(f"**/{filename}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {filename} beneath {directory}, found {len(matches)}"
        )
    return matches[0]


def merge_fuzzer_corpus(directories: list[Path]) -> dict[str, Any]:
    if not directories:
        raise ValueError("at least one input directory is required")

    records: list[dict[str, Any]] = []
    games: list[chess.pgn.Game] = []
    seen_rounds: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    for directory in directories:
        report = json.loads(
            _resolve_file(directory, "report.json").read_text(encoding="utf-8")
        )
        shard_summaries.append(
            {
                "directory": str(directory),
                "summary": report["summary"],
            }
        )
        records.extend(report["records"])

        pgn_path = _resolve_file(directory, "decisive-games.pgn")
        with pgn_path.open(encoding="utf-8") as stream:
            while game := chess.pgn.read_game(stream):
                round_id = game.headers.get("Round")
                if not round_id:
                    raise ValueError(f"game without Round header in {pgn_path}")
                if round_id in seen_rounds:
                    continue
                seen_rounds.add(round_id)
                games.append(game)

    outcomes = Counter(record["outcome"] for record in records)
    loss_records = [record for record in records if record["outcome"] == "loss"]
    win_records = [record for record in records if record["outcome"] == "win"]
    loss_roots = {record["root_id"] for record in loss_records}
    win_roots = {record["root_id"] for record in win_records}
    return {
        "summary": {
            "shards": len(directories),
            "branches": len(records),
            "losses": outcomes["loss"],
            "wins": outcomes["win"],
            "draws": outcomes["draw"],
            "frontiers": outcomes["frontier"],
            "protocol_failures": outcomes["protocol_failure"],
            "decisive_games": len(games),
            "independent_loss_families": len(loss_roots),
            "independent_win_families": len(win_roots),
            "unique_loss_terminal_fens": len(
                {record["final_fen"] for record in loss_records}
            ),
            "unique_win_terminal_fens": len(
                {record["final_fen"] for record in win_records}
            ),
        },
        "loss_root_ids": sorted(loss_roots),
        "win_root_ids": sorted(win_roots),
        "shards": shard_summaries,
        "records": records,
        "games": games,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    result = merge_fuzzer_corpus(arguments.input)
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "games"}
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.output / "decisive-games.pgn").write_text(
        "\n\n".join(str(game) for game in result["games"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
