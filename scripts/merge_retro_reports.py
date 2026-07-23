"""Merge independently written retro-expansion shard reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for report in reports for record in report.get("records", [])]
    seeds = [seed for report in reports for seed in report.get("seeds", [])]
    seed_indices = [seed["seed_index"] for seed in seeds]
    duplicates = sorted(
        index for index, count in Counter(seed_indices).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate seed indices: {duplicates}")
    fen_counts = Counter(record["fen"] for record in records)
    duplicate_fens = sum(count - 1 for count in fen_counts.values() if count > 1)
    statuses = Counter(record["status"] for record in records)
    proven_extended = sum(
        record["status"] == "proven"
        and record.get(
            "distance_gain_verified", record.get("distance_gain", 0) > 0
        )
        for record in records
    )
    return {
        "shard_count": len(reports),
        "seed_count": len(seeds),
        "seed_indices": sorted(seed_indices),
        "summary": {
            "generated_ancestors": sum(
                seed.get("generated_ancestors", 0) for seed in seeds
            ),
            "searched": len(records),
            "proven": statuses["proven"],
            "proven_extended": proven_extended,
            "refuted": statuses["refuted"],
            "unknown": statuses["unknown"],
            "duplicate_fens": duplicate_fens,
        },
        "seeds": sorted(seeds, key=lambda seed: seed["seed_index"]),
        "records": records,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in arguments.input
    ]
    merged = merge_reports(reports)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(merged["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
