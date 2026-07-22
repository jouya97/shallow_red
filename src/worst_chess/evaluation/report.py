"""Machine-readable and PGN evaluation artifacts."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

from worst_chess import __version__
from worst_chess.evaluation.match import MatchResult
from worst_chess.evaluation.metrics import EvaluationSummary, summarize
from worst_chess.evaluation.tournament import TournamentConfig


def _color_name(color: chess.Color | None) -> str | None:
    if color is None:
        return None
    return "white" if color else "black"


def _summary_dict(summary: EvaluationSummary) -> dict[str, Any]:
    data = asdict(summary)
    data["termination_counts"] = dict(summary.termination_counts)
    data["by_target_color"] = [asdict(item) for item in summary.by_target_color]
    return data


def _match_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "game_id": result.config.game_id,
        "seed": result.config.seed,
        "target_color": _color_name(result.config.target_color),
        "white_agent": result.white_agent,
        "black_agent": result.black_agent,
        "plies": len(result.plies),
        "winner": _color_name(result.winner),
        "termination": result.termination,
        "target_utility": result.target_utility,
        "target_was_checkmated": result.target_was_checkmated,
        "target_won": result.target_won,
        "protocol_failure": (
            asdict(result.protocol_failure)
            if result.protocol_failure is not None
            else None
        ),
        "truncated": result.truncated,
        "final_fen": result.final_fen,
    }


def write_report(
    output_directory: str | Path,
    tournament: TournamentConfig,
    results: tuple[MatchResult, ...] | list[MatchResult],
) -> tuple[Path, Path]:
    """Write `report.json` and a concatenated `games.pgn` artifact."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    materialized = tuple(results)
    summary = summarize(materialized)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_version": __version__,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "tournament": {
            "tournament_id": tournament.tournament_id,
            "pairs": tournament.pairs,
            "base_seed": tournament.base_seed,
            "opening_fens": list(tournament.opening_fens),
            "draw_policy": tournament.draw_policy.value,
            "max_plies": tournament.max_plies,
        },
        "summary": _summary_dict(summary),
        "games": [_match_dict(result) for result in materialized],
    }

    report_path = output / "report.json"
    pgn_path = output / "games.pgn"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pgn_path.write_text(
        "\n\n".join(result.pgn.rstrip() for result in materialized) + "\n",
        encoding="utf-8",
    )
    return report_path, pgn_path


__all__ = ["write_report"]

