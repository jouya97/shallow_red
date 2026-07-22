from __future__ import annotations

import json

import chess

from worst_chess.agents.random import RandomAgent
from worst_chess.evaluation.openings import generate_random_openings
from worst_chess.evaluation.report import write_report
from worst_chess.evaluation.tournament import TournamentConfig, run_paired_tournament


def test_paired_tournament_balances_target_color_and_writes_report(tmp_path) -> None:
    config = TournamentConfig(
        tournament_id="paired",
        pairs=2,
        base_seed=10,
        max_plies=4,
    )

    results = run_paired_tournament(RandomAgent(), RandomAgent(), config)

    assert len(results) == 4
    assert [result.config.target_color for result in results] == [
        chess.WHITE,
        chess.BLACK,
        chess.WHITE,
        chess.BLACK,
    ]
    assert [result.config.seed for result in results] == [10, 10, 11, 11]
    assert all(result.truncated for result in results)

    report_path, pgn_path = write_report(tmp_path, config, results)
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 1
    assert report["summary"]["overall"]["games"] == 4
    assert report["summary"]["overall"]["truncations"] == 4
    assert len(report["games"]) == 4
    assert pgn_path.read_text().count('[Result "*"]') == 4


def test_random_opening_suite_is_unique_reproducible_and_valid() -> None:
    first = generate_random_openings(count=10, plies=6, seed=42)
    second = generate_random_openings(count=10, plies=6, seed=42)

    assert first == second
    assert len(set(first)) == 10
    assert all(chess.Board(fen).is_valid() for fen in first)
