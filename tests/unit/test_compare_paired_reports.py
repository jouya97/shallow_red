from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "compare_paired_reports.py"


def _game(
    seed: int,
    color: str,
    outcome: str,
    plies: int,
) -> dict[str, object]:
    winner: str | None = None
    selfmate = outcome == "selfmate"
    target_win = outcome == "target_win"
    if selfmate:
        winner = "black" if color == "white" else "white"
    elif target_win:
        winner = color
    return {
        "seed": seed,
        "target_color": color,
        "plies": plies,
        "winner": winner,
        "target_was_checkmated": selfmate,
        "target_won": target_win,
        "protocol_failure": None,
        "truncated": outcome == "truncation",
    }


def _write_report(
    path: Path,
    games: list[dict[str, object]],
    *,
    openings: list[str] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "tournament": {
            "pairs": 2,
            "base_seed": 100,
            "opening_fens": openings or ["opening-a", "opening-b"],
            "max_plies": 200,
        },
        "games": games,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compares_exact_keys_and_bootstraps_openings(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_games = [
        _game(100, "white", "selfmate", 100),
        _game(100, "black", "selfmate", 120),
        _game(101, "white", "draw", 70),
        _game(101, "black", "selfmate", 80),
    ]
    candidate_games = [
        _game(100, "white", "selfmate", 90),
        _game(100, "black", "draw", 100),
        _game(101, "white", "selfmate", 70),
        _game(101, "black", "selfmate", 60),
    ]
    _write_report(baseline_path, baseline_games)
    # Report ordering is deliberately irrelevant to exact-key alignment.
    _write_report(candidate_path, list(reversed(candidate_games)))

    command = [
        sys.executable,
        str(SCRIPT),
        str(baseline_path),
        str(candidate_path),
        "--resamples",
        "200",
        "--seed",
        "17",
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    result = json.loads(first.stdout)

    assert result["alignment"]["games"] == 4
    assert result["alignment"]["opening_clusters"] == 2
    assert result["baseline"]["overall"]["selfmates"] == 3
    assert result["candidate"]["overall"]["selfmates"] == 3
    assert result["paired"]["both_selfmate"] == 2
    assert result["paired"]["candidate_only_selfmate"] == 1
    assert result["paired"]["baseline_only_selfmate"] == 1
    assert result["paired"]["joint_selfmate_mean_plies_improvement"] == 15
    assert result["paired"]["joint_selfmate_median_plies_improvement"] == 15
    assert result["paired"]["restricted_plies_mean_improvement"] == 20
    reliability = result["bootstrap"]["metrics"][
        "selfmate_rate_difference_pp"
    ]
    assert reliability["estimate"] == 0


def test_rejects_misaligned_report_keys(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    games = [
        _game(100, "white", "selfmate", 10),
        _game(100, "black", "selfmate", 10),
        _game(101, "white", "selfmate", 10),
        _game(101, "black", "selfmate", 10),
    ]
    _write_report(baseline_path, games)
    _write_report(candidate_path, games, openings=["opening-a", "changed"])

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(baseline_path),
            str(candidate_path),
            "--resamples",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "report keys do not align" in completed.stderr
