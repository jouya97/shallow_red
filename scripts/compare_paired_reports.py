#!/usr/bin/env python3
"""Compare two exactly aligned color-paired tournament reports.

The match report schema stores opening FENs once in the tournament metadata,
not on every game.  This script reconstructs the opening assigned to each
game from the deterministic tournament schedule, aligns reports on
``(opening FEN, seed, target color)``, and resamples whole opening clusters.
Consequently, both target colors and any repeated random seeds for an opening
remain together in every bootstrap sample.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, order=True)
class GameKey:
    """Exact key for one scheduled game."""

    opening_fen: str
    seed: int
    target_color: str


@dataclass(frozen=True, slots=True)
class Game:
    """Comparison fields extracted from a report game."""

    key: GameKey
    plies: int
    selfmate: bool
    draw: bool
    target_win: bool
    truncation: bool
    protocol_failure: bool


@dataclass(frozen=True, slots=True)
class LoadedReport:
    """Validated report data needed by the comparison."""

    path: Path
    max_plies: int
    games: dict[GameKey, Game]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact-key aligned tournament report JSONs with an "
            "opening-cluster bootstrap."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the JSON result; JSON is always printed",
    )
    return parser


def load_report(path: Path) -> LoadedReport:
    """Load a schema-v1 report and reconstruct every scheduled game key."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")

    tournament = _mapping(payload.get("tournament"), path, "tournament")
    pairs = _integer(tournament.get("pairs"), path, "tournament.pairs")
    base_seed = _integer(
        tournament.get("base_seed"), path, "tournament.base_seed"
    )
    max_plies = _integer(
        tournament.get("max_plies"), path, "tournament.max_plies"
    )
    if pairs < 1 or max_plies < 1:
        raise ValueError(f"report {path} has non-positive tournament dimensions")
    opening_fens = tournament.get("opening_fens")
    if not isinstance(opening_fens, list) or not opening_fens:
        raise ValueError(f"report {path} has no tournament opening_fens")
    if not all(isinstance(fen, str) and fen for fen in opening_fens):
        raise ValueError(f"report {path} contains an invalid opening FEN value")

    raw_games = payload.get("games")
    if not isinstance(raw_games, list):
        raise ValueError(f"report {path} has no games list")
    if len(raw_games) != 2 * pairs:
        raise ValueError(
            f"report {path} has {len(raw_games)} games, expected {2 * pairs}"
        )

    games: dict[GameKey, Game] = {}
    seen_pair_colors: set[tuple[int, str]] = set()
    for game_index, raw_game in enumerate(raw_games):
        game = _mapping(raw_game, path, f"games[{game_index}]")
        seed = _integer(game.get("seed"), path, f"games[{game_index}].seed")
        pair_index = seed - base_seed
        if not 0 <= pair_index < pairs:
            raise ValueError(
                f"report {path} game seed {seed} is outside its tournament schedule"
            )
        target_color = game.get("target_color")
        if target_color not in {"white", "black"}:
            raise ValueError(
                f"report {path} game {game_index} has invalid target_color"
            )
        pair_color = (pair_index, target_color)
        if pair_color in seen_pair_colors:
            raise ValueError(
                f"report {path} repeats pair {pair_index} color {target_color}"
            )
        seen_pair_colors.add(pair_color)

        key = GameKey(
            opening_fen=opening_fens[pair_index % len(opening_fens)],
            seed=seed,
            target_color=target_color,
        )
        extracted = _extract_game(path, game_index, game, key)
        if key in games:
            raise ValueError(f"report {path} contains duplicate game key {key}")
        games[key] = extracted

    return LoadedReport(path=path, max_plies=max_plies, games=games)


def compare_reports(
    baseline: LoadedReport,
    candidate: LoadedReport,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Return summaries and paired opening-cluster bootstrap intervals."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if baseline.max_plies != candidate.max_plies:
        raise ValueError(
            "reports use different max_plies, so restricted-time comparison "
            "would not be aligned"
        )
    _verify_aligned_keys(baseline, candidate)

    ordered_keys = sorted(baseline.games)
    baseline_games = [baseline.games[key] for key in ordered_keys]
    candidate_games = [candidate.games[key] for key in ordered_keys]
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(ordered_keys):
        clusters[key.opening_fen].append(index)
    cluster_indices = [
        np.asarray(clusters[fen], dtype=np.int64) for fen in sorted(clusters)
    ]

    baseline_summary = _summarize(baseline_games, baseline.max_plies)
    candidate_summary = _summarize(candidate_games, candidate.max_plies)
    paired = _paired_summary(
        baseline_games, candidate_games, baseline.max_plies
    )
    bootstrap = _cluster_bootstrap(
        baseline_games,
        candidate_games,
        cluster_indices,
        max_plies=baseline.max_plies,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "baseline_report": str(baseline.path),
        "candidate_report": str(candidate.path),
        "alignment": {
            "key_fields": ["opening_fen", "seed", "target_color"],
            "games": len(ordered_keys),
            "opening_clusters": len(cluster_indices),
            "max_plies": baseline.max_plies,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired": paired,
        "bootstrap": {
            "method": "percentile bootstrap by opening FEN",
            "resamples": resamples,
            "confidence": confidence,
            "seed": seed,
            "metrics": bootstrap,
        },
    }


def _extract_game(
    path: Path, game_index: int, game: dict[str, Any], key: GameKey
) -> Game:
    prefix = f"report {path} game {game_index}"
    plies = _integer(game.get("plies"), path, f"games[{game_index}].plies")
    if plies < 0:
        raise ValueError(f"{prefix} has negative plies")
    selfmate = _boolean(game.get("target_was_checkmated"), prefix, "selfmate")
    target_win = _boolean(game.get("target_won"), prefix, "target_win")
    truncation = _boolean(game.get("truncated"), prefix, "truncated")
    protocol_failure = game.get("protocol_failure") is not None
    winner = game.get("winner")
    if winner not in {"white", "black", None}:
        raise ValueError(f"{prefix} has invalid winner")
    draw = winner is None and not truncation and not protocol_failure

    terminal_flags = sum((selfmate, target_win, draw, truncation, protocol_failure))
    if terminal_flags != 1:
        raise ValueError(
            f"{prefix} does not map to exactly one outcome class "
            f"(found {terminal_flags})"
        )
    return Game(
        key=key,
        plies=plies,
        selfmate=selfmate,
        draw=draw,
        target_win=target_win,
        truncation=truncation,
        protocol_failure=protocol_failure,
    )


def _verify_aligned_keys(
    baseline: LoadedReport, candidate: LoadedReport
) -> None:
    baseline_keys = set(baseline.games)
    candidate_keys = set(candidate.games)
    if baseline_keys == candidate_keys:
        return
    missing_candidate = sorted(baseline_keys - candidate_keys)[:3]
    missing_baseline = sorted(candidate_keys - baseline_keys)[:3]
    raise ValueError(
        "report keys do not align; "
        f"missing from candidate: {missing_candidate}; "
        f"missing from baseline: {missing_baseline}"
    )


def _summarize(games: list[Game], max_plies: int) -> dict[str, Any]:
    overall = _summarize_stratum(games, max_plies)
    by_color = {
        color: _summarize_stratum(
            [game for game in games if game.key.target_color == color],
            max_plies,
        )
        for color in ("white", "black")
    }
    return {"overall": overall, "by_target_color": by_color}


def _summarize_stratum(games: list[Game], max_plies: int) -> dict[str, Any]:
    total = len(games)
    selfmate_plies = [game.plies for game in games if game.selfmate]
    restricted = [game.plies if game.selfmate else max_plies for game in games]
    counts = {
        "selfmates": sum(game.selfmate for game in games),
        "draws": sum(game.draw for game in games),
        "target_wins": sum(game.target_win for game in games),
        "truncations": sum(game.truncation for game in games),
        "protocol_failures": sum(game.protocol_failure for game in games),
    }
    return {
        "games": total,
        **counts,
        "selfmate_rate": counts["selfmates"] / total if total else None,
        "draw_rate": counts["draws"] / total if total else None,
        "target_win_rate": counts["target_wins"] / total if total else None,
        "truncation_rate": counts["truncations"] / total if total else None,
        "protocol_failure_rate": (
            counts["protocol_failures"] / total if total else None
        ),
        "conditional_selfmate_plies_mean": (
            statistics.fmean(selfmate_plies) if selfmate_plies else None
        ),
        "conditional_selfmate_plies_median": (
            statistics.median(selfmate_plies) if selfmate_plies else None
        ),
        "restricted_plies_mean": statistics.fmean(restricted) if restricted else None,
    }


def _paired_summary(
    baseline: list[Game], candidate: list[Game], max_plies: int
) -> dict[str, Any]:
    joint_deltas = [
        old.plies - new.plies
        for old, new in zip(baseline, candidate, strict=True)
        if old.selfmate and new.selfmate
    ]
    restricted_deltas = [
        (old.plies if old.selfmate else max_plies)
        - (new.plies if new.selfmate else max_plies)
        for old, new in zip(baseline, candidate, strict=True)
    ]
    return {
        "both_selfmate": sum(
            old.selfmate and new.selfmate
            for old, new in zip(baseline, candidate, strict=True)
        ),
        "candidate_only_selfmate": sum(
            not old.selfmate and new.selfmate
            for old, new in zip(baseline, candidate, strict=True)
        ),
        "baseline_only_selfmate": sum(
            old.selfmate and not new.selfmate
            for old, new in zip(baseline, candidate, strict=True)
        ),
        "neither_selfmate": sum(
            not old.selfmate and not new.selfmate
            for old, new in zip(baseline, candidate, strict=True)
        ),
        "joint_selfmate_mean_plies_improvement": (
            statistics.fmean(joint_deltas) if joint_deltas else None
        ),
        "joint_selfmate_median_plies_improvement": (
            statistics.median(joint_deltas) if joint_deltas else None
        ),
        "restricted_plies_mean_improvement": statistics.fmean(restricted_deltas),
    }


def _cluster_bootstrap(
    baseline: list[Game],
    candidate: list[Game],
    clusters: list[NDArray[np.int64]],
    *,
    max_plies: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    old_selfmate = np.asarray([game.selfmate for game in baseline], dtype=float)
    new_selfmate = np.asarray([game.selfmate for game in candidate], dtype=float)
    old_draw = np.asarray([game.draw for game in baseline], dtype=float)
    new_draw = np.asarray([game.draw for game in candidate], dtype=float)
    old_win = np.asarray([game.target_win for game in baseline], dtype=float)
    new_win = np.asarray([game.target_win for game in candidate], dtype=float)
    old_trunc = np.asarray([game.truncation for game in baseline], dtype=float)
    new_trunc = np.asarray([game.truncation for game in candidate], dtype=float)
    old_failure = np.asarray(
        [game.protocol_failure for game in baseline], dtype=float
    )
    new_failure = np.asarray(
        [game.protocol_failure for game in candidate], dtype=float
    )
    old_plies = np.asarray([game.plies for game in baseline], dtype=float)
    new_plies = np.asarray([game.plies for game in candidate], dtype=float)
    old_restricted = np.where(old_selfmate == 1.0, old_plies, max_plies)
    new_restricted = np.where(new_selfmate == 1.0, new_plies, max_plies)
    joint = (old_selfmate == 1.0) & (new_selfmate == 1.0)
    speed_delta = old_plies - new_plies

    difference_arrays = {
        "selfmate_rate_difference_pp": 100.0 * (new_selfmate - old_selfmate),
        "draw_rate_difference_pp": 100.0 * (new_draw - old_draw),
        "target_win_rate_difference_pp": 100.0 * (new_win - old_win),
        "truncation_rate_difference_pp": 100.0 * (new_trunc - old_trunc),
        "protocol_failure_rate_difference_pp": 100.0
        * (new_failure - old_failure),
        "restricted_plies_mean_improvement": old_restricted - new_restricted,
    }
    rng = np.random.default_rng(seed)
    sampled_clusters = rng.integers(
        0, len(clusters), size=(resamples, len(clusters))
    )
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {}

    for name, values in difference_arrays.items():
        cluster_sums = np.asarray([values[index].sum() for index in clusters])
        cluster_counts = np.asarray([index.size for index in clusters])
        samples = cluster_sums[sampled_clusters].sum(axis=1) / cluster_counts[
            sampled_clusters
        ].sum(axis=1)
        result[name] = _interval(values.mean(), samples, alpha)

    joint_cluster_values = [speed_delta[index[joint[index]]] for index in clusters]
    joint_sums = np.asarray([values.sum() for values in joint_cluster_values])
    joint_counts = np.asarray([values.size for values in joint_cluster_values])
    sampled_joint_counts = joint_counts[sampled_clusters].sum(axis=1)
    valid = sampled_joint_counts > 0
    if joint.any() and valid.any():
        mean_samples = joint_sums[sampled_clusters].sum(axis=1)[valid] / (
            sampled_joint_counts[valid]
        )
        median_samples = np.asarray(
            [
                np.median(
                    np.concatenate(
                        [joint_cluster_values[index] for index in sampled]
                    )
                )
                for sampled in sampled_clusters[valid]
            ]
        )
        result["joint_selfmate_mean_plies_improvement"] = _interval(
            speed_delta[joint].mean(), mean_samples, alpha
        )
        result["joint_selfmate_median_plies_improvement"] = _interval(
            np.median(speed_delta[joint]), median_samples, alpha
        )
    else:
        result["joint_selfmate_mean_plies_improvement"] = None
        result["joint_selfmate_median_plies_improvement"] = None
    return result


def _interval(
    estimate: float, samples: NDArray[np.float64], alpha: float
) -> dict[str, float]:
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "confidence_low": float(low),
        "confidence_high": float(high),
    }


def _mapping(value: Any, path: Path, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"report {path} field {field} must be an object")
    return value


def _integer(value: Any, path: Path, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"report {path} field {field} must be an integer")
    return value


def _boolean(value: Any, prefix: str, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{prefix} field {field} must be a boolean")
    return value


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        result = compare_reports(
            load_report(arguments.baseline),
            load_report(arguments.candidate),
            resamples=arguments.resamples,
            confidence=arguments.confidence,
            seed=arguments.seed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
