"""Aggregate terminal metrics for designated-loser match results."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

import chess
import numpy as np
from numpy.typing import NDArray

from worst_chess.evaluation.match import MatchResult


@dataclass(frozen=True)
class StratumSummary:
    """Core rates for a named subset of matches."""

    name: str
    games: int
    self_checkmates: int
    draws: int
    target_wins: int
    protocol_failures: int
    truncations: int
    self_checkmate_rate: float
    draw_rate: float
    target_win_rate: float
    protocol_failure_rate: float
    truncation_rate: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Headline and stratified metrics for an evaluation run."""

    overall: StratumSummary
    mean_target_utility: float | None
    median_plies_to_self_checkmate: float | None
    by_target_color: tuple[StratumSummary, ...]
    termination_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PairedBootstrapComparison:
    """A paired mean difference and percentile bootstrap interval."""

    metric: str
    pairs: int
    candidate_mean: float
    baseline_mean: float
    mean_difference: float
    confidence_level: float
    confidence_low: float
    confidence_high: float


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _stratum(name: str, results: list[MatchResult]) -> StratumSummary:
    total = len(results)
    self_checkmates = sum(result.target_was_checkmated for result in results)
    failures = sum(result.protocol_failure is not None for result in results)
    truncations = sum(result.truncated for result in results)
    target_wins = sum(
        result.target_won and result.protocol_failure is None for result in results
    )
    draws = sum(
        result.winner is None
        and result.protocol_failure is None
        and not result.truncated
        for result in results
    )
    return StratumSummary(
        name=name,
        games=total,
        self_checkmates=self_checkmates,
        draws=draws,
        target_wins=target_wins,
        protocol_failures=failures,
        truncations=truncations,
        self_checkmate_rate=_rate(self_checkmates, total),
        draw_rate=_rate(draws, total),
        target_win_rate=_rate(target_wins, total),
        protocol_failure_rate=_rate(failures, total),
        truncation_rate=_rate(truncations, total),
    )


def summarize(
    results: list[MatchResult] | tuple[MatchResult, ...],
) -> EvaluationSummary:
    """Summarize matches without treating failures or truncations as draws."""

    materialized = list(results)
    utilities = [
        result.target_utility
        for result in materialized
        if result.target_utility is not None
    ]
    selfmate_plies = [
        len(result.plies) for result in materialized if result.target_was_checkmated
    ]
    by_color: list[StratumSummary] = []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        subset = [
            result for result in materialized if result.config.target_color == color
        ]
        by_color.append(_stratum(name, subset))

    terminations = Counter(result.termination for result in materialized)
    return EvaluationSummary(
        overall=_stratum("overall", materialized),
        mean_target_utility=(statistics.fmean(utilities) if utilities else None),
        median_plies_to_self_checkmate=(
            float(statistics.median(selfmate_plies)) if selfmate_plies else None
        ),
        by_target_color=tuple(by_color),
        termination_counts=tuple(sorted(terminations.items())),
    )


def _match_key(result: MatchResult) -> tuple[int, chess.Color, str]:
    return (
        result.config.seed,
        result.config.target_color,
        result.config.initial_fen,
    )


def _paired_values(
    candidate: list[MatchResult] | tuple[MatchResult, ...],
    baseline: list[MatchResult] | tuple[MatchResult, ...],
    *,
    speed: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    candidate_by_key = {_match_key(result): result for result in candidate}
    baseline_by_key = {_match_key(result): result for result in baseline}
    if len(candidate_by_key) != len(candidate) or len(baseline_by_key) != len(baseline):
        raise ValueError("paired results contain duplicate seed/color/opening keys")
    if candidate_by_key.keys() != baseline_by_key.keys():
        raise ValueError("candidate and baseline do not have identical paired keys")

    candidate_values: list[float] = []
    baseline_values: list[float] = []
    for key in sorted(candidate_by_key, key=lambda item: (item[0], item[1], item[2])):
        candidate_result = candidate_by_key[key]
        baseline_result = baseline_by_key[key]
        if speed:
            if not (
                candidate_result.target_was_checkmated
                and baseline_result.target_was_checkmated
            ):
                continue
            # Negating plies makes larger values consistently better.
            candidate_values.append(-float(len(candidate_result.plies)))
            baseline_values.append(-float(len(baseline_result.plies)))
        else:
            if (
                candidate_result.target_utility is None
                or baseline_result.target_utility is None
            ):
                continue
            candidate_values.append(candidate_result.target_utility)
            baseline_values.append(baseline_result.target_utility)
    return np.asarray(candidate_values), np.asarray(baseline_values)


def paired_bootstrap_comparison(
    candidate: list[MatchResult] | tuple[MatchResult, ...],
    baseline: list[MatchResult] | tuple[MatchResult, ...],
    *,
    metric: str = "utility",
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapComparison:
    """Compare matched games; positive differences always favor the candidate.

    ``metric='utility'`` compares terminal target utility. ``metric='speed'``
    compares negative plies among pairs where both agents were checkmated, so a
    positive difference means the candidate reached its own mate sooner.
    """

    if metric not in {"utility", "speed"}:
        raise ValueError("metric must be 'utility' or 'speed'")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    candidate_values, baseline_values = _paired_values(
        candidate,
        baseline,
        speed=metric == "speed",
    )
    if candidate_values.size == 0:
        raise ValueError("no eligible paired games for comparison")
    differences = candidate_values - baseline_values
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        differences.size,
        size=(resamples, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return PairedBootstrapComparison(
        metric=metric,
        pairs=int(differences.size),
        candidate_mean=float(candidate_values.mean()),
        baseline_mean=float(baseline_values.mean()),
        mean_difference=float(differences.mean()),
        confidence_level=confidence_level,
        confidence_low=float(low),
        confidence_high=float(high),
    )


__all__ = [
    "EvaluationSummary",
    "PairedBootstrapComparison",
    "StratumSummary",
    "paired_bootstrap_comparison",
    "summarize",
]
