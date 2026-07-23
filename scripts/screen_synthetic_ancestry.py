"""Screen earlier decisions from decisive synthetic-loser games.

The search deliberately evaluates only a diverse shortlist at first.  This
keeps the expensive population rollouts focused on moves proposed by the
frozen policy, two cheap loser heuristics, and the move actually played.
Promising positions can be fed back through ``--screen-report`` with more
rollouts for confirmation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chess
import chess.pgn

from worst_chess.agents.base import MoveContext
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.opponent_model import (
    RandomReplySearchAgent,
    StalemateAwareRandomReplySearchAgent,
)
from worst_chess.agents.synthetic_loser import build_synthetic_loser_league
from worst_chess.training.rollout_teacher import (
    LexicographicRolloutScorer,
    RolloutConfig,
    RolloutMoveScore,
)

Outcome = Literal["loss", "win"]


@dataclass(frozen=True, slots=True)
class AncestryPosition:
    fen: str
    target_color: chess.Color
    source: str
    game_id: str
    source_outcome: Outcome
    actual_move: str
    target_turns_before_end: int
    plies_before_end: int


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    checkpoint: Path
    model_top_k: int
    heuristic_top_k: int
    random_reply_top_k: int
    continuation_top_k: int
    rollouts: int
    rollout_plies: int
    seed: int


_WORKER_CONFIG: ScreeningConfig | None = None
_WORKER_NEURAL: NeuralAgent | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pgn", type=Path, nargs="+")
    inputs.add_argument("--screen-report", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--outcome",
        choices=("loss", "win"),
        action="append",
        dest="outcomes",
        help="Terminal target outcome to include; repeatable (default: both).",
    )
    parser.add_argument("--tail-target-positions", type=int, default=12)
    parser.add_argument("--model-top-k", type=int, default=4)
    parser.add_argument("--heuristic-top-k", type=int, default=2)
    parser.add_argument("--random-reply-top-k", type=int, default=2)
    parser.add_argument("--continuation-top-k", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--rollout-plies", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def extract_decisive_positions(
    paths: list[Path],
    *,
    outcomes: set[Outcome],
    tail_target_positions: int,
) -> list[AncestryPosition]:
    if tail_target_positions < 1:
        raise ValueError("tail_target_positions must be positive")
    positions: list[AncestryPosition] = []
    seen: set[tuple[str, chess.Color, Outcome]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            while game := chess.pgn.read_game(source):
                target_header = game.headers.get("Target", "").lower()
                if target_header not in {"white", "black"}:
                    continue
                target_color = target_header == "white"
                board = game.board()
                target_turns: list[tuple[str, str, int]] = []
                for move in game.mainline_moves():
                    if board.turn == target_color:
                        target_turns.append(
                            (
                                board.fen(en_passant="fen"),
                                move.uci(),
                                board.ply(),
                            )
                        )
                    board.push(move)
                result = board.outcome(claim_draw=False)
                if result is None or result.winner is None:
                    continue
                source_outcome: Outcome = (
                    "win" if result.winner == target_color else "loss"
                )
                if source_outcome not in outcomes:
                    continue
                selected = target_turns[-tail_target_positions:]
                for reverse_index, (fen, actual_move, ply) in enumerate(
                    reversed(selected), start=1
                ):
                    key = (fen, target_color, source_outcome)
                    if key in seen:
                        continue
                    seen.add(key)
                    positions.append(
                        AncestryPosition(
                            fen=fen,
                            target_color=target_color,
                            source=path.name,
                            game_id=game.headers.get("Round", "unknown"),
                            source_outcome=source_outcome,
                            actual_move=actual_move,
                            target_turns_before_end=reverse_index,
                            plies_before_end=board.ply() - ply,
                        )
                    )
    return positions


def promising_positions(report: dict[str, Any]) -> list[AncestryPosition]:
    """Load only screened positions with at least one observed selfmate."""

    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("screen report records must be a list")
    positions: list[AncestryPosition] = []
    for record in records:
        candidates = record.get("candidates", [])
        if not any(candidate.get("selfmates", 0) > 0 for candidate in candidates):
            continue
        color = record.get("target_color")
        outcome = record.get("source_outcome")
        if color not in {"white", "black"} or outcome not in {"loss", "win"}:
            raise ValueError("screen report contains invalid target metadata")
        positions.append(
            AncestryPosition(
                fen=record["fen"],
                target_color=color == "white",
                source=record["source"],
                game_id=record["game_id"],
                source_outcome=outcome,
                actual_move=record["actual_move"],
                target_turns_before_end=int(record["target_turns_before_end"]),
                plies_before_end=int(record["plies_before_end"]),
            )
        )
    return positions


def shortlist_candidates(
    board: chess.Board,
    context: MoveContext,
    neural: NeuralAgent,
    *,
    actual_move: chess.Move,
    model_top_k: int,
    heuristic_top_k: int,
    random_reply_top_k: int,
) -> tuple[tuple[chess.Move, ...], dict[chess.Move, list[str]], str]:
    """Return a stable union of policy and loser-oriented proposals."""

    sources: dict[chess.Move, list[str]] = {}

    def add(move: chess.Move, label: str) -> None:
        sources.setdefault(move, []).append(label)

    model_ranked = neural.rank_moves(board, context, top_k=model_top_k)
    for item in model_ranked:
        add(item.move, "model")
    model_move = model_ranked[0].move

    legal = tuple(board.legal_moves)
    heuristic = HeuristicAgent()
    heuristic_ranked = sorted(
        legal,
        key=lambda move: (
            -heuristic.score_move(board, move, context.target_color),
            move.uci(),
        ),
    )[:heuristic_top_k]
    for move in heuristic_ranked:
        add(move, "heuristic")

    random_reply = RandomReplySearchAgent(None, top_k=64)
    reply_ranked = sorted(
        legal,
        key=lambda move: (
            -random_reply.score_move(board, move, context.target_color),
            move.uci(),
        ),
    )[:random_reply_top_k]
    for move in reply_ranked:
        add(move, "random_reply")

    add(actual_move, "actual")
    return tuple(sorted(sources, key=chess.Move.uci)), sources, model_move.uci()


def _initialize_worker(config: ScreeningConfig) -> None:
    global _WORKER_CONFIG, _WORKER_NEURAL
    _WORKER_CONFIG = config
    _WORKER_NEURAL = NeuralAgent.from_checkpoint(config.checkpoint, device="cpu")


def _screen_one(index_position: tuple[int, AncestryPosition]) -> dict[str, Any]:
    index, position = index_position
    if _WORKER_CONFIG is None or _WORKER_NEURAL is None:
        raise RuntimeError("screening worker was not initialized")
    config = _WORKER_CONFIG
    neural = _WORKER_NEURAL
    board = chess.Board(position.fen)
    if board.turn != position.target_color:
        raise ValueError("target must be on move in every ancestry position")
    actual_move = chess.Move.from_uci(position.actual_move)
    if actual_move not in board.legal_moves:
        raise ValueError(f"recorded actual move is illegal: {position.actual_move}")
    context = MoveContext(
        game_id=(
            f"synthetic-ancestry/{position.source}/{position.game_id}/"
            f"{position.target_turns_before_end}"
        ),
        ply=board.ply(),
        seed=config.seed + index,
        target_color=position.target_color,
    )
    candidates, sources, model_move = shortlist_candidates(
        board,
        context,
        neural,
        actual_move=actual_move,
        model_top_k=config.model_top_k,
        heuristic_top_k=config.heuristic_top_k,
        random_reply_top_k=config.random_reply_top_k,
    )
    target_policy = StalemateAwareRandomReplySearchAgent(
        neural,
        top_k=config.continuation_top_k,
    )
    scorer = LexicographicRolloutScorer(
        target_policy,
        build_synthetic_loser_league(neural, salt="ancestry-rollout-v1"),
        RolloutConfig(
            rollouts=config.rollouts,
            max_plies=config.rollout_plies,
            seed=config.seed,
        ),
    )
    scores = scorer.evaluate_candidates(board, context, candidates)
    ordered = sorted(scores, key=_rollout_sort_key)
    best = ordered[0]
    score_by_move = {score.move: score for score in scores}
    model_score = score_by_move[chess.Move.from_uci(model_move)]
    actual_score = score_by_move[actual_move]
    return {
        "index": index,
        "fen": position.fen,
        "target_color": "white" if position.target_color else "black",
        "source": position.source,
        "game_id": position.game_id,
        "source_outcome": position.source_outcome,
        "actual_move": position.actual_move,
        "model_move": model_move,
        "target_turns_before_end": position.target_turns_before_end,
        "plies_before_end": position.plies_before_end,
        "legal_moves": board.legal_moves.count(),
        "candidate_count": len(scores),
        "best_moves": [
            score.move.uci()
            for score in ordered
            if score.ranking_score == best.ranking_score
        ],
        "best_improves_over_model": best.ranking_score > model_score.ranking_score,
        "best_improves_over_actual": best.ranking_score > actual_score.ranking_score,
        "candidates": [_score_record(score, sources[score.move]) for score in ordered],
    }


def _rollout_sort_key(score: RolloutMoveScore) -> tuple[float, str]:
    return (-score.ranking_score, score.move.uci())


def _score_record(score: RolloutMoveScore, sources: list[str]) -> dict[str, Any]:
    return {
        "move": score.move.uci(),
        "sources": sources,
        "selfmates": score.selfmates,
        "selfmate_plies_sum": score.selfmate_plies_sum,
        "mean_selfmate_plies": (
            score.selfmate_plies_sum / score.selfmates if score.selfmates else None
        ),
        "target_wins": score.target_wins,
        "draws": score.draws,
        "truncations": score.truncations,
        "ranking_score": score.ranking_score,
    }


def screen_positions(
    positions: list[AncestryPosition],
    *,
    config: ScreeningConfig,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not positions:
        return _build_report([], config=config, elapsed_seconds=0.0)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    indexed = list(enumerate(positions))
    if workers == 1:
        _initialize_worker(config)
        for item in indexed:
            records.append(_screen_one(item))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            futures = [executor.submit(_screen_one, item) for item in indexed]
            for completed, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                print(
                    f"positions={completed}/{len(indexed)}",
                    flush=True,
                )
    records.sort(key=lambda record: record["index"])
    return _build_report(
        records,
        config=config,
        elapsed_seconds=time.perf_counter() - started,
    )


def _build_report(
    records: list[dict[str, Any]],
    *,
    config: ScreeningConfig,
    elapsed_seconds: float,
) -> dict[str, Any]:
    outcomes = Counter(record["source_outcome"] for record in records)
    promising = [
        record
        for record in records
        if any(candidate["selfmates"] > 0 for candidate in record["candidates"])
    ]
    return {
        "checkpoint": str(config.checkpoint),
        "config": {
            "model_top_k": config.model_top_k,
            "heuristic_top_k": config.heuristic_top_k,
            "random_reply_top_k": config.random_reply_top_k,
            "continuation_top_k": config.continuation_top_k,
            "rollouts": config.rollouts,
            "rollout_plies": config.rollout_plies,
            "seed": config.seed,
        },
        "elapsed_seconds": elapsed_seconds,
        "summary": {
            "positions": len(records),
            "loss_source_positions": outcomes["loss"],
            "win_source_positions": outcomes["win"],
            "positions_with_observed_selfmate": len(promising),
            "positions_best_improves_over_model": sum(
                record["best_improves_over_model"] for record in records
            ),
            "positions_best_improves_over_actual": sum(
                record["best_improves_over_actual"] for record in records
            ),
            "candidate_evaluations": sum(
                record["candidate_count"] for record in records
            ),
        },
        "records": records,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    positive_names = (
        "model_top_k",
        "heuristic_top_k",
        "random_reply_top_k",
        "continuation_top_k",
        "rollouts",
        "rollout_plies",
        "workers",
    )
    for name in positive_names:
        if getattr(arguments, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if arguments.pgn:
        selected_outcomes: set[Outcome] = set(arguments.outcomes or ("loss", "win"))
        positions = extract_decisive_positions(
            arguments.pgn,
            outcomes=selected_outcomes,
            tail_target_positions=arguments.tail_target_positions,
        )
    else:
        report = json.loads(arguments.screen_report.read_text(encoding="utf-8"))
        positions = promising_positions(report)
    config = ScreeningConfig(
        checkpoint=arguments.checkpoint,
        model_top_k=arguments.model_top_k,
        heuristic_top_k=arguments.heuristic_top_k,
        random_reply_top_k=arguments.random_reply_top_k,
        continuation_top_k=arguments.continuation_top_k,
        rollouts=arguments.rollouts,
        rollout_plies=arguments.rollout_plies,
        seed=arguments.seed,
    )
    result = screen_positions(positions, config=config, workers=arguments.workers)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
