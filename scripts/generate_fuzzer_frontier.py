"""Generate independent initial-board midgame seeds for the selfmate fuzzer."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.opponent_model import StalemateAwareRandomReplySearchAgent
from worst_chess.agents.synthetic_loser import (
    ExploringLoserAgent,
    build_synthetic_loser_league,
)
from worst_chess.evaluation.match import MatchConfig, play_match


@dataclass(frozen=True, slots=True)
class FrontierConfig:
    checkpoint: Path
    warmup_plies: int
    target_top_k: int
    target_exploration: float
    seed: int


_WORKER_CONFIG: FrontierConfig | None = None
_WORKER_NEURAL: NeuralAgent | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-plies", type=int, default=40)
    parser.add_argument("--target-top-k", type=int, default=12)
    parser.add_argument("--target-exploration", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def target_aligned_plies(
    minimum_plies: int,
    target_color: chess.Color,
) -> int:
    """Return at least ``minimum_plies`` and leave the target on move."""

    if minimum_plies < 1:
        raise ValueError("minimum_plies must be positive")
    target_moves_after_even = target_color == chess.WHITE
    if (minimum_plies % 2 == 0) == target_moves_after_even:
        return minimum_plies
    return minimum_plies + 1


def _initialize_worker(config: FrontierConfig) -> None:
    global _WORKER_CONFIG, _WORKER_NEURAL
    _WORKER_CONFIG = config
    _WORKER_NEURAL = NeuralAgent.from_checkpoint(config.checkpoint, device="cpu")


def _generate_one(index: int) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_NEURAL is None:
        raise RuntimeError("frontier worker was not initialized")
    config = _WORKER_CONFIG
    neural = _WORKER_NEURAL
    target_color = chess.WHITE if index % 2 == 0 else chess.BLACK
    game_id = f"fresh-{index:05d}-{'white' if target_color else 'black'}"
    target = ExploringLoserAgent(
        StalemateAwareRandomReplySearchAgent(
            neural,
            top_k=config.target_top_k,
        ),
        exploration_probability=config.target_exploration,
        salt=f"fresh-frontier-target/{game_id}",
    )
    opponent = build_synthetic_loser_league(
        neural,
        salt=f"fresh-frontier-opponent/{game_id}",
    )
    max_plies = target_aligned_plies(config.warmup_plies, target_color)
    match = play_match(
        target if target_color == chess.WHITE else opponent,
        opponent if target_color == chess.WHITE else target,
        MatchConfig(
            game_id=game_id,
            seed=config.seed + index,
            target_color=target_color,
            max_plies=max_plies,
        ),
    )
    if match.protocol_failure is not None:
        outcome = "protocol_failure"
    elif match.target_was_checkmated:
        outcome = "loss"
    elif match.target_won:
        outcome = "win"
    elif match.truncated:
        outcome = "frontier"
    else:
        outcome = "draw"
    node: dict[str, Any] | None = None
    if outcome == "frontier":
        board = chess.Board(match.final_fen)
        if board.turn != target_color:
            raise RuntimeError("warmup must leave the target on move")
        node = {
            "fen": board.fen(en_passant="fen"),
            "target_color": "white" if target_color else "black",
            "root_id": game_id,
            "lineage": game_id,
            "generation": 0,
            "pressure": 0.0,
            "immediate_mate_probability": 0.0,
            "sampled_target_wins": 0,
            "sampled_losses": 0,
        }
    return {
        "game_id": game_id,
        "target_color": "white" if target_color else "black",
        "outcome": outcome,
        "plies": len(match.plies),
        "termination": match.termination,
        "final_fen": match.final_fen,
        "node": node,
        "pgn": match.pgn if outcome in {"loss", "win"} else None,
    }


def generate_frontier(
    *,
    config: FrontierConfig,
    positions: int,
    start_index: int,
    workers: int,
) -> dict[str, Any]:
    if positions < 1 or workers < 1:
        raise ValueError("positions and workers must be positive")
    if start_index < 0:
        raise ValueError("start_index must be nonnegative")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        records = list(
            executor.map(
                _generate_one,
                range(start_index, start_index + positions),
                chunksize=1,
            )
        )
    counts = Counter(record["outcome"] for record in records)
    nodes = [record["node"] for record in records if record["node"] is not None]
    pgns = [record["pgn"] for record in records if record["pgn"] is not None]
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "requested_positions": positions,
            "start_index": start_index,
            "end_index_exclusive": start_index + positions,
            "frontier_positions": len(nodes),
            "losses": counts["loss"],
            "wins": counts["win"],
            "draws": counts["draw"],
            "protocol_failures": counts["protocol_failure"],
            "white_frontiers": sum(node["target_color"] == "white" for node in nodes),
            "black_frontiers": sum(node["target_color"] == "black" for node in nodes),
        },
        "records": [
            {key: value for key, value in record.items() if key not in {"node", "pgn"}}
            for record in records
        ],
        "frontier": nodes,
        "decisive_pgns": pgns,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    for name in ("positions", "warmup_plies", "target_top_k", "workers"):
        if getattr(arguments, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= arguments.target_exploration <= 1.0:
        raise ValueError("target-exploration must be in [0, 1]")
    result = generate_frontier(
        config=FrontierConfig(
            checkpoint=arguments.checkpoint,
            warmup_plies=arguments.warmup_plies,
            target_top_k=arguments.target_top_k,
            target_exploration=arguments.target_exploration,
            seed=arguments.seed,
        ),
        positions=arguments.positions,
        start_index=arguments.start_index,
        workers=arguments.workers,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "decisive_pgns"}
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.output / "frontier.jsonl").write_text(
        "".join(json.dumps(node, sort_keys=True) + "\n" for node in result["frontier"]),
        encoding="utf-8",
    )
    (arguments.output / "decisive-games.pgn").write_text(
        "\n\n".join(result["decisive_pgns"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
