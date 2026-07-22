"""Evaluate the exact TypeScript web policy on a frozen random-opponent suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.pgn


def generate_random_openings(count: int, plies: int, seed: int) -> tuple[str, ...]:
    """Mirror worst_chess.evaluation.openings without importing the ML CLI."""

    openings: list[str] = []
    seen: set[str] = set()
    attempt = 0
    while len(openings) < count and attempt < max(100, count * 100):
        board = chess.Board()
        for ply in range(plies):
            legal = sorted(board.legal_moves, key=chess.Move.uci)
            payload = f"opening-v1\0{seed}\0{attempt}\0{ply}\0{board.fen()}"
            digest = hashlib.sha256(payload.encode()).digest()
            board.push(legal[int.from_bytes(digest, "big") % len(legal)])
        attempt += 1
        if len(board.move_stack) != plies or board.is_game_over(claim_draw=False):
            continue
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            openings.append(fen)
    if len(openings) != count:
        raise RuntimeError(f"could generate only {len(openings)} unique openings")
    return tuple(openings)


class WebWorker:
    def __init__(self, web_directory: Path) -> None:
        executable = web_directory / "node_modules" / ".bin" / "tsx"
        script = web_directory / "scripts" / "engine-jsonl.ts"
        self.process = subprocess.Popen(
            [str(executable), str(script)],
            cwd=web_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def choose(self, board: chess.Board) -> chess.Move:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("web worker pipes are unavailable")
        self.process.stdin.write(json.dumps({"fen": board.fen()}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            detail = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"web worker stopped without a response: {detail}")
        response: Any = json.loads(line)
        if not isinstance(response, dict) or "error" in response:
            raise RuntimeError(f"web worker error: {response}")
        move = chess.Move.from_uci(response["moveUci"])
        if move not in board.legal_moves:
            raise RuntimeError(f"web worker returned illegal move {move.uci()}")
        return move

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __enter__(self) -> WebWorker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MoveWorker(Protocol):
    def choose(self, board: chess.Board) -> chess.Move: ...


def random_move(board: chess.Board, game_id: str, seed: int, ply: int) -> chess.Move:
    legal = sorted(board.legal_moves, key=chess.Move.uci)
    payload = f"worst-chess-random-v1\0{seed}\0{game_id}\0{ply}".encode()
    digest = hashlib.sha256(payload).digest()
    return legal[int.from_bytes(digest, "big") % len(legal)]


def play_game(
    worker: MoveWorker,
    *,
    game_id: str,
    seed: int,
    initial_fen: str,
    target_color: chess.Color,
    max_plies: int,
) -> tuple[dict[str, Any], str, int, float]:
    board = chess.Board(initial_fen)
    target_decisions = 0
    target_seconds = 0.0
    outcome = None
    while len(board.move_stack) < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        if board.turn == target_color:
            started = time.perf_counter()
            move = worker.choose(board)
            target_seconds += time.perf_counter() - started
            target_decisions += 1
        else:
            move = random_move(board, game_id, seed, len(board.move_stack))
        board.push(move)
    else:
        outcome = board.outcome(claim_draw=True)

    truncated = outcome is None
    winner = outcome.winner if outcome is not None else None
    termination = (
        outcome.termination.name.lower() if outcome is not None else "max_plies"
    )
    target_was_checkmated = bool(
        outcome is not None
        and outcome.termination is chess.Termination.CHECKMATE
        and winner != target_color
    )
    target_won = winner == target_color
    target_utility = (
        None
        if truncated
        else (-1.0 if target_won else 1.0 if winner is not None else 0.0)
    )
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Shallow Red web frozen evaluation"
    game.headers["Round"] = game_id
    game.headers["White"] = "web_distilled_v1" if target_color else "random"
    game.headers["Black"] = "random" if target_color else "web_distilled_v1"
    game.headers["Target"] = "white" if target_color else "black"
    game.headers["Seed"] = str(seed)
    record = {
        "game_id": game_id,
        "seed": seed,
        "target_color": "white" if target_color else "black",
        "plies": len(board.move_stack),
        "winner": None if winner is None else "white" if winner else "black",
        "termination": termination,
        "target_utility": target_utility,
        "target_was_checkmated": target_was_checkmated,
        "target_won": target_won,
        "protocol_failure": None,
        "truncated": truncated,
        "final_fen": board.fen(),
    }
    return record, str(game), target_decisions, target_seconds


def stratum(name: str, games: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(games)
    selfmates = sum(bool(game["target_was_checkmated"]) for game in games)
    draws = sum(
        not game["truncated"]
        and not game["target_was_checkmated"]
        and not game["target_won"]
        for game in games
    )
    wins = sum(bool(game["target_won"]) for game in games)
    truncations = sum(bool(game["truncated"]) for game in games)
    return {
        "name": name,
        "games": count,
        "self_checkmates": selfmates,
        "self_checkmate_rate": selfmates / count,
        "draws": draws,
        "draw_rate": draws / count,
        "target_wins": wins,
        "target_win_rate": wins / count,
        "protocol_failures": 0,
        "protocol_failure_rate": 0.0,
        "truncations": truncations,
        "truncation_rate": truncations / count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--tournament-id", default="smoke-stalemate-aware")
    parser.add_argument("--web-directory", type=Path, default=Path("web"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    openings = generate_random_openings(args.pairs, args.opening_plies, args.seed)
    pair_count = args.pair_count if args.pair_count is not None else args.pairs
    pair_stop = args.pair_start + pair_count
    if args.pair_start < 0 or pair_count < 1 or pair_stop > args.pairs:
        raise ValueError("requested pair shard is outside the frozen suite")
    records: list[dict[str, Any]] = []
    pgns: list[str] = []
    target_decisions = 0
    target_seconds = 0.0
    started = time.perf_counter()
    with WebWorker(args.web_directory.resolve()) as worker:
        for pair_index in range(args.pair_start, pair_stop):
            initial_fen = openings[pair_index]
            seed = args.seed + pair_index
            for color, color_name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
                game_id = f"{args.tournament_id}-p{pair_index:04d}-{color_name}"
                record, pgn, decisions, seconds = play_game(
                    worker,
                    game_id=game_id,
                    seed=seed,
                    initial_fen=initial_fen,
                    target_color=color,
                    max_plies=args.max_plies,
                )
                records.append(record)
                pgns.append(pgn)
                target_decisions += decisions
                target_seconds += seconds
                if len(records) % 2 == 0:
                    current = stratum("progress", records)
                    print(
                        f"games={len(records)} losses={current['self_checkmates']} "
                        f"draws={current['draws']} wins={current['target_wins']} "
                        f"unresolved={current['truncations']}",
                        flush=True,
                    )

    mate_plies = [
        game["plies"] for game in records if game["target_was_checkmated"]
    ]
    utilities = [
        game["target_utility"]
        for game in records
        if game["target_utility"] is not None
    ]
    overall = stratum("overall", records)
    colors = [
        stratum(color, [game for game in records if game["target_color"] == color])
        for color in ("white", "black")
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "tournament": {
            "tournament_id": args.tournament_id,
            "pairs": pair_count,
            "suite_pairs": args.pairs,
            "pair_start": args.pair_start,
            "base_seed": args.seed,
            "opening_fens": list(openings[args.pair_start : pair_stop]),
            "draw_policy": "claim_available",
            "max_plies": args.max_plies,
        },
        "engine": {
            "name": "web_distilled_v1",
            "target_decisions": target_decisions,
            "target_decision_seconds": target_seconds,
            "mean_target_decision_ms": 1000 * target_seconds / target_decisions,
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": {
            "overall": overall,
            "by_target_color": colors,
            "mean_target_utility": statistics.fmean(utilities),
            "median_plies_to_self_checkmate": statistics.median(mate_plies),
            "termination_counts": dict(
                Counter(game["termination"] for game in records)
            ),
        },
        "games": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "games.pgn").write_text("\n\n".join(pgns) + "\n", encoding="utf-8")
    print(json.dumps(overall, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
