"""Test reachable loss-trajectory states against the synthetic loser league."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.opponent_model import StalemateAwareRandomReplySearchAgent
from worst_chess.agents.synthetic_loser import build_synthetic_loser_league
from worst_chess.evaluation.match import MatchConfig, MatchResult, play_match


@dataclass(frozen=True, slots=True)
class CurriculumPosition:
    fen: str
    target_color: chess.Color
    target_turns_before_loss: int
    source_game_id: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--max-source-games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def extract_curriculum_positions(
    paths: list[Path],
    *,
    distances: tuple[int, ...],
    max_source_games: int,
) -> tuple[CurriculumPosition, ...]:
    if not distances or any(distance < 1 for distance in distances):
        raise ValueError("distances must contain positive integers")
    if len(set(distances)) != len(distances):
        raise ValueError("distances must not contain duplicates")
    if max_source_games < 1:
        raise ValueError("max_source_games must be positive")

    selected: list[CurriculumPosition] = []
    source_losses = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            while source_losses < max_source_games and (
                game := chess.pgn.read_game(stream)
            ):
                target_name = game.headers.get("Target")
                if target_name not in {"white", "black"}:
                    continue
                target_color = target_name == "white"
                board = game.board()
                target_fens: list[str] = []
                for move in game.mainline_moves():
                    if board.turn == target_color:
                        target_fens.append(board.fen(en_passant="fen"))
                    board.push(move)
                if not (board.is_checkmate() and board.turn == target_color):
                    continue
                source_game_id = game.headers.get(
                    "Round",
                    f"{path.stem}-{source_losses}",
                )
                for distance in distances:
                    if distance <= len(target_fens):
                        selected.append(
                            CurriculumPosition(
                                fen=target_fens[-distance],
                                target_color=target_color,
                                target_turns_before_loss=distance,
                                source_game_id=source_game_id,
                            )
                        )
                source_losses += 1
            if source_losses >= max_source_games:
                break
    return tuple(selected)


def evaluate_curriculum(
    positions: tuple[CurriculumPosition, ...],
    *,
    checkpoint: Path,
    device: str,
    max_plies: int,
    seed: int,
) -> tuple[MatchResult, ...]:
    policy = NeuralAgent.from_checkpoint(
        checkpoint,
        device=device,
        agent_name="curriculum-v03",
    )
    target = StalemateAwareRandomReplySearchAgent(policy, top_k=12)
    opponent = build_synthetic_loser_league(target, salt="curriculum-league-v1")
    results: list[MatchResult] = []
    for index, position in enumerate(positions):
        config = MatchConfig(
            game_id=(
                f"curriculum-{index:04d}-"
                f"d{position.target_turns_before_loss}-"
                f"{position.source_game_id}"
            ),
            seed=seed + index,
            target_color=position.target_color,
            initial_fen=position.fen,
            max_plies=max_plies,
        )
        white, black = (
            (target, opponent)
            if position.target_color == chess.WHITE
            else (opponent, target)
        )
        results.append(play_match(white, black, config))
    return tuple(results)


def summarize_by_distance(
    positions: tuple[CurriculumPosition, ...],
    results: tuple[MatchResult, ...],
) -> dict[str, dict[str, int | float]]:
    if len(positions) != len(results):
        raise ValueError("positions and results must have equal length")
    grouped: dict[int, list[MatchResult]] = {}
    for position, result in zip(positions, results, strict=True):
        grouped.setdefault(position.target_turns_before_loss, []).append(result)

    summary: dict[str, dict[str, int | float]] = {}
    for distance, games in sorted(grouped.items()):
        counts: Counter[str] = Counter()
        loss_plies: list[int] = []
        for result in games:
            if result.target_was_checkmated:
                counts["losses"] += 1
                loss_plies.append(len(result.plies))
            elif result.target_won:
                counts["wins"] += 1
            elif result.truncated:
                counts["truncations"] += 1
            else:
                counts["draws"] += 1
        summary[str(distance)] = {
            "games": len(games),
            "losses": counts["losses"],
            "draws": counts["draws"],
            "wins": counts["wins"],
            "truncations": counts["truncations"],
            "mean_loss_plies": (
                sum(loss_plies) / len(loss_plies) if loss_plies else 0.0
            ),
        }
    return summary


def main() -> int:
    arguments = build_parser().parse_args()
    positions = extract_curriculum_positions(
        arguments.pgn,
        distances=tuple(arguments.distances),
        max_source_games=arguments.max_source_games,
    )
    if not positions:
        raise ValueError("no target-loss curriculum positions were found")
    results = evaluate_curriculum(
        positions,
        checkpoint=arguments.checkpoint,
        device=arguments.device,
        max_plies=arguments.max_plies,
        seed=arguments.seed,
    )
    summary = summarize_by_distance(positions, results)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "report.json").write_text(
        json.dumps(
            {
                "source_positions": len(positions),
                "distances": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (arguments.output / "games.pgn").write_text(
        "\n\n".join(result.pgn.rstrip() for result in results) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
