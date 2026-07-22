"""Color-paired tournament scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from worst_chess.agents.base import Agent
from worst_chess.chess.outcomes import DrawPolicy
from worst_chess.evaluation.match import MatchConfig, MatchResult, play_match


@dataclass(frozen=True)
class TournamentConfig:
    """Inputs shared by a reproducible sequence of paired matches."""

    tournament_id: str
    pairs: int
    base_seed: int = 0
    opening_fens: tuple[str, ...] = (chess.STARTING_FEN,)
    draw_policy: DrawPolicy = DrawPolicy.NEVER_CLAIM
    max_plies: int = 1_024

    def __post_init__(self) -> None:
        if not self.tournament_id:
            raise ValueError("tournament_id must not be empty")
        if self.pairs < 1:
            raise ValueError("pairs must be positive")
        if not self.opening_fens:
            raise ValueError("opening_fens must not be empty")
        for fen in self.opening_fens:
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError(f"invalid opening FEN: {fen}")


def run_paired_tournament(
    target: Agent,
    opponent: Agent,
    config: TournamentConfig,
) -> tuple[MatchResult, ...]:
    """Run the target once as each color for every paired seed/opening."""

    results: list[MatchResult] = []
    for pair_index in range(config.pairs):
        seed = config.base_seed + pair_index
        initial_fen = config.opening_fens[pair_index % len(config.opening_fens)]
        for target_color, color_name in (
            (chess.WHITE, "white"),
            (chess.BLACK, "black"),
        ):
            game_id = f"{config.tournament_id}-p{pair_index:04d}-{color_name}"
            match_config = MatchConfig(
                game_id=game_id,
                seed=seed,
                target_color=target_color,
                initial_fen=initial_fen,
                draw_policy=config.draw_policy,
                max_plies=config.max_plies,
            )
            white, black = (
                (target, opponent)
                if target_color == chess.WHITE
                else (opponent, target)
            )
            results.append(play_match(white, black, match_config))
    return tuple(results)


__all__ = ["TournamentConfig", "run_paired_tournament"]

