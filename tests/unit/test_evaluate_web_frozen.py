from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.evaluate_web_frozen import (  # noqa: E402
    generate_random_openings,
    play_game,
    random_move,
)
from worst_chess.agents.base import MoveContext  # noqa: E402
from worst_chess.agents.random import RandomAgent  # noqa: E402
from worst_chess.evaluation.openings import (  # noqa: E402
    generate_random_openings as research_openings,
)


class _NeverCalledWorker:
    def choose(self, board: chess.Board) -> chess.Move:
        del board
        raise AssertionError("a browser-terminal position must not reach the worker")


def test_frozen_web_openings_match_the_research_generator() -> None:
    assert generate_random_openings(5, 6, 20261221) == research_openings(
        5, 6, 20261221
    )


def test_frozen_web_random_opponent_matches_the_research_agent() -> None:
    board = chess.Board()
    context = MoveContext("same-game", 0, 17, chess.WHITE)

    assert random_move(board, context.game_id, context.seed, context.ply) == (
        RandomAgent().select_move(board, context)
    )


def test_frozen_web_evaluation_honors_browser_claim_draws() -> None:
    record, _, decisions, _ = play_game(
        _NeverCalledWorker(),
        game_id="claim-draw",
        seed=1,
        initial_fen="8/8/8/8/8/2k5/8/K6R w - - 100 1",
        target_color=chess.WHITE,
        max_plies=600,
    )

    assert record["termination"] == "fifty_moves"
    assert record["target_was_checkmated"] is False
    assert record["target_won"] is False
    assert decisions == 0
