from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.finalize_synthetic_ancestry_dataset import (  # noqa: E402
    decode_rollout_score,
    finalize_dataset,
    safety_first_score,
)
from worst_chess.agents.base import MoveContext  # noqa: E402
from worst_chess.chess.actions import encode_move  # noqa: E402
from worst_chess.training.ranked_dataset import rank_position  # noqa: E402
from worst_chess.training.rollout_teacher import (  # noqa: E402
    RolloutConfig,
    rollout_ranking_score,
)


def _score(
    outcome: tuple[int, int, int, int, int],
    config: RolloutConfig,
) -> float:
    selfmates, plies_sum, wins, _draws, truncations = outcome
    return rollout_ranking_score(
        selfmates=selfmates,
        selfmate_plies_sum=plies_sum,
        target_wins=wins,
        truncations=truncations,
        config=config,
    )


def test_decodes_exact_rollout_scores_and_prioritizes_zero_wins() -> None:
    config = RolloutConfig(rollouts=4, max_plies=120)
    risky = (1, 22, 1, 0, 2)
    safe = (0, 0, 0, 0, 4)

    assert decode_rollout_score(_score(risky, config), config) == risky
    assert decode_rollout_score(_score(safe, config), config) == safe
    assert safety_first_score(safe, config) > safety_first_score(risky, config)


def test_finalize_changes_only_safety_ranking() -> None:
    config = RolloutConfig(rollouts=4, max_plies=120)
    board = chess.Board()
    risky_move = chess.Move.from_uci("e2e4")
    safe_move = chess.Move.from_uci("d2d4")
    risky_score = _score((1, 22, 1, 0, 2), config)
    safe_score = _score((0, 0, 0, 0, 4), config)

    def scorer(
        scoring_board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        del context
        return {
            move: risky_score if move == risky_move else safe_score
            for move in scoring_board.legal_moves
        }

    position = rank_position(
        board,
        target_color=chess.WHITE,
        scorer=scorer,
        context=MoveContext("safety", 0, 1, chess.WHITE),
        source_id="rollout",
        trajectory_id="safety-game",
        value_target=-1.0,
    )
    manifest = {
        "records": [
            {
                "fen": board.fen(en_passant="fen"),
                "target_color": "white",
                "seed_kind": "observed-win-safety",
            }
        ]
    }

    finalized, report = finalize_dataset((position,), manifest, config=config)

    ranks = {target.action: target.rank for target in finalized[0].move_targets}
    assert ranks[encode_move(board, risky_move)] > 1
    assert ranks[encode_move(board, safe_move)] == 1
    assert report["safety_best_changed"] == 1
