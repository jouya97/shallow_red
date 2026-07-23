from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.build_proof_ranked_dataset import (  # noqa: E402
    label_proof_position,
    mirror_move,
)
from worst_chess.chess.actions import encode_move  # noqa: E402

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w KQ - 7 6"


def test_labels_every_move_and_mirrors_the_proven_policy_action() -> None:
    labeled = label_proof_position(
        {
            "fen": SELF_MATE_IN_ONE,
            "target_color": "white",
            "status": "proven",
            "forced_plies": 2,
            "root_source_id": 1,
        },
        node_budget=10_000,
        mirror=True,
    )

    assert len(labeled.positions) == 2
    assert labeled.report["proven"] == 1
    assert labeled.report["unknown"] == 0
    white, black = labeled.positions
    white_board = white.board()
    black_board = black.board()
    selfmating_move = chess.Move.from_uci("f3h4")
    assert white.best_actions == (encode_move(white_board, selfmating_move),)
    assert black.target_color == chess.BLACK
    assert black.best_actions == (
        encode_move(black_board, mirror_move(selfmating_move)),
    )
    assert white.value_target == black.value_target == 1.0
    assert white.trajectory_id == black.trajectory_id


def test_mirror_move_round_trips() -> None:
    move = chess.Move.from_uci("a7a8q")

    assert mirror_move(mirror_move(move)) == move
