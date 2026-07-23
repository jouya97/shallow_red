from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.build_synthetic_ancestry_dataset import (  # noqa: E402
    build_ranked_seeds,
    select_training_seeds,
)


def _record(
    *,
    fen: str,
    outcome: str,
    turns_before_end: int,
    selfmates: int,
) -> dict[str, object]:
    return {
        "fen": fen,
        "target_color": "white",
        "source": "games.pgn",
        "game_id": f"{outcome}-{turns_before_end}",
        "source_outcome": outcome,
        "actual_move": "e2e4",
        "model_move": "d2d4",
        "target_turns_before_end": turns_before_end,
        "plies_before_end": turns_before_end * 2,
        "candidates": [{"move": "e2e4", "selfmates": selfmates}],
    }


def test_selects_confirmed_losses_and_only_final_win_safety_turns() -> None:
    fen = chess.Board().fen(en_passant="fen")
    positive = _record(fen=fen, outcome="loss", turns_before_end=2, selfmates=3)
    unconfirmed = _record(
        fen=fen.replace(" w ", " b "),
        outcome="loss",
        turns_before_end=3,
        selfmates=0,
    )
    win_board = chess.Board()
    win_board.push_uci("e2e4")
    win_board.push_uci("e7e5")
    win_fen = win_board.fen(en_passant="fen")
    final_win = _record(
        fen=win_fen,
        outcome="win",
        turns_before_end=1,
        selfmates=0,
    )
    earlier_win = _record(
        fen=win_fen,
        outcome="win",
        turns_before_end=2,
        selfmates=0,
    )
    recovered_board = chess.Board()
    recovered_board.push_uci("d2d4")
    recovered_board.push_uci("d7d5")
    recovered_win = _record(
        fen=recovered_board.fen(en_passant="fen"),
        outcome="win",
        turns_before_end=3,
        selfmates=1,
    )

    selected = select_training_seeds(
        {"records": [final_win, earlier_win]},
        {"records": [positive, unconfirmed, recovered_win]},
        win_tail_target_positions=1,
    )

    assert [record["seed_kind"] for record in selected] == [
        "confirmed-loss-steering",
        "recovered-win-steering",
        "observed-win-safety",
    ]
    assert [record["value_target"] for record in selected] == [1.0, None, -1.0]


def test_build_ranked_seeds_covers_every_legal_move() -> None:
    fen = chess.Board().fen(en_passant="fen")
    records = [
        {
            **_record(
                fen=fen,
                outcome="loss",
                turns_before_end=1,
                selfmates=1,
            ),
            "seed_kind": "confirmed-loss-steering",
            "value_target": 1.0,
        }
    ]

    positions = build_ranked_seeds(records)

    assert len(positions) == 1
    assert len(positions[0].move_targets) == chess.Board().legal_moves.count()
    assert positions[0].value_target == 1.0
