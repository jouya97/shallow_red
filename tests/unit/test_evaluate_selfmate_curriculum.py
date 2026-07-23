from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.evaluate_selfmate_curriculum import (  # noqa: E402
    extract_curriculum_positions,
)


def test_extracts_requested_target_turn_distances_from_losses(tmp_path: Path) -> None:
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    game = chess.pgn.Game.from_board(board)
    game.headers["Target"] = "white"
    game.headers["Round"] = "fools-mate"
    path = tmp_path / "loss.pgn"
    path.write_text(str(game) + "\n", encoding="utf-8")

    positions = extract_curriculum_positions(
        [path],
        distances=(1, 2),
        max_source_games=1,
    )

    assert [item.target_turns_before_loss for item in positions] == [1, 2]
    assert all(item.target_color == chess.WHITE for item in positions)
    assert positions[0].source_game_id == "fools-mate"
    assert chess.Board(positions[0].fen).turn == chess.WHITE
