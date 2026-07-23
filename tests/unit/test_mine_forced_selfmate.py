from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.mine_forced_selfmate import (  # noqa: E402
    extract_candidates,
    search_candidates,
)

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w KQ - 7 6"


def test_search_candidates_finds_a_proven_reachable_selfmate() -> None:
    candidates = [
        {
            "fen": SELF_MATE_IN_ONE,
            "target_color": "white",
            "source": "composition",
            "game_id": "s1",
            "ply": 10,
            "plies_before_observed_mate": 2,
        }
    ]

    result = search_candidates(
        candidates,
        start=0,
        count=None,
        horizons=(2, 4),
        node_budget=10_000,
    )

    assert result["summary"] == {"proven": 1, "refuted": 0, "unknown": 0}
    assert result["records"][0]["attempts"][0]["principal_variation"] == [
        "f3h4",
        "d8h4",
    ]


def test_extract_candidates_keeps_only_tail_positions_from_target_losses(
    tmp_path: Path,
) -> None:
    board = chess.Board()
    for san in [
        "f4",
        "e5",
        "g4",
        "Ke7",
        "Nc3",
        "Kf6",
        "Nf3",
        "Kg6",
        "Nd5",
        "Nh6",
        "Nh4+",
        "Qxh4#",
    ]:
        board.push_san(san)
    game = chess.pgn.Game.from_board(board)
    game.headers["Target"] = "white"
    game.headers["Round"] = "reachable-s1"
    path = tmp_path / "game.pgn"
    path.write_text(str(game) + "\n", encoding="utf-8")

    candidates = extract_candidates([path], tail_target_positions=1)

    assert len(candidates) == 1
    assert candidates[0]["fen"] == SELF_MATE_IN_ONE
    assert candidates[0]["plies_before_observed_mate"] == 2
    assert candidates[0]["observed_target_loss"] is True


def test_extract_candidates_can_include_reachable_non_losses(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    game = chess.pgn.Game.from_board(board)
    game.headers["Target"] = "white"
    game.headers["Round"] = "unfinished"
    path = tmp_path / "unfinished.pgn"
    path.write_text(str(game) + "\n", encoding="utf-8")

    default = extract_candidates([path], tail_target_positions=2)
    included = extract_candidates(
        [path],
        tail_target_positions=2,
        include_non_losses=True,
    )

    assert default == []
    assert len(included) == 1
    assert included[0]["observed_target_loss"] is False
    assert included[0]["plies_before_observed_mate"] is None
