from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.expand_selfmate_ancestors import (  # noqa: E402
    expand_report,
    quiet_predecessors,
    two_ply_ancestors,
)

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w KQ - 7 6"


def test_every_quiet_predecessor_replays_to_the_child() -> None:
    child = chess.Board(SELF_MATE_IN_ONE)

    predecessors = list(quiet_predecessors(child))

    assert predecessors
    for predecessor in predecessors:
        replayed = predecessor.board.copy(stack=False)
        assert predecessor.forward_move in replayed.legal_moves
        assert not replayed.is_capture(predecessor.forward_move)
        replayed.push(predecessor.forward_move)
        assert replayed.board_fen() == child.board_fen()
        assert replayed.turn == child.turn


def test_two_ply_ancestors_replay_both_moves() -> None:
    seed = chess.Board(SELF_MATE_IN_ONE)

    ancestors = two_ply_ancestors(seed)

    assert ancestors
    assert len({ancestor.board.fen() for ancestor in ancestors}) == len(ancestors)
    for ancestor in ancestors:
        replayed = ancestor.board.copy(stack=False)
        for move in ancestor.forward_moves:
            assert move in replayed.legal_moves
            assert not replayed.is_capture(move)
            replayed.push(move)
        assert replayed.board_fen() == seed.board_fen()
        assert replayed.turn == seed.turn


def test_expansion_reports_separately_when_proof_distance_increases() -> None:
    report = {
        "records": [
            {
                "fen": SELF_MATE_IN_ONE,
                "target_color": "white",
                "source": "test",
                "source_id": 1,
                "status": "proven",
                "attempts": [{"status": "proven", "forced_plies": 2}],
            }
        ]
    }

    result = expand_report(
        report,
        start=0,
        count=None,
        max_candidates_per_seed=1,
        max_extended_per_seed=1,
        node_budget=1_000,
        shuffle_seed=1,
    )

    assert result["summary"]["generated_ancestors"] > 0
    assert result["summary"]["searched"] == 1
    assert "proven_extended" in result["summary"]
    record = result["records"][0]
    assert record["max_plies"] == 4
    if record["status"] == "proven":
        assert record["distance_gain"] == record["forced_plies"] - 2
        if record["distance_gain"] > 0:
            assert record["lower_horizon_status"] in {
                "proven",
                "refuted",
                "unknown",
            }
    else:
        assert record["distance_gain"] is None
        assert record["distance_gain_verified"] is False
