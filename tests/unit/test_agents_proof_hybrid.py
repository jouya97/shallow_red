from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.agents.proof_hybrid import (
    ProofGuidedSelfmateAgent,
    SelfmateProofBook,
)
from worst_chess.cli import _proof_guided_agent, build_parser
from worst_chess.objective.proof_search import ProofSearchConfig

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w KQ - 7 6"


class FixedAgent:
    name = "fixed"

    def __init__(self, move: chess.Move) -> None:
        self.move = move
        self.calls = 0

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        self.calls += 1
        return self.move


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext(game_id="proof-test", ply=0, seed=1, target_color=color)


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "fen": SELF_MATE_IN_ONE,
                        "status": "proven",
                        "target_color": "white",
                        "forced_plies": 2,
                        "principal_variation": ["f3h4", "d8h4"],
                        "root_source_id": 123,
                    },
                    {
                        "fen": chess.STARTING_FEN,
                        "status": "refuted",
                        "target_color": "white",
                        "forced_plies": None,
                        "principal_variation": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_book_loads_proven_records_and_safe_color_mirrors(tmp_path: Path) -> None:
    report = tmp_path / "proof.json"
    _write_report(report)

    book = SelfmateProofBook.from_reports((report,))
    board = chess.Board(SELF_MATE_IN_ONE)
    mirrored = board.mirror()

    entry = book.lookup(board)
    mirrored_entry = book.lookup(mirrored)
    assert len(book) == 2
    assert entry is not None and entry.move.uci() == "f3h4"
    assert entry.forced_plies == 2
    assert mirrored_entry is not None and mirrored_entry.move.uci() == "f6h5"
    assert mirrored_entry.move in mirrored.legal_moves


def test_book_override_is_exact_and_does_not_call_fallback(tmp_path: Path) -> None:
    report = tmp_path / "proof.json"
    _write_report(report)
    board = chess.Board(SELF_MATE_IN_ONE)
    fallback = FixedAgent(chess.Move.from_uci("d5c7"))
    agent = ProofGuidedSelfmateAgent(
        fallback,
        book=SelfmateProofBook.from_reports((report,)),
    )
    original = board.fen()

    assert agent.select_move(board, _context()) == chess.Move.from_uci("f3h4")
    assert fallback.calls == 0
    assert agent.stats.book_hits == 1
    assert board.fen() == original


def test_live_search_finds_unbooked_forced_selfmate_and_caches_it() -> None:
    board = chess.Board(SELF_MATE_IN_ONE)
    fallback = FixedAgent(chess.Move.from_uci("d5c7"))
    agent = ProofGuidedSelfmateAgent(
        fallback,
        search_config=ProofSearchConfig(max_plies=2, node_budget=10_000),
    )

    assert agent.select_move(board, _context()) == chess.Move.from_uci("f3h4")
    assert agent.select_move(board, _context()) == chess.Move.from_uci("f3h4")
    assert fallback.calls == 0
    assert agent.stats.search_hits == 2
    assert agent.stats.decision_cache_hits == 1
    assert agent.stats.search_nodes > 0


def test_unknown_live_search_falls_back_without_claiming_a_proof() -> None:
    board = chess.Board()
    fallback = FixedAgent(chess.Move.from_uci("e2e4"))
    agent = ProofGuidedSelfmateAgent(
        fallback,
        search_config=ProofSearchConfig(max_plies=4, node_budget=1),
    )

    assert agent.select_move(board, _context()) == chess.Move.from_uci("e2e4")
    assert agent.stats.search_unknowns == 1
    assert agent.stats.search_hits == 0
    assert agent.stats.fallbacks == 1


def test_rejects_a_report_whose_line_does_not_end_in_selfmate(tmp_path: Path) -> None:
    report = tmp_path / "bad.json"
    _write_report(report)
    raw = json.loads(report.read_text(encoding="utf-8"))
    raw["records"][0]["principal_variation"] = ["d5c7", "h8g8"]
    report.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="does not selfmate"):
        SelfmateProofBook.from_reports((report,))


def test_cli_configures_book_and_live_search(tmp_path: Path) -> None:
    report = tmp_path / "proof.json"
    _write_report(report)
    arguments = build_parser().parse_args(
        [
            "smoke",
            "--proof-report",
            str(report),
            "--selfmate-search-plies",
            "4",
            "--selfmate-search-nodes",
            "1234",
        ]
    )
    fallback = FixedAgent(chess.Move.from_uci("e2e4"))

    agent = _proof_guided_agent(
        fallback,
        arguments.proof_report,
        search_plies=arguments.selfmate_search_plies,
        search_nodes=arguments.selfmate_search_nodes,
    )

    assert isinstance(agent, ProofGuidedSelfmateAgent)
    assert len(agent.book) == 2
    assert agent.search_config == ProofSearchConfig(max_plies=4, node_budget=1234)
