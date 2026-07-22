from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.cli import build_parser
from worst_chess.uci import UCICommandError, UCISession, run_uci


@dataclass
class RecordingAgent:
    contexts: list[MoveContext] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "recording"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        self.contexts.append(context)
        self.positions.append(board.fen())
        move = min(board.legal_moves, key=chess.Move.uci)
        board.push(move)  # Deliberately mutate the copy supplied by the session.
        return move


class IllegalAgent:
    @property
    def name(self) -> str:
        return "illegal"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        return chess.Move.from_uci("a1a8")


def test_uci_handshake_readiness_and_quit() -> None:
    output = StringIO()

    run_uci(
        RecordingAgent(),
        StringIO("uci\nisready\nquit\nisready\n"),
        output,
    )

    assert output.getvalue().splitlines() == [
        "id name Worst Chess Ever",
        "id author worst-chess-ever contributors",
        "uciok",
        "readyok",
    ]


def test_position_moves_and_go_use_current_side_without_mutating_session() -> None:
    agent = RecordingAgent()
    output = StringIO()
    session = UCISession(agent=agent, output=output)

    session.handle("position startpos moves e2e4 e7e5")
    expected_fen = session.board.fen()
    session.handle("go movetime 100")

    assert output.getvalue() == "bestmove a2a3\n"
    assert session.board.fen() == expected_fen
    assert agent.positions == [expected_fen]
    assert agent.contexts[0].target_color == chess.WHITE
    assert agent.contexts[0].ply == 2


def test_position_accepts_full_fen_and_reports_terminal_null_move() -> None:
    output = StringIO()
    session = UCISession(agent=RecordingAgent(), output=output)

    session.handle("position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    session.handle("go")

    assert output.getvalue() == "bestmove 0000\n"


def test_invalid_position_is_atomic_and_protocol_loop_recovers() -> None:
    agent = RecordingAgent()
    output = StringIO()

    run_uci(
        agent,
        StringIO(
            "position startpos moves e2e4\n"
            "position startpos moves e2e5\n"
            "go\n"
            "quit\n"
        ),
        output,
    )

    lines = output.getvalue().splitlines()
    assert lines[0].startswith("info string error: invalid move 'e2e5'")
    assert lines[1] == "bestmove a7a5"
    expected = chess.Board()
    expected.push_uci("e2e4")
    assert agent.positions == [expected.fen()]


def test_bad_position_command_raises_clear_error_when_used_directly() -> None:
    session = UCISession(agent=RecordingAgent(), output=StringIO())

    with pytest.raises(UCICommandError, match="six FEN fields"):
        session.handle("position fen 8/8/8/8/8/8/8/8 w - -")


def test_illegal_agent_move_unblocks_controller_with_null_move() -> None:
    output = StringIO()

    run_uci(IllegalAgent(), StringIO("go\nquit\n"), output)

    assert output.getvalue().splitlines() == [
        "info string error: agent returned illegal move: a1a8",
        "bestmove 0000",
    ]


def test_cli_exposes_checkpoint_device_and_optional_search_for_uci() -> None:
    arguments = build_parser().parse_args(
        [
            "uci",
            "--checkpoint",
            "model.pt",
            "--device",
            "mps",
            "--search-stockfish",
            "/engine/stockfish",
            "--search-nodes",
            "32",
            "--search-top-k",
            "4",
        ]
    )

    assert arguments.command == "uci"
    assert str(arguments.checkpoint) == "model.pt"
    assert arguments.device == "mps"
    assert arguments.search_stockfish == "/engine/stockfish"
    assert arguments.search_nodes == 32
    assert arguments.search_top_k == 4
