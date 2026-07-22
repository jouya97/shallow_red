"""Minimal Universal Chess Interface adapter for worst-chess agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TextIO

import chess

from worst_chess.agents.base import Agent, MoveContext

ENGINE_NAME = "Worst Chess Ever"
ENGINE_AUTHOR = "worst-chess-ever contributors"


class UCICommandError(ValueError):
    """Raised when a UCI command cannot be parsed or applied."""


@dataclass
class UCISession:
    """Synchronous UCI session over an arbitrary agent.

    Neural inference is bounded and fast, so the intentionally small adapter
    answers each ``go`` command synchronously.  The position is kept inside the
    session and an agent always receives a copy, preventing a faulty agent from
    corrupting subsequent protocol commands.
    """

    agent: Agent
    output: TextIO
    engine_name: str = ENGINE_NAME
    engine_author: str = ENGINE_AUTHOR
    board: chess.Board = field(default_factory=chess.Board, init=False)
    game_number: int = field(default=0, init=False)

    def send(self, message: str) -> None:
        """Write and immediately flush one protocol line."""

        self.output.write(f"{message}\n")
        self.output.flush()

    def handle(self, command: str) -> bool:
        """Handle one command, returning ``False`` when the session should end."""

        line = command.strip()
        if not line:
            return True
        name, _, arguments = line.partition(" ")

        if name == "uci":
            self.send(f"id name {_one_line(self.engine_name)}")
            self.send(f"id author {_one_line(self.engine_author)}")
            self.send("uciok")
        elif name == "isready":
            self.send("readyok")
        elif name == "ucinewgame":
            self.game_number += 1
            self.board = chess.Board()
        elif name == "position":
            self._set_position(arguments)
        elif name == "go":
            self._go()
        elif name == "quit":
            return False
        elif name in {"debug", "setoption", "register", "stop", "ponderhit"}:
            # No configurable options or background search are exposed.
            pass
        else:
            self.send(f"info string ignored unknown command: {_one_line(line)}")
        return True

    def _set_position(self, arguments: str) -> None:
        tokens = arguments.split()
        if not tokens:
            raise UCICommandError("position requires 'startpos' or 'fen'")

        if tokens[0] == "startpos":
            candidate = chess.Board()
            index = 1
        elif tokens[0] == "fen":
            if len(tokens) < 7:
                raise UCICommandError("position fen requires all six FEN fields")
            fen = " ".join(tokens[1:7])
            try:
                candidate = chess.Board(fen)
            except ValueError as error:
                raise UCICommandError(f"invalid FEN: {error}") from error
            index = 7
        else:
            raise UCICommandError("position requires 'startpos' or 'fen'")

        if index < len(tokens):
            if tokens[index] != "moves":
                raise UCICommandError("expected 'moves' after base position")
            for move_text in tokens[index + 1 :]:
                try:
                    candidate.push_uci(move_text)
                except ValueError as error:
                    raise UCICommandError(
                        f"invalid move {move_text!r} in position sequence"
                    ) from error
        self.board = candidate

    def _go(self) -> None:
        if self.board.is_game_over(claim_draw=False):
            self.send("bestmove 0000")
            return

        context = MoveContext(
            game_id=f"uci-{self.game_number}",
            ply=self.board.ply(),
            seed=0,
            target_color=self.board.turn,
        )
        move = self.agent.select_move(self.board.copy(stack=True), context)
        if not isinstance(move, chess.Move) or move not in self.board.legal_moves:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise UCICommandError(f"agent returned illegal move: {detail}")
        self.send(f"bestmove {move.uci()}")


def run_uci(agent: Agent, input_stream: TextIO, output_stream: TextIO) -> None:
    """Run a UCI command loop until ``quit`` or end-of-file.

    Malformed commands and inference failures are reported using UCI ``info
    string`` records.  A failed ``go`` also emits the null move so controllers
    waiting for ``bestmove`` do not hang.
    """

    session = UCISession(agent=agent, output=output_stream)
    for command in input_stream:
        try:
            if not session.handle(command):
                break
        except Exception as error:  # Protocol boundary: remain alive for the GUI.
            session.send(f"info string error: {_one_line(str(error))}")
            if command.strip().partition(" ")[0] == "go":
                session.send("bestmove 0000")


def _one_line(value: str) -> str:
    return " ".join(value.splitlines())


__all__ = [
    "ENGINE_AUTHOR",
    "ENGINE_NAME",
    "UCICommandError",
    "UCISession",
    "run_uci",
]
