from __future__ import annotations

import io
from pathlib import Path

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.agents.web import WebEngineAgent


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO('{"moveUci":"e2e4"}\n')
        self.stderr = io.StringIO()
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int) -> int:
        del timeout
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def test_web_agent_bridges_fen_to_a_legal_uci_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_directory = tmp_path / "web"
    executable = web_directory / "node_modules" / ".bin" / "tsx"
    worker = web_directory / "scripts" / "engine-jsonl.ts"
    executable.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True)
    executable.touch()
    worker.touch()
    process = _FakeProcess()

    monkeypatch.setattr(
        "worst_chess.agents.web.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    board = chess.Board()
    context = MoveContext("web-test", 0, 1, chess.WHITE)
    agent = WebEngineAgent(web_directory)
    move = agent.select_move(board, context)
    request = process.stdin.getvalue()
    agent.close()

    assert move == chess.Move.from_uci("e2e4")
    assert board == chess.Board()
    assert '"fen":"' in request
