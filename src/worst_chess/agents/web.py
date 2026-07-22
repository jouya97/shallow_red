"""Adapter for evaluating the exact lightweight TypeScript web policy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any

import chess

from worst_chess.agents.base import AgentError, MoveContext


class WebEngineAgent:
    """Keep the browser engine in a persistent JSON-lines subprocess."""

    def __init__(self, web_directory: str | Path | None = None) -> None:
        repository = Path(__file__).resolve().parents[3]
        self._web_directory = Path(web_directory or repository / "web").resolve()
        executable = self._web_directory / "node_modules" / ".bin" / "tsx"
        worker = self._web_directory / "scripts" / "engine-jsonl.ts"
        if not executable.is_file():
            raise AgentError(
                f"web engine runtime is missing at {executable}; run npm install in "
                f"{self._web_directory}"
            )
        if not worker.is_file():
            raise AgentError(f"web engine worker is missing at {worker}")

        try:
            self._process: subprocess.Popen[str] = subprocess.Popen(
                [str(executable), str(worker)],
                cwd=self._web_directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise AgentError(f"could not start the web engine: {error}") from error

    @property
    def name(self) -> str:
        return "web_distilled_v1"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        if board.is_game_over(claim_draw=False):
            raise AgentError("WebEngineAgent cannot move from a terminal position")
        if self._process.poll() is not None:
            raise AgentError(self._worker_failure("web engine worker exited"))
        if self._process.stdin is None or self._process.stdout is None:
            raise AgentError("web engine worker has no input/output pipes")

        request = json.dumps({"fen": board.fen()}, separators=(",", ":"))
        try:
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise AgentError(self._worker_failure(str(error))) from error
        if not response_line:
            raise AgentError(self._worker_failure("web engine returned no response"))

        try:
            response: Any = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise AgentError("web engine returned invalid JSON") from error
        if not isinstance(response, dict):
            raise AgentError("web engine returned a non-object response")
        if "error" in response:
            raise AgentError(f"web engine rejected the position: {response['error']}")
        move_uci = response.get("moveUci")
        if not isinstance(move_uci, str):
            raise AgentError("web engine response has no moveUci")
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as error:
            raise AgentError(f"web engine returned invalid UCI: {move_uci}") from error
        if move not in board.legal_moves:
            raise AgentError(f"web engine returned illegal move: {move_uci}")
        return move

    def _worker_failure(self, prefix: str) -> str:
        detail = ""
        if self._process.poll() is not None and self._process.stderr is not None:
            detail = self._process.stderr.read().strip()
        return f"{prefix}{f': {detail}' if detail else ''}"

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> WebEngineAgent:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = ["WebEngineAgent"]
