from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy
import pytest

from worst_chess import cli
from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.tablebase import SyzygyLosingAgent


class FallbackAgent:
    def __init__(self, move: chess.Move | None = None, *, mutate: bool = False) -> None:
        self.move = move
        self.mutate = mutate
        self.calls = 0

    @property
    def name(self) -> str:
        return "fallback"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        self.calls += 1
        move = self.move or min(board.legal_moves, key=chess.Move.uci)
        if self.mutate:
            board.push(move)
        return move


class FakeTablebase:
    def __init__(
        self,
        wdl: dict[str, int | None],
        dtz: dict[str, int | None] | None = None,
    ) -> None:
        self.wdl = wdl
        self.dtz = dtz or {}
        self.closed = False

    def get_wdl(
        self, board: chess.Board, default: int | None = None
    ) -> int | None:
        return self.wdl.get(board.peek().uci(), default)

    def get_dtz(
        self, board: chess.Board, default: int | None = None
    ) -> int | None:
        return self.dtz.get(board.peek().uci(), default)

    def close(self) -> None:
        self.closed = True


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("syzygy-test", 0, 3, color)


def _install_tablebase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake: FakeTablebase,
) -> Path:
    directory = tmp_path / "syzygy"
    directory.mkdir()
    monkeypatch.setattr(
        chess.syzygy,
        "open_tablebase",
        lambda path, max_fds: fake,
    )
    return directory


def test_syzygy_agent_maximizes_opponent_standard_wdl_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = chess.Board()
    desired = chess.Move.from_uci("e2e4")
    values = {move.uci(): -2 for move in board.legal_moves}
    values["d2d4"] = 0
    values[desired.uci()] = 2
    fake = FakeTablebase(values)
    directory = _install_tablebase(monkeypatch, tmp_path, fake)
    fallback = FallbackAgent()
    agent = SyzygyLosingAgent(directory, fallback)  # type: ignore[arg-type]
    original = board.fen()

    selected = agent.select_move(board, _context())

    assert selected == desired
    assert fallback.calls == 0
    assert board.fen() == original
    assert agent.tablebase_available
    agent.close()
    assert fake.closed


def test_syzygy_uses_dtz_only_as_same_wdl_progress_tiebreak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = chess.Board()
    values = {move.uci(): 2 for move in board.legal_moves}
    dtz = {move.uci(): 50 for move in board.legal_moves}
    dtz["e2e4"] = 12
    dtz["d2d4"] = 4
    fake = FakeTablebase(values, dtz)
    directory = _install_tablebase(monkeypatch, tmp_path, fake)
    agent = SyzygyLosingAgent(directory, FallbackAgent())  # type: ignore[arg-type]

    assert agent.select_move(board, _context()) == chess.Move.from_uci("d2d4")


def test_partial_or_missing_tablebase_falls_back_as_a_whole(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = chess.Board()
    fallback_move = chess.Move.from_uci("g1f3")
    fallback = FallbackAgent(fallback_move, mutate=True)
    values = {move.uci(): 2 for move in board.legal_moves}
    values["a2a3"] = None
    fake = FakeTablebase(values)
    directory = _install_tablebase(monkeypatch, tmp_path, fake)
    original = board.fen()

    partial = SyzygyLosingAgent(directory, fallback)  # type: ignore[arg-type]
    assert partial.select_move(board, _context()) == fallback_move
    assert fallback.calls == 1
    assert board.fen() == original

    missing = SyzygyLosingAgent(
        tmp_path / "does-not-exist", fallback  # type: ignore[arg-type]
    )
    assert missing.select_move(board, _context()) == fallback_move
    assert fallback.calls == 2
    assert not missing.tablebase_available


def test_syzygy_rejects_wrong_role_terminal_and_bad_fallback(
    tmp_path: Path,
) -> None:
    agent = SyzygyLosingAgent(None, FallbackAgent())  # type: ignore[arg-type]
    with pytest.raises(AgentError, match="target color"):
        agent.select_move(chess.Board(), _context(chess.BLACK))
    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    with pytest.raises(AgentError, match="terminal"):
        agent.select_move(terminal, _context(chess.BLACK))

    bad = SyzygyLosingAgent(
        None,
        FallbackAgent(chess.Move.from_uci("a1a8")),  # type: ignore[arg-type]
    )
    with pytest.raises(AgentError, match="fallback.*illegal"):
        bad.select_move(chess.Board(), _context())


def test_terminal_target_mate_is_scored_as_opponent_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    fake = FakeTablebase(
        {move.uci(): 0 for move in board.legal_moves},
        {move.uci(): 0 for move in board.legal_moves},
    )
    directory = _install_tablebase(monkeypatch, tmp_path, fake)
    agent = SyzygyLosingAgent(directory, FallbackAgent())  # type: ignore[arg-type]

    scores = agent.evaluate_moves(board, _context())

    assert scores is not None
    target_mates = []
    for score in scores:
        after = board.copy(stack=False)
        after.push(score.move)
        if after.is_checkmate():
            target_mates.append(score)
    assert target_mates
    assert all(score.opponent_wdl == -2 for score in target_mates)


def test_smoke_cli_accepts_optional_local_tablebase(tmp_path: Path) -> None:
    arguments = cli.build_parser().parse_args(
        ["smoke", "--tablebase", str(tmp_path / "syzygy")]
    )

    assert arguments.tablebase == tmp_path / "syzygy"
