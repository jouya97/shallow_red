from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chess
import chess.engine
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.stockfish import ReverseStockfishAgent, StockfishAgent


class FakeEngine:
    def __init__(self, score_for: Callable[[chess.Move], int]) -> None:
        self.score_for = score_for
        self.moves: list[chess.Move] = []
        self.configured: dict[str, int] | None = None
        self.quit_called = False
        self.played_games: list[object] = []

    def configure(self, options: dict[str, int]) -> None:
        self.configured = options

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        *,
        game: object,
        root_moves: list[chess.Move],
        info: int,
    ) -> dict[str, Any]:
        del game, limit, info
        move = root_moves[0]
        self.moves.append(move)
        return {
            "score": chess.engine.PovScore(
                chess.engine.Cp(self.score_for(move)), board.turn
            )
        }

    def quit(self) -> None:
        self.quit_called = True

    def play(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        *,
        game: object,
    ) -> chess.engine.PlayResult:
        del limit
        self.played_games.append(game)
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext(game_id="stockfish", ply=0, seed=2, target_color=color)


def test_reverse_stockfish_scores_every_legal_root_and_selects_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wanted = chess.Move.from_uci("e2e4")
    fake = FakeEngine(lambda move: -400 if move == wanted else 200)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)
    board = chess.Board()
    original_fen = board.fen()
    legal = set(board.legal_moves)

    agent = ReverseStockfishAgent("fake-stockfish", nodes=123)
    selected = agent.select_move(board, _context())

    assert selected == wanted
    assert set(fake.moves) == legal
    assert len(fake.moves) == len(legal)
    assert fake.configured == {"Threads": 1, "Hash": 16}
    assert board.fen() == original_fen
    agent.close()
    assert fake.quit_called


def test_reverse_stockfish_exposes_ranked_loser_scores_for_root_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wanted = chess.Move.from_uci("e2e4")
    other = chess.Move.from_uci("d2d4")
    fake = FakeEngine(lambda move: -600 if move == wanted else 300)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)
    board = chess.Board()
    agent = ReverseStockfishAgent("fake-stockfish", nodes=12)

    scores = agent.evaluate_moves(board, _context(), root_moves=[other, wanted])

    assert [item.move for item in scores] == [wanted, other]
    assert scores[0].ranking_score > scores[1].ranking_score
    assert scores[0].loser_value > scores[1].loser_value
    assert all(-1.0 <= item.loser_value <= 1.0 for item in scores)
    assert set(fake.moves) == {wanted, other}


def test_reverse_stockfish_rejects_invalid_root_subsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEngine(lambda move: 0)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)
    board = chess.Board()
    legal = next(iter(board.legal_moves))
    agent = ReverseStockfishAgent("fake-stockfish", nodes=12)

    with pytest.raises(ValueError, match="must not be empty"):
        agent.evaluate_moves(board, _context(), root_moves=[])
    with pytest.raises(ValueError, match="duplicates"):
        agent.evaluate_moves(board, _context(), root_moves=[legal, legal])
    with pytest.raises(ValueError, match="only legal"):
        agent.evaluate_moves(
            board,
            _context(),
            root_moves=[chess.Move.from_uci("e2e5")],
        )


def test_reverse_stockfish_uses_target_point_of_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Scores are reported from White's POV.  For a Black target the largest
    # White score is the smallest conventional outcome for Black.
    wanted = chess.Move.from_uci("d2d4")
    fake = FakeEngine(lambda move: 500 if move == wanted else -100)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)

    agent = ReverseStockfishAgent("fake-stockfish", depth=3)
    assert agent.select_move(chess.Board(), _context(chess.BLACK)) == wanted


def test_reverse_stockfish_tie_breaks_by_uci_and_context_manager_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEngine(lambda move: 0)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)
    board = chess.Board()
    expected = min(board.legal_moves, key=chess.Move.uci)

    with ReverseStockfishAgent("fake-stockfish", nodes=10) as agent:
        assert agent.select_move(board, _context()) == expected

    assert fake.quit_called
    with pytest.raises(AgentError, match="closed"):
        agent.select_move(board, _context())


def test_reverse_stockfish_reports_missing_binary_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(executable: str) -> None:
        raise FileNotFoundError(executable)

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", missing)
    agent = ReverseStockfishAgent("does-not-exist", nodes=1)

    with pytest.raises(AgentError, match="could not start Stockfish binary"):
        agent.select_move(chess.Board(), _context())


def test_conventional_stockfish_agent_uses_engine_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEngine(lambda move: 0)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda executable: fake)
    board = chess.Board()
    agent = StockfishAgent("fake-stockfish", nodes=50)

    move = agent.select_move(board, _context())

    assert move in board.legal_moves
    assert fake.played_games == ["stockfish"]
    assert agent.name == "stockfish_nodes_50"
    agent.close()


@pytest.mark.parametrize(
    ("nodes", "depth"),
    [(0, None), (-1, None), (1, 1), (None, 0), (None, -1)],
)
def test_reverse_stockfish_rejects_invalid_limits(
    nodes: int | None, depth: int | None
) -> None:
    with pytest.raises(ValueError):
        ReverseStockfishAgent("stockfish", nodes=nodes, depth=depth)
