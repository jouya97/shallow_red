"""Reverse-evaluation Stockfish baseline."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import chess
import chess.engine

from worst_chess.agents.base import AgentError, MoveContext


@dataclass(frozen=True, slots=True)
class ReverseMoveScore:
    """One legal root move scored from the designated loser's perspective.

    ``loser_value`` is a WDL-derived value in ``[-1, 1]`` where larger is
    better for losing: +1 means Stockfish expects the target to lose, 0 a
    draw, and -1 a target win. ``ranking_score`` preserves the existing
    lexicographic WDL/centipawn ordering in one sortable float; it is intended
    for ranks and distillation weights rather than calibrated value learning.
    """

    move: chess.Move
    loser_value: float
    ranking_score: float
    expectation_twice: int
    conventional_score: int


class ReverseStockfishAgent:
    """Choose the move Stockfish evaluates worst for the designated loser.

    Every legal root move is analysed independently with the same fixed node or
    depth limit.  The primary ordering is Stockfish's WDL expectation from the
    target side; centipawn/mate score resolves WDL ties, followed by ascending
    UCI notation.  The latter is obtained by iterating a sorted move list and
    retaining the first equal score.
    """

    def __init__(
        self,
        executable: str | Path,
        *,
        nodes: int | None = None,
        depth: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        wdl_model: chess.engine.WdlModel = "sf",
        uci_options: Mapping[str, str | int | None] | None = None,
    ) -> None:
        if nodes is not None and depth is not None:
            raise ValueError("configure exactly one of nodes or depth")
        if nodes is None and depth is None:
            nodes = 10_000
        if nodes is not None and nodes <= 0:
            raise ValueError("nodes must be positive")
        if depth is not None and depth <= 0:
            raise ValueError("depth must be positive")
        if threads <= 0 or hash_mb <= 0:
            raise ValueError("threads and hash_mb must be positive")

        self.executable = str(executable)
        self.nodes = nodes
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.wdl_model = wdl_model
        self.uci_options = dict(uci_options or {})
        self._engine: chess.engine.SimpleEngine | None = None
        self._closed = False

    @property
    def name(self) -> str:
        budget = (
            f"nodes_{self.nodes}" if self.nodes is not None else f"depth_{self.depth}"
        )
        return f"reverse_stockfish_{budget}"

    def __enter__(self) -> ReverseStockfishAgent:
        self._get_engine()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Quit the child engine process.  Calling this repeatedly is safe."""

        engine, self._engine = self._engine, None
        self._closed = True
        if engine is not None:
            # The process may already have died; there is nothing else to
            # release in the Python wrapper.
            with suppress(chess.engine.EngineError, OSError):
                engine.quit()

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        return self.evaluate_moves(board, context)[0].move

    def score_moves(
        self,
        board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        """Return every legal move's finite loser-oriented ranking score."""

        return {
            item.move: item.ranking_score
            for item in self.evaluate_moves(board, context)
        }

    def evaluate_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        root_moves: list[chess.Move] | tuple[chess.Move, ...] | None = None,
    ) -> tuple[ReverseMoveScore, ...]:
        """Return detailed legal-root evaluations, best losing move first.

        Supplying ``root_moves`` supports policy-guided search that spends the
        fixed per-root budget only on a neural policy's most promising moves.
        The result is deterministic: exact ties are resolved by ascending UCI.
        """

        available = tuple(board.legal_moves)
        if not available:
            raise AgentError(
                "ReverseStockfishAgent cannot score a terminal position"
            )
        if root_moves is None:
            candidates = sorted(available, key=chess.Move.uci)
        else:
            if not root_moves:
                raise ValueError("root_moves must not be empty")
            if len(set(root_moves)) != len(root_moves):
                raise ValueError("root_moves must not contain duplicates")
            if any(move not in available for move in root_moves):
                raise ValueError("root_moves must contain only legal moves")
            candidates = sorted(root_moves, key=chess.Move.uci)

        engine = self._get_engine()
        limit = chess.engine.Limit(nodes=self.nodes, depth=self.depth)
        scored: list[ReverseMoveScore] = []
        for move in candidates:
            try:
                info = engine.analyse(
                    board,
                    limit,
                    game=context.game_id,
                    root_moves=[move],
                    info=chess.engine.INFO_SCORE,
                )
            except (chess.engine.EngineError, OSError) as exc:
                raise AgentError(
                    f"Stockfish analysis failed for root move {move.uci()}: {exc}"
                ) from exc
            expectation_twice, conventional_score = self._evaluation_key(
                info, context.target_color, board.ply()
            )
            # expectation_twice is a conventional expected score on a 0..2000
            # scale. The fractional centipawn term is strictly below one, so
            # it breaks WDL ties without reversing the primary ordering.
            ranking_score = -float(expectation_twice) - (
                conventional_score / 200_001.0
            )
            scored.append(
                ReverseMoveScore(
                    move=move,
                    loser_value=1.0 - expectation_twice / 1000.0,
                    ranking_score=ranking_score,
                    expectation_twice=expectation_twice,
                    conventional_score=conventional_score,
                )
            )
        return tuple(
            sorted(scored, key=lambda item: (-item.ranking_score, item.move.uci()))
        )

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._closed:
            raise AgentError("ReverseStockfishAgent has been closed")
        if self._engine is not None:
            return self._engine
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.executable)
        except (chess.engine.EngineError, OSError) as exc:
            raise AgentError(
                "could not start Stockfish binary at "
                f"{self.executable!r}; install Stockfish and provide an executable path"
            ) from exc
        try:
            engine.configure(
                {
                    "Threads": self.threads,
                    "Hash": self.hash_mb,
                    **self.uci_options,
                }
            )
        except (chess.engine.EngineError, OSError) as exc:
            with suppress(chess.engine.EngineError, OSError):
                engine.quit()
            raise AgentError(f"could not configure Stockfish: {exc}") from exc
        self._engine = engine
        return engine

    def _evaluation_key(
        self,
        info: Mapping[str, Any],
        target_color: chess.Color,
        ply: int,
    ) -> tuple[int, int]:
        raw_score = info.get("score")
        if not isinstance(raw_score, chess.engine.PovScore):
            raise AgentError("Stockfish analysis returned no usable score")
        score = raw_score.pov(target_color)
        try:
            wdl = score.wdl(model=self.wdl_model, ply=ply)
        except (ValueError, TypeError) as exc:
            raise AgentError(
                f"Stockfish score could not be converted to WDL: {exc}"
            ) from exc
        # WDL totals use a fixed 1000 permille denominator.  Twice wins plus
        # draws is exact expected score without introducing floating rounding.
        expectation_twice = 2 * wdl.wins + wdl.draws
        conventional_score = score.score(mate_score=100_000)
        if conventional_score is None:
            raise AgentError("Stockfish returned an unrankable mate score")
        return expectation_twice, conventional_score


class StockfishAgent(ReverseStockfishAgent):
    """Conventional fixed-budget Stockfish opponent using the same lifecycle."""

    @property
    def name(self) -> str:
        budget = (
            f"nodes_{self.nodes}" if self.nodes is not None else f"depth_{self.depth}"
        )
        return f"stockfish_{budget}"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.is_game_over(claim_draw=False):
            raise AgentError("StockfishAgent cannot move from a terminal position")
        engine = self._get_engine()
        limit = chess.engine.Limit(nodes=self.nodes, depth=self.depth)
        try:
            result = engine.play(board, limit, game=context.game_id)
        except (chess.engine.EngineError, OSError) as exc:
            raise AgentError(f"Stockfish move selection failed: {exc}") from exc
        if result.move is None or result.move not in board.legal_moves:
            raise AgentError("Stockfish returned no legal move")
        return result.move


class LimitedStrengthStockfishAgent(StockfishAgent):
    """Stockfish configured to intentionally choose weaker ordinary moves."""

    def __init__(
        self,
        executable: str | Path,
        *,
        elo: int | None = None,
        skill_level: int | None = None,
        nodes: int | None = 1_000,
        depth: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
    ) -> None:
        if (elo is None) == (skill_level is None):
            raise ValueError("configure exactly one of elo or skill_level")
        if elo is not None:
            if not 1320 <= elo <= 3190:
                raise ValueError("elo must be in Stockfish's [1320, 3190] range")
            options: dict[str, str | int | None] = {
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
            }
            label = f"elo_{elo}"
        else:
            assert skill_level is not None
            if not 0 <= skill_level <= 20:
                raise ValueError("skill_level must be in [0, 20]")
            options = {"Skill Level": skill_level}
            label = f"skill_{skill_level}"
        self.strength_label = label
        super().__init__(
            executable,
            nodes=nodes,
            depth=depth,
            threads=threads,
            hash_mb=hash_mb,
            uci_options=options,
        )

    @property
    def name(self) -> str:
        budget = (
            f"nodes_{self.nodes}" if self.nodes is not None else f"depth_{self.depth}"
        )
        return f"stockfish_{self.strength_label}_{budget}"


__all__ = [
    "LimitedStrengthStockfishAgent",
    "ReverseMoveScore",
    "ReverseStockfishAgent",
    "StockfishAgent",
]
