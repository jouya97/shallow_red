"""Exact standard-chess Syzygy guidance for the designated losing side."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import chess
import chess.syzygy

from worst_chess.agents.base import Agent, AgentError, MoveContext


@dataclass(frozen=True, slots=True)
class SyzygyMoveScore:
    """Post-move standard-chess result from the opponent's perspective.

    ``opponent_wdl`` uses Syzygy's five-valued scale: ``2`` is an
    unconditional standard win for the opponent to move, ``1`` a cursed win,
    ``0`` a draw, ``-1`` a blessed loss, and ``-2`` an unconditional loss.
    A larger value is therefore better for the designated target's goal of
    eventually being checkmated.
    """

    move: chess.Move
    opponent_wdl: int
    opponent_dtz: int | None


class SyzygyLosingAgent:
    """Use local Syzygy results when complete, otherwise use ``fallback``.

    Syzygy solves ordinary adversarial chess, not the inverted game.  After a
    target candidate, the opponent is the side to move.  A positive WDL means
    that opponent can force a standard win even if the target resists, and is
    consequently strong guidance for losing.  It does *not* show that an
    opponent trying to avoid mate will cooperate.  DTZ is only a secondary
    zeroing-progress tie break; it is not distance to checkmate.
    """

    def __init__(
        self,
        directory: str | Path | None,
        fallback: Agent,
        *,
        max_fds: int = 128,
    ) -> None:
        if max_fds <= 0:
            raise ValueError("max_fds must be positive")
        self.directory = None if directory is None else Path(directory)
        self.fallback = fallback
        self.max_fds = max_fds
        self._tablebase: chess.syzygy.Tablebase | None = None
        self._open_attempted = False

    @property
    def name(self) -> str:
        return f"syzygy_losing_with_{self.fallback.name}"

    @property
    def tablebase_available(self) -> bool:
        """Whether a local tablebase has been opened successfully."""

        return self._get_tablebase() is not None

    def __enter__(self) -> SyzygyLosingAgent:
        self._get_tablebase()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close open table files; repeated calls are safe."""

        tablebase, self._tablebase = self._tablebase, None
        if tablebase is not None:
            with suppress(OSError):
                tablebase.close()

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("SyzygyLosingAgent must act for the target color")
        if board.is_game_over(claim_draw=False):
            raise AgentError("SyzygyLosingAgent cannot move from a terminal position")
        scores = self.evaluate_moves(board, context)
        if scores is None:
            return self._fallback_move(board, context)

        best_wdl = max(score.opponent_wdl for score in scores)
        best = [score for score in scores if score.opponent_wdl == best_wdl]
        if len(best) > 1 and all(score.opponent_dtz is not None for score in best):
            best_progress = max(
                _dtz_progress_key(best_wdl, score.opponent_dtz)
                for score in best
            )
            best = [
                score
                for score in best
                if _dtz_progress_key(best_wdl, score.opponent_dtz)
                == best_progress
            ]
        return min(best, key=lambda score: score.move.uci()).move

    def evaluate_moves(
        self, board: chess.Board, context: MoveContext
    ) -> tuple[SyzygyMoveScore, ...] | None:
        """Probe every legal candidate, or return ``None`` for full fallback.

        Partial local table coverage is never mixed with heuristic values.  If
        any candidate lacks WDL coverage, the entire position falls back.
        """

        if board.turn != context.target_color:
            raise AgentError("SyzygyLosingAgent must evaluate the target color")
        tablebase = self._get_tablebase()
        if tablebase is None:
            return None
        scores: list[SyzygyMoveScore] = []
        for move in sorted(board.legal_moves, key=chess.Move.uci):
            position = board.copy(stack=True)
            position.push(move)
            outcome = position.outcome(claim_draw=False)
            wdl: int | None
            dtz: int | None
            if outcome is not None:
                opponent = not context.target_color
                if outcome.winner is None:
                    wdl = 0
                elif outcome.winner == opponent:
                    wdl = 2
                else:
                    wdl = -2
                dtz = None
            else:
                try:
                    wdl = tablebase.get_wdl(position)
                    dtz = tablebase.get_dtz(position)
                except (KeyError, OSError):
                    return None
                if wdl is None:
                    return None
            scores.append(
                SyzygyMoveScore(
                    move=move,
                    opponent_wdl=wdl,
                    opponent_dtz=dtz,
                )
            )
        return tuple(scores) if scores else None

    def _fallback_move(
        self, board: chess.Board, context: MoveContext
    ) -> chess.Move:
        fallback_board = board.copy(stack=True)
        move = self.fallback.select_move(fallback_board, context)
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(f"tablebase fallback returned illegal move {detail}")
        return move

    def _get_tablebase(self) -> chess.syzygy.Tablebase | None:
        if self._tablebase is not None:
            return self._tablebase
        if self._open_attempted:
            return None
        self._open_attempted = True
        if self.directory is None or not self.directory.is_dir():
            return None
        try:
            self._tablebase = chess.syzygy.open_tablebase(
                str(self.directory),
                max_fds=self.max_fds,
            )
        except (OSError, ValueError):
            self._tablebase = None
        return self._tablebase


def _dtz_progress_key(wdl: int, dtz: int | None) -> int:
    if dtz is None or wdl == 0:
        return 0
    # When the opponent is winning, prefer quicker zeroing progress.  When the
    # opponent is losing, prefer delaying its forced standard-chess loss.
    return -abs(dtz) if wdl > 0 else abs(dtz)


__all__ = ["SyzygyLosingAgent", "SyzygyMoveScore"]
