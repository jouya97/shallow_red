"""Shared interface implemented by every chess agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import chess


class AgentError(RuntimeError):
    """Raised when an agent cannot return a valid move."""


@dataclass(frozen=True)
class MoveContext:
    """Immutable metadata supplied to an agent for a single decision."""

    game_id: str
    ply: int
    seed: int
    target_color: chess.Color


@runtime_checkable
class Agent(Protocol):
    """A deterministic-under-context policy over legal chess moves."""

    @property
    def name(self) -> str:
        """Stable identifier stored in game artifacts."""

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        """Select a move without mutating ``board``."""

