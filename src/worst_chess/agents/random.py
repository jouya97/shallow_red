"""Deterministic uniform legal-move baseline."""

from __future__ import annotations

import hashlib

import chess

from worst_chess.agents.base import AgentError, MoveContext


class RandomAgent:
    """Choose a legal move uniformly from a context-derived pseudo-random index.

    The index is derived with a cryptographic hash instead of Python's ``hash``.
    It is therefore stable across interpreter processes and independent of
    ``PYTHONHASHSEED``.  Sorting by UCI also makes the population order stable.
    """

    @property
    def name(self) -> str:
        return "random"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        legal_moves = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal_moves:
            raise AgentError("RandomAgent cannot move from a terminal position")

        payload = (
            f"worst-chess-random-v1\0{context.seed}\0{context.game_id}\0{context.ply}"
        ).encode()
        digest = hashlib.sha256(payload).digest()
        index = int.from_bytes(digest, byteorder="big") % len(legal_moves)
        return legal_moves[index]
