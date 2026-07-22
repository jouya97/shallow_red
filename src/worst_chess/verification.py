"""High-volume deterministic verification outside the regular unit-test budget."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import chess
import numpy as np

from worst_chess.chess.actions import decode_action, encode_move, legal_action_mask


@dataclass(frozen=True)
class ActionVerificationResult:
    requested_transitions: int
    verified_transitions: int
    positions: int
    completed_games: int
    elapsed_seconds: float

    @property
    def transitions_per_second(self) -> float:
        return self.verified_transitions / self.elapsed_seconds


def _select_move(
    moves: tuple[chess.Move, ...], seed: int, position_index: int, fen: str
) -> chess.Move:
    payload = f"action-verifier-v1\0{seed}\0{position_index}\0{fen}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest(), "big") % len(moves)
    return moves[index]


def verify_action_roundtrips(
    requested_transitions: int,
    *,
    seed: int = 20260721,
) -> ActionVerificationResult:
    """Verify at least the requested number of legal move/action round trips."""

    if requested_transitions < 1:
        raise ValueError("requested_transitions must be positive")

    started = time.perf_counter()
    board = chess.Board()
    transitions = 0
    positions = 0
    games = 0
    while transitions < requested_transitions:
        if board.is_game_over(claim_draw=False):
            games += 1
            board.reset()
        legal_moves = tuple(sorted(board.legal_moves, key=chess.Move.uci))
        if not legal_moves:
            raise AssertionError("nonterminal board unexpectedly has no legal moves")

        actions: set[int] = set()
        for move in legal_moves:
            action = encode_move(board, move)
            if action in actions:
                raise AssertionError(f"action collision at {board.fen()}: {action}")
            actions.add(action)
            if decode_action(board, action) != move:
                raise AssertionError(
                    f"round-trip mismatch at {board.fen()}: {move.uci()}"
                )
        mask = legal_action_mask(board)
        if int(np.count_nonzero(mask)) != len(legal_moves):
            raise AssertionError(f"legal mask mismatch at {board.fen()}")

        transitions += len(legal_moves)
        positions += 1
        board.push(_select_move(legal_moves, seed, positions, board.fen()))

    elapsed = time.perf_counter() - started
    return ActionVerificationResult(
        requested_transitions=requested_transitions,
        verified_transitions=transitions,
        positions=positions,
        completed_games=games,
        elapsed_seconds=elapsed,
    )


__all__ = ["ActionVerificationResult", "verify_action_roundtrips"]

