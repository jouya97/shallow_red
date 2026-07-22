"""Deterministic legal opening-position suites."""

from __future__ import annotations

import hashlib

import chess


def _choice_index(payload: str, population: int) -> int:
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest, "big") % population


def generate_random_openings(
    count: int,
    plies: int,
    seed: int,
) -> tuple[str, ...]:
    """Generate unique, reproducible FENs from uniformly sampled legal moves.

    These positions are intended for smoke-test diversity, not as a substitute
    for the frozen, opening-family-stratified release suite.
    """

    if count < 1:
        raise ValueError("count must be positive")
    if plies < 0:
        raise ValueError("plies must not be negative")
    if plies == 0 and count > 1:
        raise ValueError("only one unique zero-ply opening exists")

    openings: list[str] = []
    seen: set[str] = set()
    attempt = 0
    max_attempts = max(100, count * 100)
    while len(openings) < count and attempt < max_attempts:
        board = chess.Board()
        for ply in range(plies):
            legal = sorted(board.legal_moves, key=chess.Move.uci)
            if not legal:
                break
            payload = f"opening-v1\0{seed}\0{attempt}\0{ply}\0{board.fen()}"
            board.push(legal[_choice_index(payload, len(legal))])
        attempt += 1
        if len(board.move_stack) != plies or board.is_game_over(claim_draw=False):
            continue
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            openings.append(fen)

    if len(openings) != count:
        raise RuntimeError(
            f"could generate only {len(openings)} unique openings after "
            f"{attempt} attempts"
        )
    return tuple(openings)


__all__ = ["generate_random_openings"]
