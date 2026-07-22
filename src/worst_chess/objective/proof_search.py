"""Budgeted AND/OR proof search for orthodox forced selfmate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess


class ProofStatus(str, Enum):
    """Whether bounded search proved, refuted, or could not finish a claim."""

    PROVEN = "proven"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProofSearchConfig:
    """Limits for one history-preserving forced-selfmate search."""

    max_plies: int = 4
    node_budget: int = 100_000

    def __post_init__(self) -> None:
        if self.max_plies < 0:
            raise ValueError("max_plies must not be negative")
        if self.node_budget < 1:
            raise ValueError("node_budget must be positive")


@dataclass(frozen=True, slots=True)
class ProofResult:
    """Auditable bounded-search result and one representative principal line."""

    status: ProofStatus
    plies: int | None
    principal_variation: tuple[chess.Move, ...]
    nodes: int
    cache_hits: int
    max_plies: int
    node_budget: int

    @property
    def first_move(self) -> chess.Move | None:
        return self.principal_variation[0] if self.principal_variation else None


@dataclass(frozen=True, slots=True)
class _NodeResult:
    status: ProofStatus
    plies: int | None = None
    principal_variation: tuple[chess.Move, ...] = ()


class _Search:
    def __init__(self, target_color: chess.Color, config: ProofSearchConfig) -> None:
        self.target_color = target_color
        self.config = config
        self.nodes = 0
        self.cache_hits = 0
        self.cache: dict[tuple[str, int], _NodeResult] = {}

    def solve(self, board: chess.Board, remaining: int) -> _NodeResult:
        if self.nodes >= self.config.node_budget:
            return _NodeResult(ProofStatus.UNKNOWN)
        self.nodes += 1

        terminal = self._terminal(board)
        if terminal is not None:
            return terminal
        if remaining == 0:
            return _NodeResult(ProofStatus.REFUTED)

        key = (_position_key(board), remaining)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        result = (
            self._target_node(board, remaining)
            if board.turn == self.target_color
            else self._opponent_node(board, remaining)
        )
        if result.status is not ProofStatus.UNKNOWN:
            self.cache[key] = result
        return result

    def _terminal(self, board: chess.Board) -> _NodeResult | None:
        if board.is_checkmate():
            if board.turn == self.target_color:
                return _NodeResult(ProofStatus.PROVEN, plies=0)
            return _NodeResult(ProofStatus.REFUTED)
        if board.is_stalemate() or board.is_insufficient_material():
            return _NodeResult(ProofStatus.REFUTED)
        if board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return _NodeResult(ProofStatus.REFUTED)
        return None

    def _target_node(self, board: chess.Board, remaining: int) -> _NodeResult:
        best: _NodeResult | None = None
        unknown = False
        for move in _ordered_target_moves(board, self.target_color):
            board.push(move)
            try:
                child = self.solve(board, remaining - 1)
            finally:
                board.pop()
            if child.status is ProofStatus.PROVEN:
                candidate = _prepend(move, child)
                if best is None or _required_plies(candidate) < _required_plies(best):
                    best = candidate
            elif child.status is ProofStatus.UNKNOWN:
                unknown = True
        if best is not None:
            return best
        return _NodeResult(
            ProofStatus.UNKNOWN if unknown else ProofStatus.REFUTED
        )

    def _opponent_node(self, board: chess.Board, remaining: int) -> _NodeResult:
        worst: _NodeResult | None = None
        unknown = False
        for move in _ordered_opponent_moves(board, self.target_color):
            board.push(move)
            try:
                child = self.solve(board, remaining - 1)
            finally:
                board.pop()
            if child.status is ProofStatus.REFUTED:
                return _prepend(move, child)
            if child.status is ProofStatus.UNKNOWN:
                unknown = True
                continue
            candidate = _prepend(move, child)
            if worst is None or _required_plies(candidate) > _required_plies(worst):
                worst = candidate
        if unknown:
            return _NodeResult(ProofStatus.UNKNOWN)
        if worst is None:
            return _NodeResult(ProofStatus.REFUTED)
        return worst


def prove_forced_selfmate(
    board: chess.Board,
    target_color: chess.Color,
    config: ProofSearchConfig | None = None,
) -> ProofResult:
    """Prove forced target checkmate within a bounded number of plies.

    Target nodes are existential: one legal continuation is sufficient.
    Opponent nodes are universal: every legal reply must remain proven. Draws
    and checkmating the opponent refute the line. ``UNKNOWN`` means the node
    budget expired; ``REFUTED`` means complete search found no proof inside the
    requested horizon.
    """

    if type(target_color) is not bool:
        raise TypeError("target_color must be chess.WHITE or chess.BLACK")
    if not board.is_valid():
        raise ValueError("board must be a valid orthodox chess position")
    resolved = config or ProofSearchConfig()
    search = _Search(target_color, resolved)
    node = search.solve(board.copy(stack=True), resolved.max_plies)
    return ProofResult(
        status=node.status,
        plies=node.plies,
        principal_variation=node.principal_variation,
        nodes=search.nodes,
        cache_hits=search.cache_hits,
        max_plies=resolved.max_plies,
        node_budget=resolved.node_budget,
    )


def prove_forced_selfmate_after_move(
    board: chess.Board,
    target_color: chess.Color,
    first_move: chess.Move,
    config: ProofSearchConfig | None = None,
) -> ProofResult:
    """Prove that one designated target move starts a forced selfmate.

    This is useful for validating generated policy labels: the root remains
    existential, but only the proposed action is tested. Every legal opponent
    reply after it is still searched universally.
    """

    if type(target_color) is not bool:
        raise TypeError("target_color must be chess.WHITE or chess.BLACK")
    if not board.is_valid():
        raise ValueError("board must be a valid orthodox chess position")
    if board.turn != target_color:
        raise ValueError("target must be the side to move")
    if first_move not in board.legal_moves:
        raise ValueError("first_move must be legal")
    resolved = config or ProofSearchConfig()
    if resolved.max_plies < 1:
        raise ValueError("max_plies must include the designated first move")

    search = _Search(target_color, resolved)
    position = board.copy(stack=True)
    position.push(first_move)
    child = search.solve(position, resolved.max_plies - 1)
    node = _prepend(first_move, child)
    return ProofResult(
        status=node.status,
        plies=node.plies,
        principal_variation=node.principal_variation,
        nodes=search.nodes,
        cache_hits=search.cache_hits,
        max_plies=resolved.max_plies,
        node_budget=resolved.node_budget,
    )


def _prepend(move: chess.Move, child: _NodeResult) -> _NodeResult:
    plies = None if child.plies is None else child.plies + 1
    return _NodeResult(
        child.status,
        plies=plies,
        principal_variation=(move, *child.principal_variation),
    )


def _required_plies(result: _NodeResult) -> int:
    if result.plies is None:
        raise ValueError("proven comparison requires a finite distance")
    return result.plies


def _position_key(board: chess.Board) -> str:
    # Fullmove number is irrelevant; halfmove clock matters for the automatic
    # 75-move rule. Legal en-passant serialization avoids phantom distinctions.
    return " ".join(board.fen(en_passant="legal").split()[:5])


def _ordered_target_moves(
    board: chess.Board,
    target_color: chess.Color,
) -> tuple[chess.Move, ...]:
    def key(move: chess.Move) -> tuple[int, int, int, str]:
        position = board.copy(stack=False)
        position.push(move)
        if position.is_checkmate():
            return (1, 0, 0, move.uci())
        replies = tuple(position.legal_moves)
        mating_replies = 0
        for reply in replies:
            child = position.copy(stack=False)
            child.push(reply)
            mating_replies += int(child.is_checkmate() and child.turn == target_color)
        return (
            0,
            -mating_replies,
            int(position.is_check()),
            move.uci(),
        )

    return tuple(sorted(board.legal_moves, key=key))


def _ordered_opponent_moves(
    board: chess.Board,
    target_color: chess.Color,
) -> tuple[chess.Move, ...]:
    def key(move: chess.Move) -> tuple[int, int, str]:
        position = board.copy(stack=False)
        position.push(move)
        target_mated = position.is_checkmate() and position.turn == target_color
        terminal_failure = position.is_game_over(claim_draw=False) and not target_mated
        return (int(target_mated), -int(terminal_failure), move.uci())

    return tuple(sorted(board.legal_moves, key=key))


__all__ = [
    "ProofResult",
    "ProofSearchConfig",
    "ProofStatus",
    "prove_forced_selfmate",
    "prove_forced_selfmate_after_move",
]
