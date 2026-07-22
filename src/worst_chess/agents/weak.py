"""Cheap, purposeful, and noisy ordinary-chess opponent policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chess

from worst_chess.agents.base import Agent, AgentError, MoveContext

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass(frozen=True)
class MaterialOpponentWeights:
    """Transparent weights for a weak one-ply ordinary-chess player."""

    material_balance: int = 10
    gives_check: int = 1_500
    target_ring_attack: int = 200
    own_ring_attack: int = -120
    center_control: int = 8
    target_mobility: int = -10
    checkmate: int = 1_000_000_000
    immediate_draw: int = -100_000_000


class MaterialOpponentAgent:
    """Maximize a shallow ordinary-chess evaluation for the non-target side."""

    def __init__(self, weights: MaterialOpponentWeights | None = None) -> None:
        self.weights = weights or MaterialOpponentWeights()

    @property
    def name(self) -> str:
        return "material_opponent"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        _require_opponent_turn(board, context, self.name)
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        best = legal[0]
        best_score = self.score_move(board, best, context.target_color)
        for move in legal[1:]:
            score = self.score_move(board, move, context.target_color)
            if score > best_score:
                best, best_score = move, score
        return best

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
    ) -> int:
        """Return ordinary-chess desirability for one legal opponent move."""

        if board.turn == target_color:
            raise AgentError("material opponent cannot move for the target color")
        if move not in board.legal_moves:
            raise AgentError(f"cannot score illegal move {move.uci()}")
        opponent = not target_color
        position = board.copy(stack=False)
        position.push(move)
        weights = self.weights
        if position.is_checkmate():
            return weights.checkmate
        if position.is_game_over(claim_draw=False):
            return weights.immediate_draw
        target_king = position.king(target_color)
        own_king = position.king(opponent)
        centers = (chess.D4, chess.E4, chess.D5, chess.E5)
        return (
            weights.material_balance
            * (_material(position, opponent) - _material(position, target_color))
            + weights.gives_check * int(position.is_check())
            + weights.target_ring_attack
            * _ring_attacks(position, target_king, opponent)
            + weights.own_ring_attack
            * _ring_attacks(position, own_king, target_color)
            + weights.center_control
            * sum(len(position.attackers(opponent, square)) for square in centers)
            + weights.target_mobility * position.legal_moves.count()
        )


class CaptureFirstOpponentAgent:
    """A very weak player that greedily captures, checks, and develops."""

    @property
    def name(self) -> str:
        return "capture_first_opponent"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        _require_opponent_turn(board, context, self.name)
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        return max(
            legal,
            key=lambda move: self._key(board, move, context.target_color),
        )

    def _key(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
    ) -> tuple[int, int, int, int, int]:
        position = board.copy(stack=False)
        captured = _captured_value(board, move, target_color)
        moving_piece = board.piece_at(move.from_square)
        position.push(move)
        central = int(move.to_square in (chess.D4, chess.E4, chess.D5, chess.E5))
        development = int(
            moving_piece is not None
            and moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and chess.square_rank(move.from_square) in (0, 7)
        )
        return (
            int(position.is_checkmate()),
            captured,
            int(position.is_check()),
            development,
            central,
        )


class NoisyOpponentAgent:
    """Mix a purposeful opponent with deterministic uniform legal mistakes."""

    def __init__(
        self,
        base: Agent,
        *,
        random_move_probability: float,
        salt: str = "default",
    ) -> None:
        if not 0.0 <= random_move_probability <= 1.0:
            raise ValueError("random_move_probability must be in [0, 1]")
        if not salt:
            raise ValueError("salt must not be empty")
        self.base = base
        self.random_move_probability = random_move_probability
        self.salt = salt

    @property
    def name(self) -> str:
        percentage = round(self.random_move_probability * 100)
        return f"noisy_{self.base.name}_{percentage}pct"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        _require_opponent_turn(board, context, self.name)
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        digest = hashlib.sha256(
            (
                f"shallow-red-noisy-v1\0{self.salt}\0{context.seed}\0"
                f"{context.game_id}\0{context.ply}\0{board.fen()}"
            ).encode()
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / 2**64
        if sample < self.random_move_probability:
            return legal[int.from_bytes(digest[8:16], "big") % len(legal)]
        move = self.base.select_move(board.copy(stack=True), context)
        if not isinstance(move, chess.Move) or move not in legal:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(f"base opponent returned illegal move {detail}")
        return move


def _require_opponent_turn(
    board: chess.Board,
    context: MoveContext,
    name: str,
) -> None:
    if board.turn == context.target_color:
        raise AgentError(f"{name} must act for the non-target color")
    if board.is_game_over(claim_draw=False):
        raise AgentError(f"{name} cannot move from a terminal position")


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )


def _ring_attacks(
    board: chess.Board,
    king_square: chess.Square | None,
    attacker: chess.Color,
) -> int:
    if king_square is None:
        return 0
    return sum(
        len(board.attackers(attacker, square))
        for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
    )


def _captured_value(
    board: chess.Board,
    move: chess.Move,
    target_color: chess.Color,
) -> int:
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        return _PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    if captured is None or captured.color != target_color:
        return 0
    return _PIECE_VALUES[captured.piece_type]


__all__ = [
    "CaptureFirstOpponentAgent",
    "MaterialOpponentAgent",
    "MaterialOpponentWeights",
    "NoisyOpponentAgent",
]
