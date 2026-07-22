"""Deterministic opponent that resists checkmating the designated loser."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from worst_chess.agents.base import AgentError, MoveContext

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass(frozen=True)
class ResistantWeights:
    """Integer weights for the resistant opponent's transparent evaluation."""

    material_balance: int = 10
    target_in_check: int = -50_000
    target_king_escape: int = 1_000
    target_legal_move: int = 20
    target_ring_attack: int = -300
    own_ring_attack: int = -100
    center_control: int = 4
    immediate_draw: int = 100_000


class ResistantOpponentAgent:
    """Play ordinary chess while making the target difficult to checkmate.

    ``MoveContext.target_color`` identifies the losing agent, not this agent.
    Consequently, this policy may only be called when ``board.turn`` is the
    opposite color.  It searches every legal move at one ply.  Immediate mates
    of the target are removed from consideration whenever any alternative is
    legal; remaining moves are scored for ordinary material and king safety,
    while checks, attacks around the target king, and restricted target
    mobility are discouraged.

    This is deliberately a cheap benchmark opponent rather than a strong chess
    engine.  It never mutates the supplied board and breaks ties by UCI order.
    """

    def __init__(self, weights: ResistantWeights | None = None) -> None:
        self.weights = weights or ResistantWeights()

    @property
    def name(self) -> str:
        return "resistant_opponent"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn == context.target_color:
            raise AgentError(
                "ResistantOpponentAgent must move for the non-target color; "
                "MoveContext.target_color identifies the designated loser"
            )

        legal_moves = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal_moves:
            raise AgentError(
                "ResistantOpponentAgent cannot move from a terminal position"
            )

        non_mating = [
            move
            for move in legal_moves
            if not _checkmates_target(board, move, context.target_color)
        ]
        candidates = non_mating or legal_moves

        best_move = candidates[0]
        best_score = self.score_move(board, best_move, context.target_color)
        for move in candidates[1:]:
            score = self.score_move(board, move, context.target_color)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
    ) -> int:
        """Score a legal opponent move; larger values are more resistant."""

        if board.turn == target_color:
            raise AgentError("cannot score a resistant move made by the target color")
        if move not in board.legal_moves:
            raise AgentError(f"cannot score illegal move {move.uci()}")

        opponent_color = not target_color
        position = board.copy(stack=False)
        position.push(move)
        weights = self.weights

        # select_move handles mate lexicographically, rather than relying on a
        # finite score that other features could theoretically overwhelm.
        if position.is_checkmate():
            return -1_000_000_000
        if position.is_stalemate() or position.is_insufficient_material():
            return weights.immediate_draw

        target_king = position.king(target_color)
        opponent_king = position.king(opponent_color)
        target_ring_attacks = _ring_attacks(
            position, target_king, attacker=opponent_color
        )
        own_ring_attacks = _ring_attacks(
            position, opponent_king, attacker=target_color
        )
        target_escapes = _legal_king_moves(position, target_king)
        target_mobility = position.legal_moves.count()

        center = (chess.D4, chess.E4, chess.D5, chess.E5)
        center_control = sum(
            len(position.attackers(opponent_color, square)) for square in center
        )

        return (
            weights.material_balance
            * (_material(position, opponent_color) - _material(position, target_color))
            + weights.target_in_check * int(position.is_check())
            + weights.target_king_escape * target_escapes
            + weights.target_legal_move * target_mobility
            + weights.target_ring_attack * target_ring_attacks
            + weights.own_ring_attack * own_ring_attacks
            + weights.center_control * center_control
        )


def _checkmates_target(
    board: chess.Board,
    move: chess.Move,
    target_color: chess.Color,
) -> bool:
    position = board.copy(stack=False)
    position.push(move)
    return position.turn == target_color and position.is_checkmate()


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )


def _ring_attacks(
    board: chess.Board,
    king_square: chess.Square | None,
    *,
    attacker: chess.Color,
) -> int:
    if king_square is None:
        return 0
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
    return sum(len(board.attackers(attacker, square)) for square in ring)


def _legal_king_moves(
    board: chess.Board,
    king_square: chess.Square | None,
) -> int:
    if king_square is None:
        return 0
    return sum(move.from_square == king_square for move in board.legal_moves)


__all__ = ["ResistantOpponentAgent", "ResistantWeights"]
