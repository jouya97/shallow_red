"""A cheap, deterministic material-shedding losing-chess baseline."""

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
    # The king is never captured.  This value is only defensive completeness.
    chess.KING: 0,
}


@dataclass(frozen=True)
class HeuristicWeights:
    """Integer weights for the interpretable move features."""

    material_shed: int = 20
    capturable_material: int = 6
    moved_piece_sacrifice: int = 8
    captured_enemy_material: int = -10
    king_ring_attack: int = 35
    available_check: int = 45
    missing_pawn_shield: int = 20
    accidental_draw: int = -100_000
    accidental_mate: int = -1_000_000


class HeuristicAgent:
    """Prefer moves that make the designated loser's position easier to ruin.

    This is intentionally a one-ply heuristic, not a claim of optimal play.  It
    rewards shedding material, offering valuable pieces to legal captures, and
    exposing the target king.  Capturing the opponent's material is penalized.
    Every candidate is examined on a copy, so the caller's board is untouched.
    """

    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        self.weights = weights or HeuristicWeights()

    @property
    def name(self) -> str:
        return "heuristic_loser"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        legal_moves = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal_moves:
            raise AgentError("HeuristicAgent cannot move from a terminal position")

        # max() deliberately retains the first item on a tie; legal_moves is in
        # UCI order, giving a deterministic and documented tie break.
        best_move = legal_moves[0]
        best_score = self.score_move(board, best_move, context.target_color)
        for move in legal_moves[1:]:
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
        """Return the losing desirability of a legal move for ``target_color``."""

        if move not in board.legal_moves:
            raise AgentError(f"cannot score illegal move {move.uci()}")

        before_material = _material(board, target_color)
        captured_enemy = _captured_piece_value(board, move, target_color)
        moved_piece = board.piece_at(move.from_square)

        position = board.copy(stack=False)
        position.push(move)

        weights = self.weights
        if position.is_checkmate():
            # A legal move can only leave the other side checkmated, which is
            # the exact opposite of the designated loser's objective.
            return weights.accidental_mate
        if position.is_stalemate() or position.is_insufficient_material():
            return weights.accidental_draw

        material_shed = before_material - _material(position, target_color)
        capturable, moved_is_capturable = _capture_opportunities(
            position, target_color, move.to_square
        )
        king_ring_attack, available_checks, missing_shield = _king_exposure(
            position, target_color
        )

        moved_value = (
            0 if moved_piece is None else _PIECE_VALUES[moved_piece.piece_type]
        )
        sacrifice_value = moved_value if moved_is_capturable else 0

        return (
            weights.material_shed * material_shed
            + weights.capturable_material * capturable
            + weights.moved_piece_sacrifice * sacrifice_value
            + weights.captured_enemy_material * captured_enemy
            + weights.king_ring_attack * king_ring_attack
            + weights.available_check * available_checks
            + weights.missing_pawn_shield * missing_shield
        )


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )


def _captured_piece_value(
    board: chess.Board, move: chess.Move, target_color: chess.Color
) -> int:
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        return _PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    if captured is None or captured.color == target_color:
        return 0
    return _PIECE_VALUES[captured.piece_type]


def _capture_opportunities(
    board: chess.Board, target_color: chess.Color, moved_to: chess.Square
) -> tuple[int, bool]:
    """Value target pieces the opponent can legally capture on its next move."""

    if board.turn == target_color or board.is_game_over(claim_draw=False):
        return 0, False

    # Count each vulnerable piece once rather than once per possible attacker.
    vulnerable: dict[chess.Square, int] = {}
    moved_is_capturable = False
    for reply in board.legal_moves:
        if not board.is_capture(reply):
            continue
        victim_square = reply.to_square
        if board.is_en_passant(reply):
            victim_square = chess.square(
                chess.square_file(reply.to_square), chess.square_rank(reply.from_square)
            )
        victim = board.piece_at(victim_square)
        if victim is None or victim.color != target_color:
            continue
        vulnerable[victim_square] = _PIECE_VALUES[victim.piece_type]
        moved_is_capturable |= victim_square == moved_to
    return sum(vulnerable.values()), moved_is_capturable


def _king_exposure(
    board: chess.Board, target_color: chess.Color
) -> tuple[int, int, int]:
    king_square = board.king(target_color)
    if king_square is None:
        return 0, 0, 0

    enemy = not target_color
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
    ring_attacks = sum(len(board.attackers(enemy, square)) for square in ring)

    available_checks = 0
    if board.turn == enemy and not board.is_game_over(claim_draw=False):
        available_checks = sum(board.gives_check(move) for move in board.legal_moves)

    direction = 1 if target_color == chess.WHITE else -1
    king_rank = chess.square_rank(king_square)
    king_file = chess.square_file(king_square)
    missing_shield = 0
    shield_rank = king_rank + direction
    if 0 <= shield_rank < 8:
        for shield_file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            shield = board.piece_at(chess.square(shield_file, shield_rank))
            if shield != chess.Piece(chess.PAWN, target_color):
                missing_shield += 1
    return ring_attacks, available_checks, missing_shield


# A descriptive alias retained for experiment configuration readability.
GreedySacrificeAgent = HeuristicAgent


__all__ = ["GreedySacrificeAgent", "HeuristicAgent", "HeuristicWeights"]
