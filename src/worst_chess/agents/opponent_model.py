"""Search tailored to weak or random opponent reply distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.stockfish import ReverseStockfishAgent

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass(frozen=True)
class RandomReplyWeights:
    """Features approximating expected progress under uniform legal replies."""

    target_checkmate: float = -1_000_000_000_000.0
    immediate_mate_probability: float = 1_000_000_000.0
    immediate_draw_probability: float = -100_000_000.0
    checking_reply_probability: float = 1_000_000.0
    captured_target_value: float = 2_000.0
    target_ring_attack: float = 20_000.0
    target_king_escape: float = -10_000.0
    target_mobility: float = -100.0
    captured_opponent_value: float = -20_000.0
    policy_logit: float = 0.01
    low_material_threshold: int = 0
    low_material_captured_target_value: float = -4_000.0
    low_material_target_king_escape: float = 5_000.0
    low_material_target_mobility: float = 2_000.0
    pressure_scale: float = 1.0
    pressure_min_material: int = 0


@dataclass(frozen=True, slots=True)
class RandomReplyEvaluation:
    """Interpretable expected features for one target move."""

    score: float
    immediate_mate_probability: float
    immediate_draw_probability: float
    checking_reply_probability: float


class RandomReplySearchAgent:
    """Choose moves likely to be punished by a uniformly random opponent.

    For every retained target move, all legal opponent replies are enumerated.
    The primary feature is the exact fraction that immediately checkmates the
    target. Secondary features favor replies that check, capture target
    material, attack the target king ring, and restrict safe king movement.
    A neural policy can optionally limit the root to ``top_k`` candidates.
    """

    def __init__(
        self,
        policy: NeuralAgent | None = None,
        *,
        top_k: int = 12,
        weights: RandomReplyWeights | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.policy = policy
        self.top_k = top_k
        self.weights = weights or RandomReplyWeights()

    @property
    def name(self) -> str:
        prefix = "policy_" if self.policy is not None else "all_"
        return f"random_reply_{prefix}top_{self.top_k}"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("RandomReplySearchAgent must act for the target color")
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal:
            raise AgentError(
                "RandomReplySearchAgent cannot move from a terminal position"
            )
        policy_logits: dict[chess.Move, float] = {}
        if self.policy is not None:
            ranked = self.policy.rank_moves(board, context, top_k=self.top_k)
            legal = [item.move for item in ranked]
            policy_logits = {item.move: item.logit for item in ranked}

        best = legal[0]
        best_score = self.score_move(
            board,
            best,
            context.target_color,
            policy_logit=policy_logits.get(best, 0.0),
        )
        for move in legal[1:]:
            score = self.score_move(
                board,
                move,
                context.target_color,
                policy_logit=policy_logits.get(move, 0.0),
            )
            if score > best_score:
                best, best_score = move, score
        return best

    def score_moves(
        self,
        board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        """Score every legal target move for ranked-data generation."""

        if board.turn != context.target_color:
            raise AgentError("RandomReplySearchAgent must score the target color")
        if board.is_game_over(claim_draw=False):
            raise AgentError("RandomReplySearchAgent cannot score a terminal position")
        return {
            move: self.score_move(board, move, context.target_color)
            for move in board.legal_moves
        }

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
        *,
        policy_logit: float = 0.0,
    ) -> float:
        """Return expected losing progress under uniform legal replies."""

        return self.evaluate_move(
            board,
            move,
            target_color,
            policy_logit=policy_logit,
        ).score

    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
        *,
        policy_logit: float = 0.0,
    ) -> RandomReplyEvaluation:
        """Return the score and key random-reply probabilities."""

        if board.turn != target_color:
            raise AgentError("cannot score a move made by the non-target color")
        if move not in board.legal_moves:
            raise AgentError(f"cannot score illegal move {move.uci()}")
        if not math.isfinite(policy_logit):
            raise ValueError("policy_logit must be finite")
        captured_opponent = _captured_value(board, move, not target_color)
        position = board.copy(stack=False)
        position.push(move)
        weights = self.weights
        if position.is_checkmate():
            # The target just mated the opponent: the worst possible move.
            return RandomReplyEvaluation(
                weights.target_checkmate,
                0.0,
                0.0,
                0.0,
            )
        if position.is_game_over(claim_draw=False):
            return RandomReplyEvaluation(
                weights.immediate_draw_probability,
                0.0,
                1.0,
                0.0,
            )

        replies = tuple(position.legal_moves)
        mate_count = 0
        draw_count = 0
        check_count = 0
        captured_target = 0.0
        ring_attacks = 0.0
        king_escapes = 0.0
        target_mobility = 0.0
        opponent_color = not target_color
        for reply in replies:
            check_count += int(position.gives_check(reply))
            captured_target += _captured_value(position, reply, target_color)
            after = position.copy(stack=False)
            after.push(reply)
            if after.is_checkmate():
                mate_count += 1
                continue
            if after.is_game_over(claim_draw=False):
                draw_count += 1
                continue
            king = after.king(target_color)
            if king is not None:
                ring_attacks += sum(
                    len(after.attackers(opponent_color, square))
                    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king])
                )
                king_escapes += sum(
                    candidate.from_square == king for candidate in after.legal_moves
                )
            target_mobility += after.legal_moves.count()

        denominator = float(len(replies))
        mate_probability = mate_count / denominator
        draw_probability = draw_count / denominator
        check_probability = check_count / denominator
        target_material = _material(position, target_color)
        low_material = (
            weights.low_material_threshold > 0
            and target_material <= weights.low_material_threshold
        )
        pressure_scale = (
            weights.pressure_scale
            if target_material >= weights.pressure_min_material
            else 1.0
        )
        captured_target_weight = (
            weights.low_material_captured_target_value
            if low_material
            else weights.captured_target_value
        )
        king_escape_weight = (
            weights.low_material_target_king_escape
            if low_material
            else weights.target_king_escape * pressure_scale
        )
        mobility_weight = (
            weights.low_material_target_mobility
            if low_material
            else weights.target_mobility * pressure_scale
        )
        score = (
            weights.immediate_mate_probability * mate_probability
            + weights.immediate_draw_probability * draw_probability
            + weights.checking_reply_probability
            * pressure_scale
            * check_probability
            + captured_target_weight * captured_target / denominator
            + weights.target_ring_attack
            * pressure_scale
            * ring_attacks
            / denominator
            + king_escape_weight * king_escapes / denominator
            + mobility_weight * target_mobility / denominator
            + weights.captured_opponent_value * captured_opponent
            + weights.policy_logit * policy_logit
        )
        return RandomReplyEvaluation(
            score,
            mate_probability,
            draw_probability,
            check_probability,
        )


class StalemateAwareRandomReplySearchAgent(RandomReplySearchAgent):
    """Preserve the last target pieces and mobility to avoid late stalemate."""

    def __init__(
        self,
        policy: NeuralAgent | None = None,
        *,
        top_k: int = 12,
        low_material_threshold: int = 1_000,
        pressure_scale: float = 1.0,
        pressure_min_material: int = 0,
    ) -> None:
        if low_material_threshold <= 0:
            raise ValueError("low_material_threshold must be positive")
        if not math.isfinite(pressure_scale) or pressure_scale <= 0:
            raise ValueError("pressure_scale must be finite and positive")
        if pressure_min_material < 0:
            raise ValueError("pressure_min_material must be nonnegative")
        super().__init__(
            policy,
            top_k=top_k,
            weights=RandomReplyWeights(
                low_material_threshold=low_material_threshold,
                pressure_scale=pressure_scale,
                pressure_min_material=pressure_min_material,
            ),
        )
        self.pressure_scale = pressure_scale
        self.pressure_min_material = pressure_min_material

    @property
    def name(self) -> str:
        prefix = "policy_" if self.policy is not None else "all_"
        return (
            f"stalemate_aware_random_reply_{prefix}top_{self.top_k}_"
            f"pressure_{self.pressure_scale:g}_"
            f"above_{self.pressure_min_material}"
        )


class OpportunisticHybridAgent:
    """Take immediate random-mate opportunities, otherwise use reverse search."""

    def __init__(
        self,
        policy: NeuralAgent,
        evaluator: ReverseStockfishAgent,
        *,
        policy_top_k: int = 12,
        reverse_top_k: int = 8,
        reply_search: RandomReplySearchAgent | None = None,
    ) -> None:
        if policy_top_k < 1 or reverse_top_k < 1:
            raise ValueError("policy_top_k and reverse_top_k must be positive")
        if reverse_top_k > policy_top_k:
            raise ValueError("reverse_top_k must not exceed policy_top_k")
        self.policy = policy
        self.evaluator = evaluator
        self.policy_top_k = policy_top_k
        self.reverse_top_k = reverse_top_k
        self.reply_search = reply_search or RandomReplySearchAgent()

    @property
    def name(self) -> str:
        return (
            f"opportunistic_policy_{self.policy_top_k}_"
            f"reverse_{self.reverse_top_k}_{self.evaluator.name}"
        )

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("OpportunisticHybridAgent must act for the target color")
        ranked = self.policy.rank_moves(
            board, context, top_k=self.policy_top_k
        )
        reply_evaluations = [
            (
                candidate,
                self.reply_search.evaluate_move(
                    board,
                    candidate.move,
                    context.target_color,
                    policy_logit=candidate.logit,
                ),
            )
            for candidate in ranked
        ]
        mating = [
            item
            for item in reply_evaluations
            if item[1].immediate_mate_probability > 0.0
        ]
        if mating:
            return max(mating, key=lambda item: item[1].score)[0].move
        evaluations = self.evaluator.evaluate_moves(
            board,
            context,
            root_moves=[
                candidate.move
                for candidate, _ in reply_evaluations[: self.reverse_top_k]
            ],
        )
        return evaluations[0].move


def _captured_value(
    board: chess.Board,
    move: chess.Move,
    victim_color: chess.Color,
) -> int:
    if not board.is_capture(move):
        return 0
    victim: chess.Piece | None
    if board.is_en_passant(move):
        victim = chess.Piece(chess.PAWN, victim_color)
    else:
        victim = board.piece_at(move.to_square)
    if victim is None or victim.color != victim_color:
        return 0
    return _PIECE_VALUES[victim.piece_type]


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )


__all__ = [
    "OpportunisticHybridAgent",
    "RandomReplyEvaluation",
    "RandomReplySearchAgent",
    "RandomReplyWeights",
    "StalemateAwareRandomReplySearchAgent",
]
