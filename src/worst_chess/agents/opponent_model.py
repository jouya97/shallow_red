"""Search tailored to weak or random opponent reply distributions."""

from __future__ import annotations

import hashlib
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
    """Preserve late mobility and optionally override for immediate mate."""

    def __init__(
        self,
        policy: NeuralAgent | None = None,
        *,
        top_k: int = 12,
        low_material_threshold: int = 1_000,
        pressure_scale: float = 1.0,
        pressure_min_material: int = 0,
        cycle_penalty: float = 0.0,
        tactical_mate_override: bool = False,
        forced_mate_override: bool = False,
    ) -> None:
        if low_material_threshold <= 0:
            raise ValueError("low_material_threshold must be positive")
        if not math.isfinite(pressure_scale) or pressure_scale <= 0:
            raise ValueError("pressure_scale must be finite and positive")
        if pressure_min_material < 0:
            raise ValueError("pressure_min_material must be nonnegative")
        if not math.isfinite(cycle_penalty) or cycle_penalty < 0:
            raise ValueError("cycle_penalty must be finite and nonnegative")
        if tactical_mate_override and forced_mate_override:
            raise ValueError(
                "tactical_mate_override and forced_mate_override are mutually "
                "exclusive"
            )
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
        self.cycle_penalty = cycle_penalty
        self.tactical_mate_override = tactical_mate_override
        self.forced_mate_override = forced_mate_override

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        """Take the best all-legal immediate mate chance when opted in."""

        if not (self.tactical_mate_override or self.forced_mate_override):
            return super().select_move(board, context)
        if board.turn != context.target_color:
            raise AgentError(
                "StalemateAwareRandomReplySearchAgent must act for the target color"
            )
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal:
            raise AgentError(
                "StalemateAwareRandomReplySearchAgent cannot move from a terminal "
                "position"
            )

        best_move: chess.Move | None = None
        best_probability = 0.0
        for move in legal:
            probability = self._immediate_mate_probability(
                board,
                move,
                context.target_color,
            )
            eligible = (
                probability == 1.0
                if self.forced_mate_override
                else probability > 0.0
            )
            if eligible and (
                best_move is None or probability > best_probability
            ):
                best_move = move
                best_probability = probability
        if best_move is not None:
            return best_move
        return super().select_move(board, context)

    @staticmethod
    def _immediate_mate_probability(
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
    ) -> float:
        """Return the exact uniform-reply chance of the target being mated."""

        position = board.copy(stack=False)
        position.push(move)
        if position.is_game_over(claim_draw=False):
            return 0.0
        replies = tuple(position.legal_moves)
        mate_count = 0
        for reply in replies:
            after = position.copy(stack=False)
            after.push(reply)
            mate_count += int(
                after.is_checkmate() and after.turn == target_color
            )
        return mate_count / len(replies)

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        target_color: chess.Color,
        *,
        policy_logit: float = 0.0,
    ) -> float:
        """Apply an optional penalty when ``move`` repeats a prior position."""

        score = super().score_move(
            board,
            move,
            target_color,
            policy_logit=policy_logit,
        )
        if self.cycle_penalty == 0.0:
            return score
        position = board.copy(stack=True)
        position.push(move)
        if position.is_repetition(2):
            score -= self.cycle_penalty
        return score

    @property
    def name(self) -> str:
        prefix = "policy_" if self.policy is not None else "all_"
        name = (
            f"stalemate_aware_random_reply_{prefix}top_{self.top_k}_"
            f"pressure_{self.pressure_scale:g}_"
            f"above_{self.pressure_min_material}"
        )
        if self.cycle_penalty > 0.0:
            name += f"_cycle_penalty_{self.cycle_penalty:g}"
        if self.tactical_mate_override:
            name += "_tactical_mate_override"
        if self.forced_mate_override:
            name += "_forced_mate_override"
        return name


@dataclass(frozen=True, slots=True)
class SampledExpectimaxConfig:
    """Controls deterministic common-random opponent-reply sampling."""

    reply_samples: int = 4
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.reply_samples < 1:
            raise ValueError("reply_samples must be positive")
        if self.reply_samples > 1_024:
            raise ValueError("reply_samples must not exceed 1024")


class TwoTurnRandomReplyAgent:
    """Look through one random opponent reply and one further target turn.

    Root moves come from the neural policy shortlist.  Each root is evaluated
    with the same deterministic random fractions, mapped onto that root's
    UCI-sorted legal replies.  The continuation is the existing one-ply
    stalemate-aware random-reply search using the same neural policy.
    """

    _SELF_MATE_VALUE = 1.0e30
    # One sampled selfmate must dominate every lower-priority sampled result.
    # Draws remain preferable to certain target wins, while nonterminal leaf
    # scores (normally around 1e12 or less) can still outrank a certain draw.
    _DRAW_VALUE = -1.0e15
    _TARGET_WIN_VALUE = -1.0e20

    def __init__(
        self,
        policy: NeuralAgent,
        *,
        top_k: int = 8,
        config: SampledExpectimaxConfig | None = None,
        continuation: StalemateAwareRandomReplySearchAgent | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.policy = policy
        self.top_k = top_k
        self.config = config or SampledExpectimaxConfig()
        self.continuation = continuation or StalemateAwareRandomReplySearchAgent(
            policy,
            top_k=top_k,
        )

    @property
    def name(self) -> str:
        return (
            f"two_turn_random_reply_top_{self.top_k}_"
            f"samples_{self.config.reply_samples}_seed_{self.config.seed}"
        )

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        scores = self.score_candidates(board, context)
        return max(sorted(scores, key=chess.Move.uci), key=scores.__getitem__)

    def score_candidates(
        self,
        board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        """Return sampled two-target-turn values without mutating ``board``."""

        if board.turn != context.target_color:
            raise AgentError("TwoTurnRandomReplyAgent must act for the target color")
        if board.is_game_over(claim_draw=False):
            raise AgentError(
                "TwoTurnRandomReplyAgent cannot move from a terminal position"
            )
        ranked = self.policy.rank_moves(board, context, top_k=self.top_k)
        candidates = sorted((item.move for item in ranked), key=chess.Move.uci)
        if not candidates:
            raise AgentError("neural policy returned no root candidates")
        for move in candidates:
            if move not in board.legal_moves:
                raise AgentError(f"neural policy returned illegal move {move.uci()}")

        sample_words = tuple(
            self._sample_word(context, sample_index)
            for sample_index in range(self.config.reply_samples)
        )
        return {
            move: self._score_root(board, move, context, sample_words)
            for move in candidates
        }

    def _score_root(
        self,
        board: chess.Board,
        move: chess.Move,
        context: MoveContext,
        sample_words: tuple[int, ...],
    ) -> float:
        position = board.copy(stack=True)
        position.push(move)
        terminal = self._terminal_value(position, context.target_color)
        if terminal is not None:
            return terminal

        replies = sorted(position.legal_moves, key=chess.Move.uci)
        leaf_values: list[float] = []
        for word in sample_words:
            reply = replies[(word * len(replies)) >> 64]
            successor = position.copy(stack=True)
            successor.push(reply)
            terminal = self._terminal_value(successor, context.target_color)
            if terminal is not None:
                leaf_values.append(terminal)
                continue

            continuation_context = MoveContext(
                game_id=context.game_id,
                ply=context.ply + 2,
                seed=context.seed,
                target_color=context.target_color,
            )
            target_move = self.continuation.select_move(
                successor,
                continuation_context,
            )
            after_target = successor.copy(stack=True)
            after_target.push(target_move)
            terminal = self._terminal_value(after_target, context.target_color)
            if terminal is not None:
                leaf_values.append(terminal)
            else:
                leaf_values.append(
                    self.continuation.score_move(
                        successor,
                        target_move,
                        context.target_color,
                    )
                )
        return math.fsum(leaf_values) / len(leaf_values)

    def _sample_word(self, context: MoveContext, sample_index: int) -> int:
        payload = (
            f"two-turn-random-reply-v1|{self.config.seed}|{context.seed}|"
            f"{context.game_id}|{context.ply}|{sample_index}"
        ).encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    @classmethod
    def _terminal_value(
        cls,
        board: chess.Board,
        target_color: chess.Color,
    ) -> float | None:
        if board.is_checkmate():
            return (
                cls._SELF_MATE_VALUE
                if board.turn == target_color
                else cls._TARGET_WIN_VALUE
            )
        if board.is_game_over(claim_draw=False):
            return cls._DRAW_VALUE
        return None


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
