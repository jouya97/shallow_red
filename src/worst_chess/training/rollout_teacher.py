"""Lexicographic counterfactual rollout targets for losing chess.

The primary statistic is the number of rollouts in which the designated
target is checkmated.  Only when that count ties does time-to-selfmate affect
the ordering.  A mixed-radix integer then adds accidental wins and truncations
as lower-priority tie breakers without allowing them to reverse either primary
criterion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import chess

from worst_chess.agents.base import AgentError, MoveContext

MoveSelector = Callable[[chess.Board, MoveContext], chess.Move]


class SelectorAgent(Protocol):
    """Minimal structural interface accepted in place of a selector callable."""

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        """Return one legal move without relying on mutation of ``board``."""


SelectorLike = MoveSelector | SelectorAgent


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Fixed sampling and horizon parameters for counterfactual rollouts."""

    rollouts: int = 8
    max_plies: int = 160
    seed: int = 20260721

    def __post_init__(self) -> None:
        if type(self.rollouts) is not int or self.rollouts <= 0:
            raise ValueError("rollouts must be a positive integer")
        if type(self.max_plies) is not int or self.max_plies <= 0:
            raise ValueError("max_plies must be a positive integer")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        # All ranking scores must survive the scorer's float API exactly.
        base = self.rollouts * self.max_plies + 1
        radix = self.rollouts + 1
        maximum_score = self.rollouts * base * radix * radix
        if maximum_score > 2**53:
            raise ValueError(
                "rollouts and max_plies produce scores larger than the exact "
                "integer range of a float"
            )


@dataclass(frozen=True, slots=True)
class RolloutMoveScore:
    """Auditable outcome counts and ranking score for one candidate move."""

    move: chess.Move
    selfmates: int
    selfmate_plies_sum: int
    target_wins: int
    draws: int
    truncations: int
    ranking_score: float


def rollout_ranking_score(
    *,
    selfmates: int,
    selfmate_plies_sum: int,
    target_wins: int,
    truncations: int,
    config: RolloutConfig,
) -> float:
    """Encode ``(selfmates, -speed, -wins, -truncations)`` exactly.

    The number of rollouts is fixed, so equal selfmate counts make a smaller
    successful-ply sum equivalent to a smaller conditional mean.  The radix
    construction guarantees that one additional selfmate dominates the full
    possible range of all lower-priority fields.
    """

    counts = {
        "selfmates": selfmates,
        "selfmate_plies_sum": selfmate_plies_sum,
        "target_wins": target_wins,
        "truncations": truncations,
    }
    if any(type(value) is not int for value in counts.values()):
        raise TypeError("rollout ranking counts must be integers")
    if selfmates < 0 or target_wins < 0 or truncations < 0:
        raise ValueError("rollout outcome counts must be non-negative")
    if selfmates + target_wins + truncations > config.rollouts:
        raise ValueError("rollout outcome counts exceed configured rollouts")
    if selfmate_plies_sum < selfmates:
        raise ValueError("each successful rollout must take at least one ply")
    if selfmate_plies_sum > selfmates * config.max_plies:
        raise ValueError("selfmate_plies_sum exceeds the configured horizon")

    speed_radix = config.rollouts * config.max_plies + 1
    outcome_radix = config.rollouts + 1
    primary_and_speed = selfmates * speed_radix - selfmate_plies_sum
    score = (
        (primary_and_speed * outcome_radix - target_wins) * outcome_radix
        - truncations
    )
    if abs(score) > 2**53 or not math.isfinite(float(score)):
        raise ValueError("rollout ranking score is not exactly representable")
    return float(score)


class LexicographicRolloutScorer:
    """Score every legal target move using matched counterfactual rollouts.

    The same rollout-specific game id and pseudo-random seed are reused across
    candidate actions.  This supplies common random numbers to policies that
    consume :class:`MoveContext`, reducing comparison noise without sharing
    mutable boards between candidates.
    """

    def __init__(
        self,
        target_policy: SelectorLike,
        opponent_policy: SelectorLike,
        config: RolloutConfig | None = None,
    ) -> None:
        self.target_selector = _coerce_selector("target_policy", target_policy)
        self.opponent_selector = _coerce_selector(
            "opponent_policy", opponent_policy
        )
        self.config = config or RolloutConfig()

    def __call__(
        self, board: chess.Board, context: MoveContext
    ) -> Mapping[chess.Move, float]:
        """Return finite loser-oriented scores for every legal move."""

        return self.score_moves(board, context)

    def score_moves(
        self, board: chess.Board, context: MoveContext
    ) -> dict[chess.Move, float]:
        """Return all legal move scores, with larger meaning better at losing."""

        return {
            item.move: item.ranking_score
            for item in self.evaluate_moves(board, context)
        }

    def evaluate_moves(
        self, board: chess.Board, context: MoveContext
    ) -> tuple[RolloutMoveScore, ...]:
        """Return UCI-ordered, auditable rollout summaries for all legal moves."""

        _validate_scoring_position(board, context)
        original_fen = board.fen(en_passant="fen")
        original_stack = tuple(board.move_stack)
        candidates = tuple(sorted(board.legal_moves, key=chess.Move.uci))
        summaries = tuple(
            self._evaluate_candidate(board, context, move) for move in candidates
        )
        if (
            board.fen(en_passant="fen") != original_fen
            or tuple(board.move_stack) != original_stack
        ):
            raise RuntimeError("rollout scoring mutated the input board")
        return summaries

    def _evaluate_candidate(
        self,
        board: chess.Board,
        context: MoveContext,
        candidate: chess.Move,
    ) -> RolloutMoveScore:
        selfmates = 0
        selfmate_plies_sum = 0
        target_wins = 0
        draws = 0
        truncations = 0
        for rollout_index in range(self.config.rollouts):
            position = board.copy(stack=True)
            position.push(candidate)
            played_plies = 1
            rollout_game_id = (
                f"{context.game_id}/lexicographic-rollout-{rollout_index:04d}"
            )
            rollout_seed = _stable_int(
                self.config.seed,
                context.seed,
                context.game_id,
                rollout_index,
            )

            while (
                played_plies < self.config.max_plies
                and not position.is_game_over(claim_draw=False)
            ):
                move_context = MoveContext(
                    game_id=rollout_game_id,
                    ply=position.ply(),
                    seed=_stable_int(rollout_seed, played_plies),
                    target_color=context.target_color,
                )
                is_target_turn = position.turn == context.target_color
                selector = (
                    self.target_selector
                    if is_target_turn
                    else self.opponent_selector
                )
                role = "target" if is_target_turn else "opponent"
                move = _select_legal_move(
                    selector, position, move_context, role=role
                )
                position.push(move)
                played_plies += 1

            outcome = position.outcome(claim_draw=False)
            if outcome is None:
                truncations += 1
            elif outcome.winner is None:
                draws += 1
            elif outcome.winner == context.target_color:
                target_wins += 1
            else:
                selfmates += 1
                selfmate_plies_sum += played_plies

        if (
            selfmates + target_wins + draws + truncations
            != self.config.rollouts
        ):
            raise RuntimeError("rollout outcomes do not sum to rollout count")
        ranking_score = rollout_ranking_score(
            selfmates=selfmates,
            selfmate_plies_sum=selfmate_plies_sum,
            target_wins=target_wins,
            truncations=truncations,
            config=self.config,
        )
        return RolloutMoveScore(
            move=candidate,
            selfmates=selfmates,
            selfmate_plies_sum=selfmate_plies_sum,
            target_wins=target_wins,
            draws=draws,
            truncations=truncations,
            ranking_score=ranking_score,
        )


def _coerce_selector(name: str, policy: SelectorLike) -> MoveSelector:
    method = getattr(policy, "select_move", None)
    if callable(method):
        return cast(MoveSelector, method)
    if callable(policy):
        return policy
    raise TypeError(f"{name} must be an Agent or move-selector callable")


def _select_legal_move(
    selector: MoveSelector,
    board: chess.Board,
    context: MoveContext,
    *,
    role: str,
) -> chess.Move:
    if (board.turn == context.target_color) != (role == "target"):
        raise RuntimeError(f"{role} selector called for the wrong side")
    selection_board = board.copy(stack=True)
    move = selector(selection_board, context)
    if not isinstance(move, chess.Move) or move not in board.legal_moves:
        detail = move.uci() if isinstance(move, chess.Move) else repr(move)
        raise AgentError(f"{role} rollout selector returned illegal move {detail}")
    return move


def _validate_scoring_position(
    board: chess.Board, context: MoveContext
) -> None:
    if not isinstance(context, MoveContext):
        raise TypeError("context must be a MoveContext")
    if board.turn != context.target_color:
        raise AgentError(
            "LexicographicRolloutScorer must score the designated target's turn"
        )
    if board.is_game_over(claim_draw=False):
        raise AgentError("LexicographicRolloutScorer cannot score a terminal position")


def _stable_int(*parts: object) -> int:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "LexicographicRolloutScorer",
    "MoveSelector",
    "RolloutConfig",
    "RolloutMoveScore",
    "SelectorAgent",
    "SelectorLike",
    "rollout_ranking_score",
]
