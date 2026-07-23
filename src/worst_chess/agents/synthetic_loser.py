"""Diverse synthetic opponents that independently try to lose."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import chess

from worst_chess.agents.adapters import SelfishLoserOpponentAgent
from worst_chess.agents.base import Agent, AgentError, MoveContext
from worst_chess.agents.exploit import FrozenTargetExploitOpponentAgent
from worst_chess.agents.heuristic import HeuristicAgent, HeuristicWeights
from worst_chess.agents.opponent_model import (
    RandomReplySearchAgent,
    StalemateAwareRandomReplySearchAgent,
)
from worst_chess.agents.portfolio import RegimeSwitchingOpponentAgent


class ExploringLoserAgent:
    """Add reproducible exploration while rejecting avoidable wins and loops.

    This wrapper acts for the color identified by ``context.target_color``.
    Its exploration is deterministic under the complete move context, so
    paired evaluations and counterfactual rollouts remain reproducible.
    """

    def __init__(
        self,
        base: Agent,
        *,
        exploration_probability: float,
        salt: str,
    ) -> None:
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        if not salt:
            raise ValueError("salt must not be empty")
        self.base = base
        self.exploration_probability = exploration_probability
        self.salt = salt

    @property
    def name(self) -> str:
        percentage = round(self.exploration_probability * 100)
        return f"exploring_{self.base.name}_{percentage}pct"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("ExploringLoserAgent must act for the losing color")
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal:
            raise AgentError("ExploringLoserAgent cannot move from a terminal position")

        candidates = _safest_exploration_candidates(board, legal)
        digest = hashlib.sha256(
            (
                f"shallow-red-synthetic-loser-v1\0{self.salt}\0{context.seed}\0"
                f"{context.game_id}\0{context.ply}\0{board.fen()}"
            ).encode()
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / 2**64
        if sample < self.exploration_probability:
            return candidates[int.from_bytes(digest[8:16], "big") % len(candidates)]

        move = self.base.select_move(board.copy(stack=True), context)
        if not isinstance(move, chess.Move) or move not in legal:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(f"synthetic loser base returned illegal move {detail}")
        if move in candidates:
            return move
        return candidates[int.from_bytes(digest[16:24], "big") % len(candidates)]


def build_synthetic_loser_league(
    frozen_target: Agent | None = None,
    *,
    salt: str = "synthetic-loser-league-v1",
) -> Agent:
    """Build varied, anti-repetition opponents for games and rollouts."""

    losing_policies: tuple[Agent, ...] = (
        ExploringLoserAgent(
            HeuristicAgent(),
            exploration_probability=0.15,
            salt=f"{salt}/sacrifice",
        ),
        ExploringLoserAgent(
            HeuristicAgent(
                HeuristicWeights(
                    material_shed=8,
                    capturable_material=14,
                    moved_piece_sacrifice=18,
                    captured_enemy_material=-18,
                    king_ring_attack=70,
                    available_check=90,
                    missing_pawn_shield=50,
                )
            ),
            exploration_probability=0.30,
            salt=f"{salt}/king-exposure",
        ),
        ExploringLoserAgent(
            RandomReplySearchAgent(None, top_k=64),
            exploration_probability=0.35,
            salt=f"{salt}/random-reply",
        ),
        ExploringLoserAgent(
            StalemateAwareRandomReplySearchAgent(
                None,
                top_k=64,
                pressure_scale=4.0,
                pressure_min_material=2_000,
                cycle_penalty=1.0e15,
                tactical_mate_override=True,
            ),
            exploration_probability=0.20,
            salt=f"{salt}/tactical",
        ),
    )
    members: list[Agent] = [
        SelfishLoserOpponentAgent(policy) for policy in losing_policies
    ]
    if frozen_target is not None:
        members.append(
            FrozenTargetExploitOpponentAgent(
                frozen_target,
                candidate_limit=24,
            )
        )
    return RegimeSwitchingOpponentAgent(
        members,
        weights=(2, 2, 2, 2, 1) if frozen_target is not None else (1, 1, 1, 1),
        regime_plies=6,
        salt=salt,
    )


def _safest_exploration_candidates(
    board: chess.Board,
    legal: list[chess.Move],
) -> list[chess.Move]:
    candidates: list[tuple[chess.Move, chess.Board]] = []
    for move in legal:
        position = board.copy(stack=True)
        position.push(move)
        candidates.append((move, position))

    def retain(predicate: Callable[[chess.Board], bool]) -> None:
        nonlocal candidates
        filtered = [item for item in candidates if predicate(item[1])]
        if filtered:
            candidates = filtered

    retain(lambda position: not position.is_checkmate())
    retain(lambda position: not position.is_game_over(claim_draw=False))
    retain(lambda position: not position.is_repetition(2))
    return [move for move, _ in candidates]


__all__ = ["ExploringLoserAgent", "build_synthetic_loser_league"]
