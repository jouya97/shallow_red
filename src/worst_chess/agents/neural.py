"""Inference agent backed by the compact policy/value network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.chess.neural_actions import (
    ABSOLUTE_ACTION_ORIENTATION,
    ACTION_ORIENTATION_METADATA_KEY,
    decode_neural_action,
    neural_legal_action_mask,
    validate_action_orientation,
)
from worst_chess.chess.observations import encode_observation
from worst_chess.training.model import (
    PolicyValueNetwork,
    load_checkpoint,
    mask_illegal_logits,
)


@dataclass(frozen=True, slots=True)
class PolicyMove:
    """One legal move, model-coordinate action, and unnormalized logit."""

    move: chess.Move
    action: int
    logit: float


class NeuralAgent:
    """Select the highest-logit legal action with deterministic tie-breaking."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        device: str | torch.device = "cpu",
        agent_name: str = "neural",
        action_orientation: str = ABSOLUTE_ACTION_ORIENTATION,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._name = agent_name
        try:
            self.action_orientation = validate_action_orientation(
                action_orientation
            )
        except ValueError as error:
            raise AgentError(str(error)) from error

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        agent_name: str = "neural",
    ) -> NeuralAgent:
        resolved = _resolve_device(device)
        model, metadata = load_checkpoint(path, device=resolved)
        orientation = metadata.get(
            ACTION_ORIENTATION_METADATA_KEY,
            ABSOLUTE_ACTION_ORIENTATION,
        )
        if not isinstance(orientation, str):
            raise AgentError(
                f"checkpoint {ACTION_ORIENTATION_METADATA_KEY} must be a string"
            )
        return cls(
            model,
            device=resolved,
            agent_name=agent_name,
            action_orientation=orientation,
        )

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        return self.rank_moves(board, context, top_k=1)[0].move

    def rank_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        top_k: int | None = None,
    ) -> tuple[PolicyMove, ...]:
        """Rank legal moves by logit with stable model-action-index ties."""

        if not any(board.legal_moves):
            raise AgentError("NeuralAgent cannot move from a terminal position")
        legal_count = board.legal_moves.count()
        if top_k is None:
            count = legal_count
        elif top_k < 1:
            raise ValueError("top_k must be positive")
        else:
            count = min(top_k, legal_count)

        observation = encode_observation(board, context.target_color)
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(
            neural_legal_action_mask(board, self.action_orientation)
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            policy_logits, _ = self.model(observation_tensor)
            masked_logits = mask_illegal_logits(policy_logits, mask)
            row = masked_logits[0].detach().cpu()
        ranked_actions = sorted(
            (int(action) for action in mask.nonzero().flatten().detach().cpu()),
            key=lambda action: (-float(row[action].item()), action),
        )[:count]
        ranked: list[PolicyMove] = []
        for action in ranked_actions:
            move = decode_neural_action(board, action, self.action_orientation)
            if move not in board.legal_moves:  # Defensive boundary assertion.
                raise AgentError(f"neural policy decoded illegal move {move.uci()}")
            ranked.append(
                PolicyMove(move=move, action=action, logit=float(row[action].item()))
            )
        return tuple(ranked)


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise AgentError(f"CUDA device requested but unavailable: {resolved}")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise AgentError("MPS device requested but unavailable")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise AgentError(f"unsupported neural inference device: {resolved}")
    return resolved


__all__ = ["NeuralAgent", "PolicyMove"]
