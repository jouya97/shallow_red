from __future__ import annotations

import chess
import pytest
import torch
from torch import Tensor, nn

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.agents.neural import NeuralAgent
from worst_chess.chess.actions import ACTION_SPACE_SIZE, encode_move
from worst_chess.chess.observations import OBSERVATION_SHAPE
from worst_chess.training.model import ModelConfig, PolicyValueNetwork


class FixedPolicy(PolicyValueNetwork):
    def __init__(self, preferred_action: int | None) -> None:
        nn.Module.__init__(self)
        self.config = ModelConfig(channels=1, residual_blocks=1)
        self.preferred_action = preferred_action
        self.last_observation: Tensor | None = None

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        self.last_observation = observations.detach().cpu().clone()
        logits = torch.zeros(
            observations.shape[0], ACTION_SPACE_SIZE, device=observations.device
        )
        if self.preferred_action is not None:
            logits[:, self.preferred_action] = 100.0
        return logits, torch.zeros(observations.shape[0], 1, device=observations.device)


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext(game_id="neural", ply=0, seed=9, target_color=color)


def test_neural_agent_selects_preferred_legal_move_without_mutation() -> None:
    board = chess.Board()
    expected = chess.Move.from_uci("e2e4")
    model = FixedPolicy(encode_move(board, expected))
    agent = NeuralAgent(model, device="cpu")
    original_fen = board.fen()

    selected = agent.select_move(board, _context())

    assert selected == expected
    assert selected in board.legal_moves
    assert board.fen() == original_fen
    assert model.training is False
    assert model.last_observation is not None
    assert tuple(model.last_observation.shape[1:]) == OBSERVATION_SHAPE


def test_neural_agent_masks_illegal_high_logit() -> None:
    board = chess.Board()
    illegal = 0  # a1's north-one plane is illegal in the initial position.
    model = FixedPolicy(illegal)

    selected = NeuralAgent(model).select_move(board, _context())

    expected = min(board.legal_moves, key=lambda move: encode_move(board, move))
    assert selected == expected


def test_neural_agent_ranks_all_legal_moves_and_limits_top_k() -> None:
    board = chess.Board()
    preferred = chess.Move.from_uci("e2e4")
    model = FixedPolicy(encode_move(board, preferred))
    agent = NeuralAgent(model)

    ranked = agent.rank_moves(board, _context(), top_k=3)
    all_ranked = agent.rank_moves(board, _context())

    assert len(ranked) == 3
    assert ranked[0].move == preferred
    assert len(all_ranked) == board.legal_moves.count()
    assert {item.move for item in all_ranked} == set(board.legal_moves)
    assert all_ranked[1:] == tuple(
        sorted(all_ranked[1:], key=lambda item: item.action)
    )


def test_neural_agent_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        NeuralAgent(FixedPolicy(None)).rank_moves(
            chess.Board(), _context(), top_k=0
        )


def test_neural_agent_encodes_from_target_perspective() -> None:
    board = chess.Board()
    model = FixedPolicy(None)

    NeuralAgent(model).select_move(board, _context(chess.BLACK))

    assert model.last_observation is not None
    # From Black's mirrored perspective, its a7 pawn occupies a rank-one cell.
    assert model.last_observation[0, 0, 1, 0] == 1.0


def test_neural_agent_rejects_terminal_position() -> None:
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    agent = NeuralAgent(FixedPolicy(None))

    with pytest.raises(AgentError, match="terminal"):
        agent.select_move(board, _context(chess.BLACK))


def test_neural_agent_reports_unavailable_accelerator() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(AgentError, match="CUDA.*unavailable"):
            NeuralAgent(FixedPolicy(None), device="cuda")
