import math

import chess
import torch

from worst_chess.agents.base import MoveContext
from worst_chess.training.dataset import generate_labeled_positions
from worst_chess.training.model import ModelConfig, PolicyValueNetwork
from worst_chess.training.trainer import (
    TrainingConfig,
    evaluate_policy,
    train_policy,
)


def first_legal(board: chess.Board, context: MoveContext) -> chess.Move:
    del context
    return min(board.legal_moves, key=chess.Move.uci)


def test_cpu_policy_training_is_finite_and_changes_parameters() -> None:
    positions = generate_labeled_positions(
        first_legal,
        trajectory_count=4,
        positions_per_trajectory=4,
        seed=5,
        source_id="trainer-test",
    )
    torch.manual_seed(9)
    model = PolicyValueNetwork(
        ModelConfig(channels=4, residual_blocks=1, value_channels=2, value_hidden=4)
    )
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    result = train_policy(
        model,
        positions[:12],
        positions[12:],
        config=TrainingConfig(epochs=2, batch_size=4, seed=11),
    )
    loss, accuracy = evaluate_policy(model, positions[12:], batch_size=4)

    assert len(result.epochs) == 2
    assert result.training_examples == 12
    assert result.validation_examples == 4
    assert result.elapsed_seconds > 0
    assert all(math.isfinite(epoch.train_loss) for epoch in result.epochs)
    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )
