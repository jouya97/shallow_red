from __future__ import annotations

import math

import chess
import torch

from worst_chess.agents.base import MoveContext
from worst_chess.training.model import ModelConfig, PolicyValueNetwork
from worst_chess.training.ranked_dataset import RankedPosition, rank_position
from worst_chess.training.ranked_trainer import (
    RankedTrainingConfig,
    evaluate_ranked,
    train_ranked,
)


def _positions(count: int, *, with_values: bool) -> tuple[RankedPosition, ...]:
    board = chess.Board()
    positions: list[RankedPosition] = []
    while len(positions) < count:
        if board.is_game_over(claim_draw=False):
            board = chess.Board()
        target_color = board.turn
        context = MoveContext(
            game_id=f"ranked-{len(positions)}",
            ply=board.ply(),
            seed=len(positions),
            target_color=target_color,
        )

        def scorer(
            scored_board: chess.Board, scored_context: MoveContext
        ) -> dict[chess.Move, float]:
            del scored_context
            ordered = sorted(scored_board.legal_moves, key=chess.Move.uci)
            return {move: -float(index) for index, move in enumerate(ordered)}

        positions.append(
            rank_position(
                board,
                target_color=target_color,
                scorer=scorer,
                context=context,
                source_id="ranked-trainer-test",
                trajectory_id=f"trajectory-{len(positions):03d}",
                value_target=1.0 if with_values else None,
            )
        )
        board.push(min(board.legal_moves, key=chess.Move.uci))
    return tuple(positions)


def test_ranked_training_is_finite_updates_policy_and_value() -> None:
    positions = _positions(12, with_values=True)
    torch.manual_seed(4)
    model = PolicyValueNetwork(
        ModelConfig(channels=4, residual_blocks=1, value_channels=2, value_hidden=4)
    )
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    initial = evaluate_ranked(model, positions[8:], batch_size=2)
    result = train_ranked(
        model,
        positions[:8],
        positions[8:],
        config=RankedTrainingConfig(epochs=2, batch_size=2, seed=7),
    )
    final = evaluate_ranked(model, positions[8:], batch_size=2)

    assert result.training_examples == 8
    assert result.validation_examples == 4
    assert len(result.epochs) == 2
    assert result.elapsed_seconds > 0
    assert all(math.isfinite(epoch.train_loss) for epoch in result.epochs)
    assert 0 <= initial.rank_one_accuracy <= 1
    assert 0 <= final.rank_one_accuracy <= 1
    assert 0 < final.mean_reciprocal_rank <= 1
    assert final.value_loss is not None
    assert final.value_mae is not None
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_ranked_evaluation_supports_missing_values() -> None:
    model = PolicyValueNetwork(
        ModelConfig(channels=4, residual_blocks=1, value_channels=2, value_hidden=4)
    )

    metrics = evaluate_ranked(model, _positions(3, with_values=False))

    assert metrics.examples == 3
    assert math.isfinite(metrics.loss)
    assert metrics.value_loss is None
    assert metrics.value_mae is None
