from __future__ import annotations

import pytest
import torch

from worst_chess.chess.actions import ACTION_PLANES, ACTION_SPACE_SIZE
from worst_chess.chess.observations import OBSERVATION_SHAPE
from worst_chess.training.model import (
    ACTION_LAYOUT,
    CHECKPOINT_SCHEMA_VERSION,
    ModelConfig,
    PolicyValueNetwork,
    load_checkpoint,
    mask_illegal_logits,
    save_checkpoint,
)


def test_model_output_shapes_and_tanh_value() -> None:
    model = PolicyValueNetwork(
        ModelConfig(channels=8, residual_blocks=1, value_channels=2, value_hidden=8)
    )
    observations = torch.randn(3, *OBSERVATION_SHAPE)

    policy, value = model(observations)

    assert policy.shape == (3, ACTION_SPACE_SIZE)
    assert value.shape == (3, 1)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_flattening_matches_from_square_times_73_plus_plane() -> None:
    model = PolicyValueNetwork(
        ModelConfig(channels=4, residual_blocks=1, value_channels=1, value_hidden=4)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head.bias.copy_(torch.arange(ACTION_PLANES, dtype=torch.float32))

    policy, _ = model(torch.zeros(1, *OBSERVATION_SHAPE))

    for square in (0, 1, 9, 63):
        for plane in (0, 7, 42, 72):
            action = square * ACTION_PLANES + plane
            assert policy[0, action].item() == plane


def test_mask_illegal_logits_blocks_illegal_actions_and_preserves_legal() -> None:
    logits = torch.arange(ACTION_SPACE_SIZE, dtype=torch.float32).repeat(2, 1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 3] = True
    mask[1, 7] = True

    masked = mask_illegal_logits(logits, mask)

    assert masked[0, 3] == logits[0, 3]
    assert masked[1, 7] == logits[1, 7]
    assert torch.isneginf(masked[0, 4])
    assert torch.argmax(masked, dim=-1).tolist() == [3, 7]


def test_mask_illegal_logits_rejects_position_without_legal_action() -> None:
    logits = torch.zeros(1, ACTION_SPACE_SIZE)
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal"):
        mask_illegal_logits(logits, mask)


def test_checkpoint_round_trip_restores_exact_outputs_and_metadata(tmp_path) -> None:
    torch.manual_seed(17)
    config = ModelConfig(
        channels=6, residual_blocks=1, value_channels=2, value_hidden=7
    )
    model = PolicyValueNetwork(config).eval()
    observations = torch.randn(2, *OBSERVATION_SHAPE)
    expected = model(observations)
    path = tmp_path / "model.pt"
    second_path = tmp_path / "model-copy.pt"

    save_checkpoint(model, path, metadata={"run": "cpu-smoke", "step": 11})
    # Metadata keys are canonicalized and tensor traversal follows state_dict
    # order, so identical model state produces byte-identical artifacts.
    save_checkpoint(model, second_path, metadata={"step": 11, "run": "cpu-smoke"})
    loaded, metadata = load_checkpoint(path)
    actual = loaded(observations)
    raw = torch.load(path, weights_only=True)

    assert loaded.config == config
    assert metadata == {"run": "cpu-smoke", "step": 11}
    assert raw["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert raw["project_schema"]["action_layout"] == ACTION_LAYOUT
    assert raw["project_schema"]["action_space_size"] == ACTION_SPACE_SIZE
    assert path.read_bytes() == second_path.read_bytes()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_model_rejects_wrong_observation_shape() -> None:
    model = PolicyValueNetwork(ModelConfig(channels=4, residual_blocks=1))
    with pytest.raises(ValueError, match="21, 8, 8"):
        model(torch.zeros(1, 20, 8, 8))
