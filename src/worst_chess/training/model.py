"""Compact policy/value network and versioned checkpoint helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from worst_chess.chess.actions import ACTION_PLANES, ACTION_SPACE_SIZE
from worst_chess.chess.observations import OBSERVATION_PLANES, OBSERVATION_SHAPE

CHECKPOINT_FORMAT = "worst_chess.policy_value"
CHECKPOINT_SCHEMA_VERSION = 1
ACTION_LAYOUT = "from_square*73+plane"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters small enough for CPU smoke training."""

    channels: int = 32
    residual_blocks: int = 2
    value_channels: int = 8
    value_hidden: int = 64

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")


class ResidualBlock(nn.Module):
    """Two-convolution residual block without size-changing operations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.activation(self.conv1(inputs))
        residual = self.conv2(residual)
        return cast(Tensor, self.activation(inputs + residual))


class PolicyValueNetwork(nn.Module):
    """Residual CNN producing absolute-board action logits and loser value."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                OBSERVATION_PLANES,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(self.config.residual_blocks))
        )

        # Spatial dimensions remain rank x file.  forward() permutes the policy
        # tensor so plane is the fastest-varying dimension for every square.
        self.policy_head = nn.Conv2d(channels, ACTION_PLANES, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, self.config.value_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(self.config.value_channels * 8 * 8, self.config.value_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if observations.ndim != 4 or tuple(observations.shape[1:]) != OBSERVATION_SHAPE:
            raise ValueError(
                "observations must have shape (batch, 21, 8, 8), got "
                f"{tuple(observations.shape)}"
            )

        features = self.residual_tower(self.stem(observations))
        policy_planes = self.policy_head(features)
        # Input rows are ranks and columns are files.  This creates the exact
        # index ((rank * 8 + file) * 73 + plane) used by encode_move().
        policy_logits = policy_planes.permute(0, 2, 3, 1).contiguous().view(
            observations.shape[0], ACTION_SPACE_SIZE
        )
        value = self.value_head(features)
        return policy_logits, value


def mask_illegal_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Replace illegal logits with ``-inf`` after validating legal actions.

    A one-dimensional mask is broadcast across a batch.  Otherwise the mask
    must have exactly the logits' shape, allowing each position in a batch to
    carry its own legal moves.
    """

    if not policy_logits.is_floating_point():
        raise ValueError("policy_logits must be floating point")
    if policy_logits.ndim < 1 or policy_logits.shape[-1] != ACTION_SPACE_SIZE:
        raise ValueError(
            f"policy_logits must end in {ACTION_SPACE_SIZE} actions, got "
            f"{tuple(policy_logits.shape)}"
        )

    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    if mask.shape == (ACTION_SPACE_SIZE,):
        mask = mask.expand_as(policy_logits)
    elif mask.shape != policy_logits.shape:
        raise ValueError(
            "legal_mask must be one-dimensional or match policy_logits; got "
            f"{tuple(mask.shape)} and {tuple(policy_logits.shape)}"
        )

    rows = mask.reshape(-1, ACTION_SPACE_SIZE)
    if not bool(torch.all(rows.any(dim=-1)).item()):
        raise ValueError("every position must have at least one legal action")
    return policy_logits.masked_fill(~mask, -torch.inf)


def checkpoint_schema_metadata(config: ModelConfig) -> dict[str, Any]:
    """Return the architecture and project schemas persisted in checkpoints."""

    return {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(config),
        "project_schema": {
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_space_size": ACTION_SPACE_SIZE,
            "action_planes": ACTION_PLANES,
            "action_layout": ACTION_LAYOUT,
        },
    }


def save_checkpoint(
    model: PolicyValueNetwork,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save weights plus all information needed to rebuild a model."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = checkpoint_schema_metadata(model.config)
    payload: dict[str, Any] = {
        **schema,
        "metadata": dict(sorted((metadata or {}).items())),
        # CPU tensors make a checkpoint portable across CPU, MPS, and CUDA.
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyValueNetwork, dict[str, Any]]:
    """Load and strictly validate a versioned policy/value checkpoint."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint has an unsupported format")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint has an unsupported schema version")

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint is missing architecture configuration")
    try:
        config = ModelConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint architecture is invalid") from exc

    expected_schema = checkpoint_schema_metadata(config)["project_schema"]
    if checkpoint.get("project_schema") != expected_schema:
        raise ValueError("checkpoint project schema does not match this build")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing a state_dict")

    model = PolicyValueNetwork(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    return model, dict(metadata)


# Concise aliases for experiment code that uses architecture-oriented names.
ChessPolicyValueNet = PolicyValueNetwork
WorstChessModel = PolicyValueNetwork
apply_legal_action_mask = mask_illegal_logits

__all__ = [
    "ACTION_LAYOUT",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "ChessPolicyValueNet",
    "ModelConfig",
    "PolicyValueNetwork",
    "ResidualBlock",
    "WorstChessModel",
    "apply_legal_action_mask",
    "checkpoint_schema_metadata",
    "load_checkpoint",
    "mask_illegal_logits",
    "save_checkpoint",
]
