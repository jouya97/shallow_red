"""Small deterministic supervised policy trainer for the GPU-gate pilot."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from worst_chess.chess.actions import legal_action_mask
from worst_chess.chess.observations import encode_observation
from worst_chess.training.dataset import LabeledPosition
from worst_chess.training.model import PolicyValueNetwork, mask_illegal_logits


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 20260721
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float | None
    validation_top1: float | None
    examples_per_second: float


@dataclass(frozen=True)
class TrainingResult:
    epochs: tuple[EpochMetrics, ...]
    training_examples: int
    validation_examples: int
    elapsed_seconds: float
    device: str


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device}")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS device requested but unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported training device: {device}")
    return device


def _batch_tensors(
    positions: list[LabeledPosition],
    indices: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = [positions[int(index)] for index in indices]
    observations = np.stack(
        [
            encode_observation(position.board(), position.target_color)
            for position in selected
        ]
    )
    masks = np.stack(
        [legal_action_mask(position.board()) for position in selected]
    )
    actions = [position.chosen_action for position in selected]
    return (
        torch.from_numpy(observations).to(device),
        torch.from_numpy(masks).to(device),
        torch.tensor(actions, dtype=torch.long, device=device),
    )


def _loss_and_correct(
    model: PolicyValueNetwork,
    observations: Tensor,
    masks: Tensor,
    actions: Tensor,
) -> tuple[Tensor, int]:
    logits, _ = model(observations)
    masked_logits = mask_illegal_logits(logits, masks)
    loss = torch.nn.functional.cross_entropy(masked_logits, actions)
    correct = int((masked_logits.argmax(dim=-1) == actions).sum().item())
    return loss, correct


def evaluate_policy(
    model: PolicyValueNetwork,
    positions: list[LabeledPosition] | tuple[LabeledPosition, ...],
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[float, float]:
    """Return masked cross-entropy and top-1 accuracy."""

    if not positions:
        raise ValueError("positions must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved = _resolve_device(device)
    materialized = list(positions)
    model = model.to(resolved)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(materialized), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(materialized)))
            observations, masks, actions = _batch_tensors(
                materialized, indices, resolved
            )
            loss, correct = _loss_and_correct(model, observations, masks, actions)
            batch_count = len(actions)
            total_loss += float(loss.item()) * batch_count
            total_correct += correct
    return total_loss / len(materialized), total_correct / len(materialized)


def train_policy(
    model: PolicyValueNetwork,
    training_positions: list[LabeledPosition] | tuple[LabeledPosition, ...],
    validation_positions: list[LabeledPosition] | tuple[LabeledPosition, ...] = (),
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the legal policy head to imitate selected losing moves."""

    settings = config or TrainingConfig()
    train_data = list(training_positions)
    validation_data = list(validation_positions)
    if not train_data:
        raise ValueError("training_positions must not be empty")
    device = _resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    all_metrics: list[EpochMetrics] = []
    training_started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        ordering = torch.randperm(len(train_data), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_data), settings.batch_size):
            indices = ordering[start : start + settings.batch_size]
            observations, masks, actions = _batch_tensors(train_data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss_and_correct(model, observations, masks, actions)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("training produced a non-finite loss")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip_norm
            )
            optimizer.step()
            batch_count = len(actions)
            total_loss += float(loss.item()) * batch_count
            seen += batch_count

        validation_loss: float | None = None
        validation_top1: float | None = None
        if validation_data:
            validation_loss, validation_top1 = evaluate_policy(
                model,
                validation_data,
                batch_size=settings.batch_size,
                device=str(device),
            )
        elapsed = time.perf_counter() - epoch_started
        rate = seen / elapsed
        if not math.isfinite(rate):
            raise FloatingPointError("invalid training throughput")
        all_metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=total_loss / seen,
                validation_loss=validation_loss,
                validation_top1=validation_top1,
                examples_per_second=rate,
            )
        )

    model.eval()
    return TrainingResult(
        epochs=tuple(all_metrics),
        training_examples=len(train_data),
        validation_examples=len(validation_data),
        elapsed_seconds=time.perf_counter() - training_started,
        device=str(device),
    )


__all__ = [
    "EpochMetrics",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_policy",
    "train_policy",
]
