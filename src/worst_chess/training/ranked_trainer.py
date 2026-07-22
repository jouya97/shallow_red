"""Joint legal-move ranking and loser-value training for ranked datasets."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from worst_chess.chess.actions import ACTION_SPACE_SIZE, legal_action_mask
from worst_chess.chess.observations import encode_observation
from worst_chess.training.model import PolicyValueNetwork, mask_illegal_logits
from worst_chess.training.ranked_dataset import RankedPosition
from worst_chess.training.trainer import _resolve_device


@dataclass(frozen=True)
class RankedTrainingConfig:
    """Hyperparameters for soft rank distillation plus value regression."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    rank_temperature: float = 2.0
    value_loss_weight: float = 0.25
    seed: int = 20260721
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.gradient_clip_norm <= 0 or self.rank_temperature <= 0:
            raise ValueError(
                "gradient_clip_norm and rank_temperature must be positive"
            )
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight must be nonnegative")


@dataclass(frozen=True)
class RankedEvaluation:
    """Policy-ranking and optional value metrics over one partition."""

    examples: int
    loss: float
    policy_loss: float
    value_loss: float | None
    rank_one_accuracy: float
    mean_reciprocal_rank: float
    value_mae: float | None


@dataclass(frozen=True)
class RankedEpochMetrics:
    epoch: int
    train_loss: float
    validation: RankedEvaluation | None
    examples_per_second: float


@dataclass(frozen=True)
class RankedTrainingResult:
    epochs: tuple[RankedEpochMetrics, ...]
    training_examples: int
    validation_examples: int
    elapsed_seconds: float
    device: str


def _batch_tensors(
    positions: list[RankedPosition],
    indices: Tensor,
    device: torch.device,
    rank_temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    probabilities = np.zeros(
        (len(selected), ACTION_SPACE_SIZE), dtype=np.float32
    )
    ranks = np.zeros((len(selected), ACTION_SPACE_SIZE), dtype=np.int64)
    values = np.zeros(len(selected), dtype=np.float32)
    value_mask = np.zeros(len(selected), dtype=np.bool_)
    for row, position in enumerate(selected):
        weights = np.asarray(
            [
                math.exp(-(target.rank - 1) / rank_temperature)
                for target in position.move_targets
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        for target, probability in zip(
            position.move_targets, weights, strict=True
        ):
            probabilities[row, target.action] = probability
            ranks[row, target.action] = target.rank
        if position.value_target is not None:
            values[row] = position.value_target
            value_mask[row] = True
    return (
        torch.from_numpy(observations).to(device),
        torch.from_numpy(masks).to(device),
        torch.from_numpy(probabilities).to(device),
        torch.from_numpy(ranks).to(device),
        torch.from_numpy(values).to(device),
        torch.from_numpy(value_mask).to(device),
    )


def _loss_and_metrics(
    model: PolicyValueNetwork,
    observations: Tensor,
    masks: Tensor,
    targets: Tensor,
    ranks: Tensor,
    values: Tensor,
    value_mask: Tensor,
    *,
    value_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor | None, int, float, float, int]:
    logits, predicted_values = model(observations)
    masked_logits = mask_illegal_logits(logits, masks)
    log_probabilities = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    # Avoid the undefined 0 * -inf product on illegal actions.
    log_probabilities = log_probabilities.masked_fill(~masks.bool(), 0.0)
    policy_per_example = -(targets * log_probabilities).sum(dim=-1)
    policy_loss = policy_per_example.mean()

    value_loss: Tensor | None = None
    value_absolute_error = 0.0
    value_count = int(value_mask.sum().item())
    if value_count:
        selected_predictions = predicted_values.squeeze(-1)[value_mask]
        selected_values = values[value_mask]
        value_loss = torch.nn.functional.mse_loss(
            selected_predictions, selected_values
        )
        value_absolute_error = float(
            torch.abs(selected_predictions.detach() - selected_values).sum().item()
        )
    total_loss = policy_loss
    if value_loss is not None:
        total_loss = total_loss + value_loss_weight * value_loss

    predicted_actions = masked_logits.argmax(dim=-1)
    predicted_ranks = ranks.gather(1, predicted_actions.unsqueeze(1)).squeeze(1)
    rank_one = int((predicted_ranks == 1).sum().item())
    reciprocal_rank_sum = float(
        (1.0 / predicted_ranks.to(torch.float32)).sum().item()
    )
    return (
        total_loss,
        policy_loss,
        value_loss,
        rank_one,
        reciprocal_rank_sum,
        value_absolute_error,
        value_count,
    )


def evaluate_ranked(
    model: PolicyValueNetwork,
    positions: list[RankedPosition] | tuple[RankedPosition, ...],
    *,
    batch_size: int = 256,
    rank_temperature: float = 2.0,
    value_loss_weight: float = 0.25,
    device: str = "cpu",
) -> RankedEvaluation:
    """Evaluate soft rank distillation, rank-one accuracy, and value error."""

    if not positions:
        raise ValueError("positions must not be empty")
    if batch_size < 1 or rank_temperature <= 0 or value_loss_weight < 0:
        raise ValueError("invalid evaluation hyperparameters")
    resolved = _resolve_device(device)
    materialized = list(positions)
    model.to(resolved).eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    rank_one = 0
    reciprocal_rank = 0.0
    value_absolute_error = 0.0
    value_count = 0
    with torch.no_grad():
        for start in range(0, len(materialized), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(materialized)))
            batch = _batch_tensors(
                materialized, indices, resolved, rank_temperature
            )
            metrics = _loss_and_metrics(
                model, *batch, value_loss_weight=value_loss_weight
            )
            count = len(indices)
            total_loss += float(metrics[0].item()) * count
            total_policy_loss += float(metrics[1].item()) * count
            if metrics[2] is not None:
                batch_value_count = metrics[6]
                total_value_loss += float(metrics[2].item()) * batch_value_count
            rank_one += metrics[3]
            reciprocal_rank += metrics[4]
            value_absolute_error += metrics[5]
            value_count += metrics[6]
    count = len(materialized)
    return RankedEvaluation(
        examples=count,
        loss=total_loss / count,
        policy_loss=total_policy_loss / count,
        value_loss=(total_value_loss / value_count if value_count else None),
        rank_one_accuracy=rank_one / count,
        mean_reciprocal_rank=reciprocal_rank / count,
        value_mae=(value_absolute_error / value_count if value_count else None),
    )


def train_ranked(
    model: PolicyValueNetwork,
    training_positions: list[RankedPosition] | tuple[RankedPosition, ...],
    validation_positions: list[RankedPosition] | tuple[RankedPosition, ...] = (),
    *,
    config: RankedTrainingConfig | None = None,
) -> RankedTrainingResult:
    """Train a policy/value network using all legal teacher ranks."""

    settings = config or RankedTrainingConfig()
    training = list(training_positions)
    validation = list(validation_positions)
    if not training:
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
    started = time.perf_counter()
    epochs: list[RankedEpochMetrics] = []
    for epoch in range(1, settings.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        ordering = torch.randperm(len(training), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(training), settings.batch_size):
            indices = ordering[start : start + settings.batch_size]
            batch = _batch_tensors(
                training, indices, device, settings.rank_temperature
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_and_metrics(
                model,
                *batch,
                value_loss_weight=settings.value_loss_weight,
            )[0]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("ranked training produced a non-finite loss")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip_norm
            )
            optimizer.step()
            batch_count = len(indices)
            total_loss += float(loss.item()) * batch_count
            seen += batch_count
        validation_metrics = None
        if validation:
            validation_metrics = evaluate_ranked(
                model,
                validation,
                batch_size=settings.batch_size,
                rank_temperature=settings.rank_temperature,
                value_loss_weight=settings.value_loss_weight,
                device=str(device),
            )
        elapsed = time.perf_counter() - epoch_started
        rate = seen / elapsed
        if not math.isfinite(rate):
            raise FloatingPointError("invalid ranked training throughput")
        epochs.append(
            RankedEpochMetrics(
                epoch=epoch,
                train_loss=total_loss / seen,
                validation=validation_metrics,
                examples_per_second=rate,
            )
        )
    model.eval()
    return RankedTrainingResult(
        epochs=tuple(epochs),
        training_examples=len(training),
        validation_examples=len(validation),
        elapsed_seconds=time.perf_counter() - started,
        device=str(device),
    )


__all__ = [
    "RankedEpochMetrics",
    "RankedEvaluation",
    "RankedTrainingConfig",
    "RankedTrainingResult",
    "evaluate_ranked",
    "train_ranked",
]
