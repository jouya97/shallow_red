"""Framework-independent data generation and training infrastructure."""

from worst_chess.training.dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetFormatError,
    DatasetSplit,
    LabeledPosition,
    Labeler,
    generate_labeled_positions,
    read_jsonl,
    split_by_trajectory,
    write_jsonl,
)
from worst_chess.training.model import ModelConfig, PolicyValueNetwork
from worst_chess.training.ranked_dataset import (
    RankedAction,
    RankedPosition,
    generate_ranked_trajectories,
    read_ranked_jsonl,
    split_ranked_by_trajectory,
    write_ranked_jsonl,
)
from worst_chess.training.ranked_trainer import (
    RankedTrainingConfig,
    evaluate_ranked,
    train_ranked,
)
from worst_chess.training.trainer import (
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    evaluate_policy,
    train_policy,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DatasetFormatError",
    "DatasetSplit",
    "LabeledPosition",
    "Labeler",
    "ModelConfig",
    "PolicyValueNetwork",
    "RankedAction",
    "RankedPosition",
    "RankedTrainingConfig",
    "EpochMetrics",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_policy",
    "evaluate_ranked",
    "generate_labeled_positions",
    "generate_ranked_trajectories",
    "read_jsonl",
    "read_ranked_jsonl",
    "split_by_trajectory",
    "split_ranked_by_trajectory",
    "train_policy",
    "train_ranked",
    "write_jsonl",
    "write_ranked_jsonl",
]
