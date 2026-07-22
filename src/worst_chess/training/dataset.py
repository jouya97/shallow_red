"""Reproducible, framework-independent supervised chess datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.chess.actions import ActionEncodingError, decode_action, encode_move

DATASET_SCHEMA_VERSION = 1
"""Current on-disk JSONL schema version."""

Labeler = Callable[[chess.Board, MoveContext], chess.Move]
"""A policy used to label a generated position with one legal move."""

PathLike = str | os.PathLike[str]


class DatasetFormatError(ValueError):
    """Raised when a serialized dataset record is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class LabeledPosition:
    """One immutable legal policy target and its leakage-group identity."""

    fen: str
    target_color: chess.Color
    chosen_action: int
    source_id: str
    trajectory_id: str

    def __post_init__(self) -> None:
        if type(self.target_color) is not bool:
            raise TypeError("target_color must be chess.WHITE or chess.BLACK")
        _validate_identifier("source_id", self.source_id)
        _validate_identifier("trajectory_id", self.trajectory_id)
        try:
            board = chess.Board(self.fen)
        except ValueError as error:
            raise ValueError(f"invalid FEN: {self.fen!r}") from error
        if not board.is_valid():
            raise ValueError(f"invalid orthodox chess position: {self.fen!r}")
        if board.turn != self.target_color:
            raise ValueError("target_color must be the side to move in fen")
        try:
            decode_action(board, self.chosen_action)
        except ActionEncodingError as error:
            raise ValueError("chosen_action is not legal in fen") from error

    def board(self) -> chess.Board:
        """Reconstruct the position represented by this example."""

        return chess.Board(self.fen)

    def chosen_move(self) -> chess.Move:
        """Decode the legal policy target represented by this example."""

        return decode_action(self.board(), self.chosen_action)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Leakage-safe train, validation, and test partitions."""

    train: tuple[LabeledPosition, ...]
    validation: tuple[LabeledPosition, ...]
    test: tuple[LabeledPosition, ...]

    @property
    def val(self) -> tuple[LabeledPosition, ...]:
        """Short alias for the validation partition."""

        return self.validation


def generate_labeled_positions(
    labeler: Labeler,
    *,
    trajectory_count: int,
    positions_per_trajectory: int,
    seed: int,
    source_id: str,
    opening_plies: int = 2,
    target_colors: Sequence[chess.Color] = (chess.WHITE, chess.BLACK),
) -> tuple[LabeledPosition, ...]:
    """Generate labels from deterministic legal random trajectories.

    Trajectory moves are chosen by SHA256 over the seed, trajectory identity,
    ply, and FEN, then indexed into UCI-sorted legal moves.  This avoids Python
    hash or PRNG implementation dependence.  ``labeler`` receives an isolated
    board copy and a :class:`MoveContext`; a deterministic labeler therefore
    produces byte-for-byte reproducible examples.

    ``opening_plies`` are played before recording examples so trajectories do
    not all contribute the identical initial position.  Terminal positions are
    never labeled.  If a game ends early, that trajectory may contribute fewer
    than ``positions_per_trajectory`` examples.
    """

    _validate_nonnegative_int("trajectory_count", trajectory_count)
    _validate_positive_int("positions_per_trajectory", positions_per_trajectory)
    _validate_nonnegative_int("opening_plies", opening_plies)
    _validate_identifier("source_id", source_id)
    colors = tuple(target_colors)
    if not colors:
        raise ValueError("target_colors must contain at least one color")
    if any(type(color) is not bool for color in colors):
        raise TypeError("target_colors must contain only chess colors")

    examples: list[LabeledPosition] = []
    for trajectory_index in range(trajectory_count):
        trajectory_id = f"{source_id}/trajectory-{trajectory_index:06d}"
        target_color = colors[trajectory_index % len(colors)]
        board = chess.Board()

        for _ in range(opening_plies):
            if board.is_game_over(claim_draw=False):
                break
            board.push(
                _stable_legal_choice(
                    board,
                    seed=seed,
                    trajectory_id=trajectory_id,
                )
            )

        recorded_positions = 0
        while recorded_positions < positions_per_trajectory:
            if board.is_game_over(claim_draw=False):
                break
            if board.turn != target_color:
                board.push(
                    _stable_legal_choice(
                        board,
                        seed=seed,
                        trajectory_id=trajectory_id,
                    )
                )
                continue

            context = MoveContext(
                game_id=trajectory_id,
                ply=board.ply(),
                seed=_stable_int(seed, trajectory_id, board.ply(), board.fen()),
                target_color=target_color,
            )
            label_board = board.copy(stack=True)
            chosen_move = labeler(label_board, context)
            try:
                chosen_action = encode_move(board, chosen_move)
            except (ActionEncodingError, AttributeError) as error:
                raise ValueError(
                    "labeler must return a legal chess.Move without relying on "
                    "mutations to its board argument"
                ) from error

            examples.append(
                LabeledPosition(
                    fen=board.fen(en_passant="fen"),
                    target_color=target_color,
                    chosen_action=chosen_action,
                    source_id=source_id,
                    trajectory_id=trajectory_id,
                )
            )
            recorded_positions += 1
            board.push(
                _stable_legal_choice(
                    board,
                    seed=seed,
                    trajectory_id=trajectory_id,
                )
            )

    return tuple(examples)


def split_by_trajectory(
    positions: Iterable[LabeledPosition],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> DatasetSplit:
    """Split complete ``(source_id, trajectory_id)`` groups deterministically."""

    _validate_fractions(train_fraction, validation_fraction)
    groups: dict[tuple[str, str], list[LabeledPosition]] = {}
    for position in positions:
        if not isinstance(position, LabeledPosition):
            raise TypeError("positions must contain only LabeledPosition values")
        key = (position.source_id, position.trajectory_id)
        groups.setdefault(key, []).append(position)

    ordered_keys = sorted(
        groups,
        key=lambda key: (_stable_int(seed, *key), key),
    )
    train_count = int(len(ordered_keys) * train_fraction)
    validation_count = int(len(ordered_keys) * validation_fraction)
    train_end = train_count
    validation_end = train_end + validation_count

    def collect(keys: Sequence[tuple[str, str]]) -> tuple[LabeledPosition, ...]:
        return tuple(position for key in keys for position in groups[key])

    return DatasetSplit(
        train=collect(ordered_keys[:train_end]),
        validation=collect(ordered_keys[train_end:validation_end]),
        test=collect(ordered_keys[validation_end:]),
    )


def write_jsonl(path: PathLike, positions: Iterable[LabeledPosition]) -> None:
    """Atomically write a versioned dataset, replacing ``path`` on success."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for position in positions:
                if not isinstance(position, LabeledPosition):
                    raise TypeError(
                        "positions must contain only LabeledPosition values"
                    )
                json.dump(
                    _to_record(position),
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_jsonl(path: PathLike) -> tuple[LabeledPosition, ...]:
    """Read and validate every record in a versioned JSONL dataset."""

    positions: list[LabeledPosition] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw: object = json.loads(line)
                positions.append(_from_record(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise DatasetFormatError(
                    f"invalid dataset record at line {line_number}: {error}"
                ) from error
    return tuple(positions)


def _stable_legal_choice(
    board: chess.Board,
    *,
    seed: int,
    trajectory_id: str,
) -> chess.Move:
    legal_moves = sorted(board.legal_moves, key=chess.Move.uci)
    if not legal_moves:
        raise ValueError("cannot select a move from a terminal position")
    choice = _stable_int(seed, trajectory_id, board.ply(), board.fen())
    return legal_moves[choice % len(legal_moves)]


def _stable_int(*parts: object) -> int:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _to_record(position: LabeledPosition) -> dict[str, object]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "fen": position.fen,
        "target_color": "white" if position.target_color else "black",
        "chosen_action": position.chosen_action,
        "source_id": position.source_id,
        "trajectory_id": position.trajectory_id,
    }


def _from_record(raw: object) -> LabeledPosition:
    if not isinstance(raw, dict):
        raise DatasetFormatError("record must be a JSON object")
    record = cast(dict[object, object], raw)
    expected_fields = {
        "schema_version",
        "fen",
        "target_color",
        "chosen_action",
        "source_id",
        "trajectory_id",
    }
    if set(record) != expected_fields:
        raise DatasetFormatError(
            f"record fields must be exactly {sorted(expected_fields)!r}"
        )
    if type(record["schema_version"]) is not int:
        raise DatasetFormatError("schema_version must be an integer")
    if record["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetFormatError(
            f"unsupported schema_version {record['schema_version']!r}"
        )
    if record["target_color"] not in ("white", "black"):
        raise DatasetFormatError("target_color must be 'white' or 'black'")
    for field in ("fen", "source_id", "trajectory_id"):
        if not isinstance(record[field], str):
            raise DatasetFormatError(f"{field} must be a string")
    if type(record["chosen_action"]) is not int:
        raise DatasetFormatError("chosen_action must be an integer")

    return LabeledPosition(
        fen=cast(str, record["fen"]),
        target_color=record["target_color"] == "white",
        chosen_action=record["chosen_action"],
        source_id=cast(str, record["source_id"]),
        trajectory_id=cast(str, record["trajectory_id"]),
    )


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not math.isfinite(train_fraction) or not math.isfinite(validation_fraction):
        raise ValueError("split fractions must be finite")
    if train_fraction < 0 or validation_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if train_fraction + validation_fraction > 1:
        raise ValueError("train_fraction + validation_fraction must not exceed 1")


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DatasetFormatError",
    "DatasetSplit",
    "LabeledPosition",
    "Labeler",
    "generate_labeled_positions",
    "read_jsonl",
    "split_by_trajectory",
    "write_jsonl",
]
