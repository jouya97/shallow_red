"""Versioned all-legal-move teacher datasets for policy ranking.

Every score in this module uses one orientation: a larger number is better
for ``target_color``, the player trying to lose the chess game.  Rank one is
therefore the teacher's best losing move.  Equal scores receive the same
dense rank, and move targets are stored in ascending action order so ties do
not introduce ordering nondeterminism.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import chess

from worst_chess.agents.base import MoveContext
from worst_chess.chess.actions import ActionEncodingError, encode_move
from worst_chess.objective.rewards import terminal_utility

RANKED_DATASET_SCHEMA = "worst-chess-ranked"
RANKED_DATASET_SCHEMA_VERSION = 2

PathLike = str | os.PathLike[str]
LegalMoveScorer = Callable[
    [chess.Board, MoveContext], Mapping[chess.Move, float]
]
"""Score every legal move; higher must mean better for the designated loser."""
MoveSelector = Callable[[chess.Board, MoveContext], chess.Move]
"""Select one legal move without depending on mutation of the supplied board."""


class RankedDatasetFormatError(ValueError):
    """Raised when a ranked JSONL record is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class RankedAction:
    """One legal action and its teacher score and dense rank."""

    action: int
    teacher_score: float
    rank: int

    def __post_init__(self) -> None:
        if type(self.action) is not int:
            raise TypeError("action must be an integer")
        if isinstance(self.teacher_score, bool) or not isinstance(
            self.teacher_score, (int, float)
        ):
            raise TypeError("teacher_score must be a real number")
        if not math.isfinite(self.teacher_score):
            raise ValueError("teacher_score must be finite")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class RankedPosition:
    """Teacher scores for every legal action in one target-to-move position.

    ``value_target`` is optional because truncated trajectories have no known
    result.  When present it is from the designated loser's perspective:
    ``+1`` means the target was checkmated, ``0`` a draw, and ``-1`` a target
    win.  Intermediate values are allowed for bootstrapped targets.
    """

    fen: str
    target_color: chess.Color
    move_targets: tuple[RankedAction, ...]
    source_id: str
    trajectory_id: str
    value_target: float | None = None

    def __post_init__(self) -> None:
        if type(self.target_color) is not bool:
            raise TypeError("target_color must be chess.WHITE or chess.BLACK")
        _validate_identifier("source_id", self.source_id)
        _validate_identifier("trajectory_id", self.trajectory_id)
        if not isinstance(self.move_targets, tuple):
            raise TypeError("move_targets must be a tuple")
        if self.value_target is not None:
            if isinstance(self.value_target, bool) or not isinstance(
                self.value_target, (int, float)
            ):
                raise TypeError("value_target must be a real number or None")
            if not math.isfinite(self.value_target) or not -1 <= self.value_target <= 1:
                raise ValueError("value_target must be finite and in [-1, 1]")

        try:
            board = chess.Board(self.fen)
        except ValueError as error:
            raise ValueError(f"invalid FEN: {self.fen!r}") from error
        if not board.is_valid():
            raise ValueError(f"invalid orthodox chess position: {self.fen!r}")
        if board.turn != self.target_color:
            raise ValueError("target_color must be the side to move in fen")

        legal_actions = {encode_move(board, move) for move in board.legal_moves}
        if not legal_actions:
            raise ValueError("ranked positions must be non-terminal")
        if any(not isinstance(target, RankedAction) for target in self.move_targets):
            raise TypeError("move_targets must contain only RankedAction values")
        actions = [target.action for target in self.move_targets]
        if actions != sorted(actions):
            raise ValueError("move_targets must be ordered by ascending action")
        if len(actions) != len(set(actions)):
            raise ValueError("move_targets must not contain duplicate actions")
        if set(actions) != legal_actions:
            raise ValueError(
                "move_targets must contain every legal action exactly once"
            )

        expected_ranks = _dense_ranks(
            [float(target.teacher_score) for target in self.move_targets]
        )
        actual_ranks = [target.rank for target in self.move_targets]
        if actual_ranks != expected_ranks:
            raise ValueError(
                "ranks must be dense descending-score ranks with ties sharing a rank"
            )

    def board(self) -> chess.Board:
        """Reconstruct the represented position."""

        return chess.Board(self.fen)

    @property
    def best_actions(self) -> tuple[int, ...]:
        """Return all rank-one actions in deterministic action order."""

        return tuple(target.action for target in self.move_targets if target.rank == 1)


@dataclass(frozen=True, slots=True)
class RankedDatasetSplit:
    """Leakage-safe partitions containing complete trajectory groups."""

    train: tuple[RankedPosition, ...]
    validation: tuple[RankedPosition, ...]
    test: tuple[RankedPosition, ...]

    @property
    def val(self) -> tuple[RankedPosition, ...]:
        """Short alias for the validation partition."""

        return self.validation


def rank_position(
    board: chess.Board,
    *,
    target_color: chess.Color,
    scorer: LegalMoveScorer,
    context: MoveContext,
    source_id: str,
    trajectory_id: str,
    value_target: float | None = None,
) -> RankedPosition:
    """Score and validate every legal move in a non-terminal target position."""

    if type(target_color) is not bool:
        raise TypeError("target_color must be chess.WHITE or chess.BLACK")
    if not isinstance(context, MoveContext):
        raise TypeError("context must be a MoveContext")
    if context.target_color != target_color:
        raise ValueError("context.target_color must match target_color")
    if board.turn != target_color:
        raise ValueError("rank_position requires target_color to be the side to move")
    if board.is_game_over(claim_draw=False):
        raise ValueError("cannot rank a terminal position")

    scoring_board = board.copy(stack=True)
    raw_scores = scorer(scoring_board, context)
    if not isinstance(raw_scores, Mapping):
        raise TypeError("scorer must return a mapping from legal moves to scores")
    legal_moves = tuple(sorted(board.legal_moves, key=chess.Move.uci))
    legal_set = set(legal_moves)
    try:
        returned_set = set(raw_scores)
    except TypeError as error:
        raise TypeError("scorer keys must be chess.Move values") from error
    if any(not isinstance(move, chess.Move) for move in returned_set):
        raise TypeError("scorer keys must be chess.Move values")
    if returned_set != legal_set:
        missing = sorted(move.uci() for move in legal_set - returned_set)
        extra = sorted(move.uci() for move in returned_set - legal_set)
        raise ValueError(
            "scorer must score every legal move exactly once; "
            f"missing={missing}, extra={extra}"
        )

    action_scores: list[tuple[int, float]] = []
    for move in legal_moves:
        raw_score = raw_scores[move]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError(f"score for {move.uci()} must be a real number")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"score for {move.uci()} must be finite")
        action_scores.append((encode_move(board, move), score))

    action_scores.sort(key=lambda item: item[0])
    ranks = _dense_ranks([score for _, score in action_scores])
    return RankedPosition(
        fen=board.fen(en_passant="fen"),
        target_color=target_color,
        move_targets=tuple(
            RankedAction(action=action, teacher_score=score, rank=rank)
            for (action, score), rank in zip(action_scores, ranks, strict=True)
        ),
        source_id=source_id,
        trajectory_id=trajectory_id,
        value_target=value_target,
    )


def generate_ranked_trajectories(
    scorer: LegalMoveScorer,
    target_policy: MoveSelector,
    opponent_policy: MoveSelector,
    *,
    trajectory_count: int,
    positions_per_trajectory: int,
    max_plies: int,
    seed: int,
    source_id: str,
    opening_plies: int = 0,
    target_colors: Sequence[chess.Color] = (chess.WHITE, chess.BLACK),
    starting_fens: Sequence[str] = (chess.STARTING_FEN,),
) -> tuple[RankedPosition, ...]:
    """Generate on-policy ranked trajectories with optional terminal values.

    Target and opponent selectors play their respective sides.  At most
    ``positions_per_trajectory`` target turns are scored, but play continues
    until termination or ``max_plies`` so recorded positions receive a known
    terminal value whenever the game finishes.  Positions from games truncated
    at ``max_plies`` retain ``None`` as their value target.
    """

    _validate_nonnegative_int("trajectory_count", trajectory_count)
    _validate_positive_int("positions_per_trajectory", positions_per_trajectory)
    _validate_positive_int("max_plies", max_plies)
    _validate_nonnegative_int("opening_plies", opening_plies)
    if opening_plies > max_plies:
        raise ValueError("opening_plies must not exceed max_plies")
    _validate_identifier("source_id", source_id)
    colors = tuple(target_colors)
    if not colors:
        raise ValueError("target_colors must not be empty")
    if any(type(color) is not bool for color in colors):
        raise TypeError("target_colors must contain only chess colors")
    fens = tuple(starting_fens)
    if not fens:
        raise ValueError("starting_fens must not be empty")
    for fen in fens:
        _validated_starting_board(fen)

    all_positions: list[RankedPosition] = []
    for trajectory_index in range(trajectory_count):
        trajectory_id = f"{source_id}/trajectory-{trajectory_index:06d}"
        target_color = colors[trajectory_index % len(colors)]
        board = chess.Board(fens[trajectory_index % len(fens)])
        trajectory_positions: list[RankedPosition] = []
        played_plies = 0

        while (
            played_plies < max_plies
            and not board.is_game_over(claim_draw=False)
        ):
            context = MoveContext(
                game_id=trajectory_id,
                ply=board.ply(),
                seed=_stable_int(seed, trajectory_id, board.ply(), board.fen()),
                target_color=target_color,
            )
            is_target_turn = board.turn == target_color
            if (
                is_target_turn
                and played_plies >= opening_plies
                and len(trajectory_positions) < positions_per_trajectory
            ):
                trajectory_positions.append(
                    rank_position(
                        board,
                        target_color=target_color,
                        scorer=scorer,
                        context=context,
                        source_id=source_id,
                        trajectory_id=trajectory_id,
                    )
                )

            selector = target_policy if is_target_turn else opponent_policy
            move = _select_legal_move(selector, board, context)
            board.push(move)
            played_plies += 1

        value_target: float | None = None
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            value_target = terminal_utility(outcome.winner, target_color)
        all_positions.extend(
            replace(position, value_target=value_target)
            for position in trajectory_positions
        )

    return tuple(all_positions)


def split_ranked_by_trajectory(
    positions: Iterable[RankedPosition],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    group_matching_suffixes: bool = False,
) -> RankedDatasetSplit:
    """Split complete trajectory groups deterministically.

    With ``group_matching_suffixes=True``, records such as
    ``source-a/trajectory-000007`` and ``source-b/trajectory-000007`` stay in
    the same partition. This prevents opening-prefix leakage when several
    opponent corpora were generated from the same indexed opening suite.
    """

    _validate_fractions(train_fraction, validation_fraction)
    groups: dict[tuple[str, str], list[RankedPosition]] = {}
    for position in positions:
        if not isinstance(position, RankedPosition):
            raise TypeError("positions must contain only RankedPosition values")
        if group_matching_suffixes:
            key = ("*", position.trajectory_id.rsplit("/", maxsplit=1)[-1])
        else:
            key = (position.source_id, position.trajectory_id)
        groups.setdefault(key, []).append(position)
    ordered_keys = sorted(groups, key=lambda key: (_stable_int(seed, *key), key))
    train_end = int(len(ordered_keys) * train_fraction)
    validation_end = train_end + int(
        len(ordered_keys) * validation_fraction
    )

    def collect(keys: Sequence[tuple[str, str]]) -> tuple[RankedPosition, ...]:
        return tuple(position for key in keys for position in groups[key])

    return RankedDatasetSplit(
        train=collect(ordered_keys[:train_end]),
        validation=collect(ordered_keys[train_end:validation_end]),
        test=collect(ordered_keys[validation_end:]),
    )


def write_ranked_jsonl(
    path: PathLike, positions: Iterable[RankedPosition]
) -> None:
    """Atomically write deterministic, strict ranked JSONL."""

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
                if not isinstance(position, RankedPosition):
                    raise TypeError(
                        "positions must contain only RankedPosition values"
                    )
                json.dump(
                    _to_record(position),
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
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


def read_ranked_jsonl(path: PathLike) -> tuple[RankedPosition, ...]:
    """Read and validate every ranked JSONL record."""

    positions: list[RankedPosition] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw: object = json.loads(line)
                positions.append(_from_record(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise RankedDatasetFormatError(
                    f"invalid ranked dataset record at line {line_number}: {error}"
                ) from error
    return tuple(positions)


def _dense_ranks(scores: Sequence[float]) -> list[int]:
    rank_for_score = {
        score: rank
        for rank, score in enumerate(sorted(set(scores), reverse=True), start=1)
    }
    return [rank_for_score[score] for score in scores]


def _select_legal_move(
    selector: MoveSelector, board: chess.Board, context: MoveContext
) -> chess.Move:
    selection_board = board.copy(stack=True)
    move = selector(selection_board, context)
    try:
        encode_move(board, move)
    except (ActionEncodingError, AttributeError) as error:
        raise ValueError(
            "move selector must return a legal chess.Move without relying on "
            "mutations to its board argument"
        ) from error
    return move


def _validated_starting_board(fen: object) -> chess.Board:
    if not isinstance(fen, str):
        raise TypeError("starting_fens must contain only strings")
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"invalid starting FEN: {fen!r}") from error
    if not board.is_valid():
        raise ValueError(f"invalid orthodox starting position: {fen!r}")
    return board


def _stable_int(*parts: object) -> int:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _to_record(position: RankedPosition) -> dict[str, object]:
    return {
        "schema": RANKED_DATASET_SCHEMA,
        "schema_version": RANKED_DATASET_SCHEMA_VERSION,
        "fen": position.fen,
        "target_color": "white" if position.target_color else "black",
        "move_targets": [
            {
                "action": target.action,
                "teacher_score": float(target.teacher_score),
                "rank": target.rank,
            }
            for target in position.move_targets
        ],
        "source_id": position.source_id,
        "trajectory_id": position.trajectory_id,
        "value_target": (
            None if position.value_target is None else float(position.value_target)
        ),
    }


def _from_record(raw: object) -> RankedPosition:
    if not isinstance(raw, dict):
        raise RankedDatasetFormatError("record must be a JSON object")
    record = cast(dict[object, object], raw)
    expected_fields = {
        "schema",
        "schema_version",
        "fen",
        "target_color",
        "move_targets",
        "source_id",
        "trajectory_id",
        "value_target",
    }
    if set(record) != expected_fields:
        raise RankedDatasetFormatError(
            f"record fields must be exactly {sorted(expected_fields)!r}"
        )
    if record["schema"] != RANKED_DATASET_SCHEMA:
        raise RankedDatasetFormatError(f"unsupported schema {record['schema']!r}")
    if type(record["schema_version"]) is not int:
        raise RankedDatasetFormatError("schema_version must be an integer")
    if record["schema_version"] != RANKED_DATASET_SCHEMA_VERSION:
        raise RankedDatasetFormatError(
            f"unsupported schema_version {record['schema_version']!r}"
        )
    if record["target_color"] not in ("white", "black"):
        raise RankedDatasetFormatError("target_color must be 'white' or 'black'")
    for field in ("fen", "source_id", "trajectory_id"):
        if not isinstance(record[field], str):
            raise RankedDatasetFormatError(f"{field} must be a string")
    raw_targets = record["move_targets"]
    if not isinstance(raw_targets, list):
        raise RankedDatasetFormatError("move_targets must be a list")
    move_targets = tuple(_ranked_action_from_record(item) for item in raw_targets)
    value_target = record["value_target"]
    if value_target is not None and (
        isinstance(value_target, bool) or not isinstance(value_target, (int, float))
    ):
        raise RankedDatasetFormatError("value_target must be a number or null")
    return RankedPosition(
        fen=cast(str, record["fen"]),
        target_color=record["target_color"] == "white",
        move_targets=move_targets,
        source_id=cast(str, record["source_id"]),
        trajectory_id=cast(str, record["trajectory_id"]),
        value_target=(None if value_target is None else float(value_target)),
    )


def _ranked_action_from_record(raw: object) -> RankedAction:
    if not isinstance(raw, dict):
        raise RankedDatasetFormatError("each move target must be an object")
    record = cast(dict[object, object], raw)
    expected_fields = {"action", "teacher_score", "rank"}
    if set(record) != expected_fields:
        raise RankedDatasetFormatError(
            f"move target fields must be exactly {sorted(expected_fields)!r}"
        )
    if type(record["action"]) is not int:
        raise RankedDatasetFormatError("action must be an integer")
    if isinstance(record["teacher_score"], bool) or not isinstance(
        record["teacher_score"], (int, float)
    ):
        raise RankedDatasetFormatError("teacher_score must be a number")
    if type(record["rank"]) is not int:
        raise RankedDatasetFormatError("rank must be an integer")
    return RankedAction(
        action=record["action"],
        teacher_score=float(record["teacher_score"]),
        rank=record["rank"],
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
    "RANKED_DATASET_SCHEMA",
    "RANKED_DATASET_SCHEMA_VERSION",
    "LegalMoveScorer",
    "MoveSelector",
    "RankedAction",
    "RankedDatasetFormatError",
    "RankedDatasetSplit",
    "RankedPosition",
    "generate_ranked_trajectories",
    "rank_position",
    "read_ranked_jsonl",
    "split_ranked_by_trajectory",
    "write_ranked_jsonl",
]
