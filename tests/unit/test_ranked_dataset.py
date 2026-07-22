from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.chess.actions import decode_action
from worst_chess.training.ranked_dataset import (
    RANKED_DATASET_SCHEMA,
    RANKED_DATASET_SCHEMA_VERSION,
    RankedAction,
    RankedDatasetFormatError,
    RankedPosition,
    generate_ranked_trajectories,
    rank_position,
    read_ranked_jsonl,
    split_ranked_by_trajectory,
    write_ranked_jsonl,
)


def _context(target_color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext(game_id="trajectory", ply=0, seed=7, target_color=target_color)


def _scorer(
    board: chess.Board, context: MoveContext
) -> dict[chess.Move, float]:
    del context
    moves = sorted(board.legal_moves, key=chess.Move.uci)
    return {
        move: 10.0 if index < 2 else -float(index // 2)
        for index, move in enumerate(moves)
    }


def _first_move(board: chess.Board, context: MoveContext) -> chess.Move:
    del context
    return min(board.legal_moves, key=chess.Move.uci)


def _example(trajectory_id: str = "source/trajectory-000000") -> RankedPosition:
    return rank_position(
        chess.Board(),
        target_color=chess.WHITE,
        scorer=_scorer,
        context=_context(),
        source_id="source",
        trajectory_id=trajectory_id,
    )


def test_rank_position_covers_every_legal_move_with_dense_tie_ranks() -> None:
    board = chess.Board()
    position = _example()

    assert len(position.move_targets) == board.legal_moves.count()
    assert [target.action for target in position.move_targets] == sorted(
        target.action for target in position.move_targets
    )
    assert {
        decode_action(board, target.action) for target in position.move_targets
    } == set(board.legal_moves)
    assert len(position.best_actions) == 2
    assert {
        target.rank for target in position.move_targets if target.teacher_score == 10
    } == {1}
    assert position.value_target is None
    with pytest.raises(FrozenInstanceError):
        position.fen = "invalid"  # type: ignore[misc]


def test_rank_position_requires_exactly_all_legal_scores() -> None:
    def incomplete(
        board: chess.Board, context: MoveContext
    ) -> dict[chess.Move, float]:
        del context
        return {next(iter(board.legal_moves)): 0.0}

    with pytest.raises(ValueError, match="every legal move exactly once"):
        rank_position(
            chess.Board(),
            target_color=chess.WHITE,
            scorer=incomplete,
            context=_context(),
            source_id="source",
            trajectory_id="trajectory",
        )


def test_rank_position_rejects_mismatched_context_orientation() -> None:
    with pytest.raises(ValueError, match="context.target_color"):
        rank_position(
            chess.Board(),
            target_color=chess.WHITE,
            scorer=_scorer,
            context=_context(chess.BLACK),
            source_id="source",
            trajectory_id="trajectory",
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "1"])
def test_rank_position_rejects_nonfinite_or_non_numeric_scores(
    score: object,
) -> None:
    def invalid(board: chess.Board, context: MoveContext) -> dict[chess.Move, object]:
        del context
        return {move: score for move in board.legal_moves}

    with pytest.raises((TypeError, ValueError)):
        rank_position(
            chess.Board(),
            target_color=chess.WHITE,
            scorer=invalid,  # type: ignore[arg-type]
            context=_context(),
            source_id="source",
            trajectory_id="trajectory",
        )


def test_ranked_position_rejects_missing_actions_bad_order_and_bad_ranks() -> None:
    position = _example()
    with pytest.raises(ValueError, match="every legal action"):
        replace(position, move_targets=position.move_targets[:-1])
    with pytest.raises(ValueError, match="ascending action"):
        replace(position, move_targets=tuple(reversed(position.move_targets)))
    changed = list(position.move_targets)
    first = changed[0]
    changed[0] = RankedAction(first.action, first.teacher_score, 999)
    with pytest.raises(ValueError, match="dense descending-score"):
        replace(position, move_targets=tuple(changed))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        replace(position, value_target=1.1)


def test_on_policy_generation_is_deterministic_and_assigns_terminal_value() -> None:
    def target_policy(board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        uci = "f2f3" if board.ply() == 0 else "g2g4"
        return chess.Move.from_uci(uci)

    def opponent_policy(board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        uci = "e7e5" if board.ply() == 1 else "d8h4"
        return chess.Move.from_uci(uci)

    arguments = dict(
        trajectory_count=1,
        positions_per_trajectory=10,
        max_plies=4,
        seed=83,
        source_id="fools-mate",
        target_colors=(chess.WHITE,),
    )
    first = generate_ranked_trajectories(
        _scorer, target_policy, opponent_policy, **arguments
    )
    second = generate_ranked_trajectories(
        _scorer, target_policy, opponent_policy, **arguments
    )

    assert first == second
    assert len(first) == 2
    assert all(position.target_color == chess.WHITE for position in first)
    assert all(position.value_target == 1.0 for position in first)
    assert [position.trajectory_id for position in first] == [
        "fools-mate/trajectory-000000",
        "fools-mate/trajectory-000000",
    ]


def test_truncated_on_policy_trajectory_has_no_value_target() -> None:
    positions = generate_ranked_trajectories(
        _scorer,
        _first_move,
        _first_move,
        trajectory_count=1,
        positions_per_trajectory=2,
        max_plies=1,
        seed=0,
        source_id="truncated",
        target_colors=(chess.WHITE,),
    )

    assert len(positions) == 1
    assert positions[0].value_target is None


def test_generation_rejects_illegal_selector_move() -> None:
    def illegal(board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci("a1a8")

    with pytest.raises(ValueError, match="move selector must return a legal"):
        generate_ranked_trajectories(
            _scorer,
            illegal,
            _first_move,
            trajectory_count=1,
            positions_per_trajectory=1,
            max_plies=2,
            seed=0,
            source_id="illegal",
        )


def test_split_keeps_source_trajectory_groups_whole_and_is_deterministic() -> None:
    positions = tuple(
        replace(
            _example(f"trajectory-{trajectory}"),
            source_id=source,
            trajectory_id=f"trajectory-{trajectory}",
        )
        for source in ("a", "b")
        for trajectory in range(5)
        for _ in range(2)
    )

    split = split_ranked_by_trajectory(
        positions, seed=9, train_fraction=0.6, validation_fraction=0.2
    )
    repeated = split_ranked_by_trajectory(
        positions, seed=9, train_fraction=0.6, validation_fraction=0.2
    )

    assert split == repeated
    assert [len(split.train), len(split.validation), len(split.test)] == [12, 4, 4]
    assert split.val is split.validation
    keys = [
        {(position.source_id, position.trajectory_id) for position in partition}
        for partition in (split.train, split.validation, split.test)
    ]
    assert keys[0].isdisjoint(keys[1])
    assert keys[0].isdisjoint(keys[2])
    assert keys[1].isdisjoint(keys[2])


def test_split_can_group_matching_trajectory_suffixes_across_sources() -> None:
    positions = tuple(
        replace(
            _example(f"{source}/trajectory-{trajectory:06d}"),
            source_id=source,
            trajectory_id=f"{source}/trajectory-{trajectory:06d}",
        )
        for source in ("ordinary", "resistant")
        for trajectory in range(10)
    )

    split = split_ranked_by_trajectory(
        positions,
        seed=11,
        train_fraction=0.6,
        validation_fraction=0.2,
        group_matching_suffixes=True,
    )

    partitions = (split.train, split.validation, split.test)
    suffix_partition: dict[str, int] = {}
    for partition_index, partition in enumerate(partitions):
        for position in partition:
            suffix = position.trajectory_id.rsplit("/", maxsplit=1)[-1]
            assigned = suffix_partition.setdefault(suffix, partition_index)
            assert assigned == partition_index
    assert [len(partition) for partition in partitions] == [12, 4, 4]


def test_ranked_jsonl_is_deterministic_versioned_and_round_trips(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "one" / "ranked.jsonl"
    second_path = tmp_path / "two" / "ranked.jsonl"
    positions = (_example(), replace(_example("trajectory-2"), value_target=0.0))

    write_ranked_jsonl(first_path, positions)
    write_ranked_jsonl(second_path, positions)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert read_ranked_jsonl(first_path) == positions
    records = [json.loads(line) for line in first_path.read_text().splitlines()]
    assert records[0]["schema"] == RANKED_DATASET_SCHEMA
    assert records[0]["schema_version"] == RANKED_DATASET_SCHEMA_VERSION


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 999},
        {"schema": "some-other-dataset"},
        {"extra": True},
    ],
)
def test_ranked_jsonl_rejects_unknown_or_extra_schema_fields(
    tmp_path: Path, change: dict[str, object]
) -> None:
    path = tmp_path / "ranked.jsonl"
    good_path = tmp_path / "good.jsonl"
    write_ranked_jsonl(good_path, (_example(),))
    record = json.loads(good_path.read_text())
    record.update(change)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(RankedDatasetFormatError, match="line 1"):
        read_ranked_jsonl(path)


def test_ranked_jsonl_write_is_atomic_on_iterator_failure(tmp_path: Path) -> None:
    path = tmp_path / "ranked.jsonl"
    path.write_text("original\n", encoding="utf-8")

    def failing_positions() -> Iterator[RankedPosition]:
        yield _example()
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        write_ranked_jsonl(path, failing_positions())

    assert path.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".ranked.jsonl.*.tmp")) == []
