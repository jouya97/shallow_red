from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.chess.actions import decode_action, encode_move
from worst_chess.training.dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetFormatError,
    LabeledPosition,
    generate_labeled_positions,
    read_jsonl,
    split_by_trajectory,
    write_jsonl,
)


def _first_legal_move(board: chess.Board, context: MoveContext) -> chess.Move:
    del context
    return min(board.legal_moves, key=chess.Move.uci)


def _example(
    trajectory_id: str = "source/trajectory-000000",
    *,
    source_id: str = "source",
) -> LabeledPosition:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    return LabeledPosition(
        fen=board.fen(),
        target_color=chess.WHITE,
        chosen_action=encode_move(board, move),
        source_id=source_id,
        trajectory_id=trajectory_id,
    )


def test_labeled_position_is_immutable_and_decodes_legal_move() -> None:
    position = _example()

    assert position.chosen_move() == chess.Move.from_uci("e2e4")
    assert position.chosen_move() in position.board().legal_moves
    with pytest.raises(FrozenInstanceError):
        position.fen = "invalid"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"fen": "not a fen"},
        {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"},
        {"target_color": chess.BLACK},
        {"chosen_action": -1},
        {"chosen_action": 0},
        {"source_id": ""},
        {"trajectory_id": "   "},
    ],
)
def test_labeled_position_rejects_invalid_data(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "fen": chess.Board().fen(),
        "target_color": chess.WHITE,
        "chosen_action": encode_move(
            chess.Board(), chess.Move.from_uci("e2e4")
        ),
        "source_id": "source",
        "trajectory_id": "trajectory",
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        LabeledPosition(**values)  # type: ignore[arg-type]


def test_generation_is_deterministic_diverse_and_legal() -> None:
    arguments = {
        "trajectory_count": 6,
        "positions_per_trajectory": 8,
        "seed": 541,
        "source_id": "heuristic-v1",
        "opening_plies": 3,
    }

    first = generate_labeled_positions(_first_legal_move, **arguments)
    second = generate_labeled_positions(_first_legal_move, **arguments)
    different_seed = generate_labeled_positions(
        _first_legal_move,
        **{**arguments, "seed": 542},
    )

    assert first == second
    assert len(first) == 48
    assert len({position.fen for position in first}) > 40
    assert [position.fen for position in first] != [
        position.fen for position in different_seed
    ]
    assert {position.target_color for position in first} == {
        chess.WHITE,
        chess.BLACK,
    }
    for position in first:
        board = position.board()
        assert board.turn == position.target_color
        assert not board.is_game_over(claim_draw=False)
        assert decode_action(board, position.chosen_action) == _first_legal_move(
            board,
            MoveContext(
                game_id=position.trajectory_id,
                ply=board.ply(),
                seed=0,
                target_color=position.target_color,
            ),
        )


def test_generation_protects_trajectory_from_labeler_board_mutation() -> None:
    seen_fens: list[str] = []

    def mutating_labeler(board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        seen_fens.append(board.fen(en_passant="fen"))
        move = min(board.legal_moves, key=chess.Move.uci)
        board.push(move)
        return move

    positions = generate_labeled_positions(
        mutating_labeler,
        trajectory_count=1,
        positions_per_trajectory=3,
        seed=4,
        source_id="mutating",
    )

    assert len(positions) == 3
    assert [position.fen for position in positions] == seen_fens


def test_generation_rejects_an_illegal_label() -> None:
    def illegal_labeler(board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci("a1a8")

    with pytest.raises(ValueError, match="labeler must return a legal"):
        generate_labeled_positions(
            illegal_labeler,
            trajectory_count=1,
            positions_per_trajectory=1,
            seed=0,
            source_id="bad",
        )


def test_split_keeps_whole_source_trajectory_groups_together() -> None:
    positions = tuple(
        _example(f"trajectory-{trajectory}", source_id=source)
        for source in ("a", "b")
        for trajectory in range(5)
        for _ in range(3)
    )

    split = split_by_trajectory(
        positions,
        seed=91,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    repeated = split_by_trajectory(
        positions,
        seed=91,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert split == repeated
    assert len(split.train) == 18
    assert len(split.validation) == 6
    assert len(split.test) == 6
    assert split.val is split.validation

    partition_keys = [
        {(position.source_id, position.trajectory_id) for position in partition}
        for partition in (split.train, split.validation, split.test)
    ]
    assert partition_keys[0].isdisjoint(partition_keys[1])
    assert partition_keys[0].isdisjoint(partition_keys[2])
    assert partition_keys[1].isdisjoint(partition_keys[2])


def test_jsonl_round_trip_has_explicit_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dataset.jsonl"
    positions = (_example(), _example("source/trajectory-000001"))

    write_jsonl(path, positions)

    assert read_jsonl(path) == positions
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(record["schema_version"] == DATASET_SCHEMA_VERSION for record in records)
    assert all(record["target_color"] == "white" for record in records)


def test_jsonl_rejects_unknown_schema_and_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"schema_version":999}\n', encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="line 1"):
        read_jsonl(path)


def test_jsonl_write_is_atomic_on_iterator_failure(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("original\n", encoding="utf-8")

    def failing_positions() -> Iterator[LabeledPosition]:
        yield _example()
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        write_jsonl(path, failing_positions())

    assert path.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".dataset.jsonl.*.tmp")) == []
