from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.import_yacpdb_selfmates import (  # noqa: E402
    diagram_to_board,
    normalize_entry,
)


def sample_entry() -> dict[str, object]:
    return {
        "id": 4116,
        "authors": ["Ceriani, Luigi"],
        "source": {"name": "L'Italia Scacchistica", "date": {"year": 1928}},
        "algebraic": {
            "white": ["Kd3", "Qd6", "Sg6", "Pa7"],
            "black": ["Kf5", "Qa4", "Sb1", "Pa3"],
        },
        "stipulation": "s#2",
        "comments": ["C+"],
    }


def test_normalizes_standard_selfmate_without_copying_solution() -> None:
    entry = sample_entry()
    entry["solution"] = "published solution text"

    record = normalize_entry(entry)

    assert record is not None
    assert record["source_id"] == 4116
    assert record["selfmate_moves"] == 2
    assert record["authors"] == ["Ceriani, Luigi"]
    assert "solution" not in record
    board = chess.Board(record["fen"])
    assert board.piece_at(chess.G6) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert board.turn == chess.WHITE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("twins", {"a": ""}),
        ("options", ["SetPlay"]),
        ("comments", ["No solution"]),
        ("stipulation", "h#2"),
    ],
)
def test_rejects_unsupported_or_unsound_entries(field: str, value: object) -> None:
    entry = sample_entry()
    entry[field] = value

    assert normalize_entry(entry) is None


def test_rejects_fairy_piece_tokens_and_invalid_diagrams() -> None:
    fairy = sample_entry()
    fairy["algebraic"] = {"white": ["Kd3", "Gd6"], "black": ["Kf5"]}
    missing_king = sample_entry()
    missing_king["algebraic"] = {"white": ["Qd6"], "black": ["Kf5"]}

    assert normalize_entry(fairy) is None
    assert normalize_entry(missing_king) is None


def test_diagram_parser_rejects_duplicate_squares() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        diagram_to_board({"white": ["Ke1"], "black": ["Ke1"]})
