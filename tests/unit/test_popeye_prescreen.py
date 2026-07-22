from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.popeye_prescreen import classify_output, popeye_input

SELF_MATE_IN_ONE = "rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R w - - 7 6"


def test_renders_white_target_as_popeye_selfmate() -> None:
    rendered = popeye_input(chess.Board(SELF_MATE_IN_ONE), chess.WHITE, 4)

    assert "forsyth rnbq1b1r/pppp1ppp/6kn/3Np3/5PP1/5N2/PPPPP2P/R1BQKB1R" in rendered
    assert "stipulation s#4" in rendered
    assert "NoCastling a1h1a8h8" in rendered


def test_mirrors_black_target_to_popeyes_white_selfmate_role() -> None:
    original = chess.Board(SELF_MATE_IN_ONE)
    black_target = original.mirror()

    rendered = popeye_input(black_target, chess.BLACK, 1)

    assert f"forsyth {original.board_fen()}" in rendered


def test_rejects_position_state_popeye_input_cannot_preserve() -> None:
    with pytest.raises(ValueError, match="castling"):
        popeye_input(chess.Board(), chess.WHITE, 1)


def test_classifies_solution_negative_and_interrupted_output() -> None:
    assert classify_output("\n  1.Nf3-h4 + !\n") == "found"
    assert classify_output("solution finished. Time = 0.1 s") == "not_found"
    assert classify_output("Solving interrupted") == "unknown"
