from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.screen_synthetic_ancestry import (  # noqa: E402
    extract_decisive_positions,
    promising_positions,
    shortlist_candidates,
)
from worst_chess.agents.base import MoveContext  # noqa: E402
from worst_chess.agents.neural import PolicyMove  # noqa: E402


class FakeNeural:
    def rank_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        top_k: int | None = None,
    ) -> tuple[PolicyMove, ...]:
        del board, context
        ranked = (
            PolicyMove(chess.Move.from_uci("d2d4"), 1, 2.0),
            PolicyMove(chess.Move.from_uci("e2e4"), 2, 1.0),
        )
        return ranked if top_k is None else ranked[:top_k]


def _fools_mate_game(*, target: str, round_name: str) -> chess.pgn.Game:
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    game = chess.pgn.Game.from_board(board)
    game.headers["Target"] = target
    game.headers["Round"] = round_name
    return game


def test_extract_decisive_positions_distinguishes_target_loss_and_win(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decisive.pgn"
    games = (
        _fools_mate_game(target="white", round_name="white-loses"),
        _fools_mate_game(target="black", round_name="black-wins"),
    )
    path.write_text("\n\n".join(str(game) for game in games) + "\n")

    positions = extract_decisive_positions(
        [path],
        outcomes={"loss", "win"},
        tail_target_positions=2,
    )

    assert len(positions) == 4
    losses = [position for position in positions if position.source_outcome == "loss"]
    wins = [position for position in positions if position.source_outcome == "win"]
    assert {position.actual_move for position in losses} == {"f2f3", "g2g4"}
    assert {position.actual_move for position in wins} == {"e7e5", "d8h4"}
    assert sorted(position.target_turns_before_end for position in losses) == [1, 2]
    assert min(position.plies_before_end for position in losses) == 2


def test_shortlist_is_a_deduplicated_union_and_keeps_actual_move() -> None:
    board = chess.Board()
    context = MoveContext("test", 0, 1, chess.WHITE)

    candidates, sources, model_move = shortlist_candidates(
        board,
        context,
        FakeNeural(),  # type: ignore[arg-type]
        actual_move=chess.Move.from_uci("e2e4"),
        model_top_k=2,
        heuristic_top_k=2,
        random_reply_top_k=2,
    )

    assert model_move == "d2d4"
    assert chess.Move.from_uci("e2e4") in candidates
    assert sources[chess.Move.from_uci("e2e4")] == ["model", "actual"]
    assert len(candidates) == len(set(candidates))
    assert board == chess.Board()


def test_promising_positions_discards_zero_selfmate_records() -> None:
    base = {
        "fen": chess.Board().fen(en_passant="fen"),
        "target_color": "white",
        "source": "games.pgn",
        "game_id": "1",
        "source_outcome": "loss",
        "actual_move": "e2e4",
        "target_turns_before_end": 1,
        "plies_before_end": 2,
    }
    report = {
        "records": [
            {**base, "candidates": [{"move": "e2e4", "selfmates": 0}]},
            {
                **base,
                "game_id": "2",
                "candidates": [{"move": "d2d4", "selfmates": 1}],
            },
        ]
    }

    positions = promising_positions(report)

    assert len(positions) == 1
    assert positions[0].game_id == "2"
