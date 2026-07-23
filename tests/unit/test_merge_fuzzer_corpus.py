from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.merge_fuzzer_corpus import merge_fuzzer_corpus  # noqa: E402


def _write_shard(
    directory: Path,
    *,
    round_id: str,
    root_id: str,
    outcome: str,
) -> None:
    directory.mkdir()
    board = chess.Board()
    game = chess.pgn.Game.from_board(board)
    game.headers["Round"] = round_id
    game.headers["Target"] = "white"
    (directory / "decisive-games.pgn").write_text(
        str(game) + "\n",
        encoding="utf-8",
    )
    record = {
        "game_id": round_id,
        "root_id": root_id,
        "outcome": outcome,
        "final_fen": board.fen(),
    }
    (directory / "report.json").write_text(
        json.dumps({"summary": {}, "records": [record]}),
        encoding="utf-8",
    )


def test_merge_counts_independent_roots_and_deduplicates_games(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(
        first,
        round_id="fresh-00001-black/g00/a1a2-s00",
        root_id="fresh-00001-black",
        outcome="loss",
    )
    _write_shard(
        second,
        round_id="fresh-00001-black/g00/a1a2-s00",
        root_id="fresh-00001-black",
        outcome="loss",
    )

    result = merge_fuzzer_corpus([first, second])

    assert result["summary"]["losses"] == 2
    assert result["summary"]["independent_loss_families"] == 1
    assert result["summary"]["decisive_games"] == 1
    assert result["loss_root_ids"] == ["fresh-00001-black"]
