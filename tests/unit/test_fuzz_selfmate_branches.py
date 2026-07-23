from __future__ import annotations

import json
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.fuzz_selfmate_branches import (  # noqa: E402
    BranchResult,
    ForcedFirstMoveAgent,
    FuzzNode,
    annotate_sibling_outcomes,
    branch_candidates,
    load_frontier_nodes,
    load_seed_nodes,
    safe_root_moves,
    select_novel_beam,
)
from worst_chess.agents.base import MoveContext  # noqa: E402
from worst_chess.agents.neural import PolicyMove  # noqa: E402
from worst_chess.training.ranked_dataset import (  # noqa: E402
    rank_position,
    write_ranked_jsonl,
)


class FixedAgent:
    name = "fixed"

    def __init__(self, move: str) -> None:
        self.move = chess.Move.from_uci(move)
        self.calls = 0

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        self.calls += 1
        return self.move


class MateFirstNeural:
    def rank_moves(
        self,
        board: chess.Board,
        context: MoveContext,
        *,
        top_k: int | None = None,
    ) -> tuple[PolicyMove, ...]:
        del board, context, top_k
        return (
            PolicyMove(chess.Move.from_uci("d8h4"), 1, 10.0),
            PolicyMove(chess.Move.from_uci("a7a6"), 2, 1.0),
        )


def _ranked_start(tmp_path: Path) -> Path:
    board = chess.Board()

    def scorer(
        scoring_board: chess.Board,
        context: MoveContext,
    ) -> dict[chess.Move, float]:
        del context
        return {
            move: float(move.uci() in {"e2e4", "d2d4"})
            for move in scoring_board.legal_moves
        }

    position = rank_position(
        board,
        target_color=chess.WHITE,
        scorer=scorer,
        context=MoveContext("seed", 0, 1, chess.WHITE),
        source_id="test",
        trajectory_id="test-seed",
    )
    path = tmp_path / "seed.jsonl"
    write_ranked_jsonl(path, (position,))
    return path


def test_load_seed_nodes_adds_legal_color_mirror(tmp_path: Path) -> None:
    nodes = load_seed_nodes(
        _ranked_start(tmp_path),
        branch_moves=2,
        min_legal_moves=2,
        mirror=True,
    )

    assert len(nodes) == 2
    assert {node.target_color for node in nodes} == {chess.WHITE, chess.BLACK}
    for node in nodes:
        board = chess.Board(node.fen)
        assert board.turn == node.target_color
        assert all(
            chess.Move.from_uci(move) in board.legal_moves
            for move in node.preferred_moves
        )


def test_forced_agent_uses_root_move_exactly_once() -> None:
    board = chess.Board()
    fallback = FixedAgent("g1f3")
    agent = ForcedFirstMoveAgent(chess.Move.from_uci("e2e4"), fallback)
    context = MoveContext("branch", 0, 1, chess.WHITE)

    assert agent.select_move(board, context).uci() == "e2e4"
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert agent.select_move(board, context).uci() == "g1f3"
    assert fallback.calls == 1


def test_safe_root_moves_excludes_avoidable_immediate_target_win() -> None:
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)

    moves = safe_root_moves(board)

    assert chess.Move.from_uci("d8h4") not in moves
    assert moves

    candidates = branch_candidates(
        board,
        MoveContext("mate-filter", board.ply(), 1, chess.BLACK),
        MateFirstNeural(),  # type: ignore[arg-type]
        count=3,
    )
    assert chess.Move.from_uci("d8h4") not in candidates


def test_load_frontier_nodes_preserves_lineage_and_deduplicates(
    tmp_path: Path,
) -> None:
    record = {
        "fen": chess.Board().fen(en_passant="fen"),
        "target_color": "white",
        "root_id": "root-a",
        "lineage": "root-a/g00/frontier",
        "generation": 1,
        "pressure": 12.5,
        "immediate_mate_probability": 0.25,
    }
    path = tmp_path / "frontier.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    nodes = load_frontier_nodes(path)

    assert len(nodes) == 1
    assert nodes[0].lineage == "root-a/g00/frontier"
    assert nodes[0].generation == 1


def test_novel_beam_deduplicates_and_round_robins_roots() -> None:
    boards: list[chess.Board] = []
    for first_move in ("e2e4", "d2d4", "c2c4", "g1f3"):
        board = chess.Board()
        board.push_uci(first_move)
        board.push_uci("e7e5" if first_move != "e2e4" else "c7c5")
        boards.append(board)
    children = [
        FuzzNode(
            fen=boards[index].fen(en_passant="fen"),
            target_color=chess.WHITE,
            root_id="a" if index < 3 else "b",
            lineage=f"child-{index}",
            generation=1,
            pressure=float(10 - index),
            immediate_mate_probability=0.0,
        )
        for index in range(4)
    ]
    children.append(
        FuzzNode(
            fen=children[0].fen,
            target_color=chess.WHITE,
            root_id="duplicate",
            lineage="weaker-duplicate",
            generation=1,
            pressure=-1.0,
            immediate_mate_probability=0.0,
        )
    )

    selected = select_novel_beam(children, beam_width=3)

    assert len(selected) == 3
    assert [node.root_id for node in selected[:2]] == ["a", "b"]
    assert all(node.lineage != "weaker-duplicate" for node in selected)


def test_safety_beam_rejects_action_with_sampled_sibling_win() -> None:
    board = chess.Board()
    risky_child = FuzzNode(
        fen=board.fen(en_passant="fen"),
        target_color=chess.WHITE,
        root_id="root",
        lineage="risky-frontier",
        generation=1,
        pressure=100.0,
        immediate_mate_probability=1.0,
    )
    safe_board = chess.Board()
    safe_board.push_uci("e2e4")
    safe_board.push_uci("e7e5")
    safe_child = FuzzNode(
        fen=safe_board.fen(en_passant="fen"),
        target_color=chess.WHITE,
        root_id="root",
        lineage="safe-frontier",
        generation=1,
        pressure=0.0,
        immediate_mate_probability=0.0,
    )

    def result(
        move: str,
        outcome: str,
        child: FuzzNode | None,
    ) -> BranchResult:
        return BranchResult(
            record={
                "root_id": "root",
                "parent_fen": chess.STARTING_FEN,
                "forced_move": move,
                "outcome": outcome,
            },
            child=child,
            pgn=None,
        )

    children = annotate_sibling_outcomes(
        [
            result("e2e4", "win", None),
            result("e2e4", "frontier", risky_child),
            result("d2d4", "loss", None),
            result("d2d4", "frontier", safe_child),
        ]
    )

    assert select_novel_beam(children, beam_width=1)[0].lineage == "safe-frontier"
    assert (
        select_novel_beam(children, beam_width=1, safety_first=False)[0].lineage
        == "risky-frontier"
    )
