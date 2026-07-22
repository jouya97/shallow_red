from __future__ import annotations

import math
from collections import defaultdict

import chess
import pytest

from worst_chess.agents.base import AgentError, MoveContext
from worst_chess.chess.actions import decode_action
from worst_chess.training.ranked_dataset import rank_position
from worst_chess.training.rollout_teacher import (
    LexicographicRolloutScorer,
    RolloutConfig,
    rollout_ranking_score,
)


def _first_legal(board: chess.Board, context: MoveContext) -> chess.Move:
    del context
    return min(board.legal_moves, key=chess.Move.uci)


def _context(color: chess.Color = chess.WHITE) -> MoveContext:
    return MoveContext("rollout-test", 0, 19, color)


def _score(
    config: RolloutConfig,
    *,
    selfmates: int,
    plies: int,
    wins: int = 0,
    truncations: int = 0,
) -> float:
    return rollout_ranking_score(
        selfmates=selfmates,
        selfmate_plies_sum=plies,
        target_wins=wins,
        truncations=truncations,
        config=config,
    )


def test_mixed_radix_preserves_strict_lexicographic_priority() -> None:
    config = RolloutConfig(rollouts=4, max_plies=100)

    assert _score(config, selfmates=2, plies=200, wins=2) > _score(
        config, selfmates=1, plies=1
    )
    assert _score(config, selfmates=2, plies=20) > _score(
        config, selfmates=2, plies=21
    )
    assert _score(config, selfmates=2, plies=20) > _score(
        config, selfmates=2, plies=20, wins=1
    )
    assert _score(config, selfmates=0, plies=0) > _score(
        config, selfmates=0, plies=0, truncations=1
    )
    assert _score(config, selfmates=2, plies=20) == _score(
        config, selfmates=2, plies=20
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"rollouts": 0},
        {"max_plies": 0},
        {"seed": 1.5},
        {"rollouts": 10**8, "max_plies": 10**8},
    ],
)
def test_rollout_config_rejects_invalid_or_inexact_ranges(
    arguments: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RolloutConfig(**arguments)  # type: ignore[arg-type]


def test_rollout_ranking_score_validates_counts() -> None:
    config = RolloutConfig(rollouts=2, max_plies=4)

    with pytest.raises(ValueError, match="exceed"):
        _score(config, selfmates=2, plies=2, wins=1)
    with pytest.raises(ValueError, match="at least one ply"):
        _score(config, selfmates=1, plies=0)
    with pytest.raises(ValueError, match="horizon"):
        _score(config, selfmates=1, plies=5)


def test_scorer_finds_fools_mate_reply_and_scores_every_move() -> None:
    board = chess.Board()
    board.push_uci("f2f3")
    board.push_uci("e7e5")

    def mating_opponent(board: chess.Board, context: MoveContext) -> chess.Move:
        assert board.turn != context.target_color
        mating = [move for move in board.legal_moves if board.gives_check(move)]
        for move in mating:
            after = board.copy(stack=False)
            after.push(move)
            if after.is_checkmate():
                return move
        return min(board.legal_moves, key=chess.Move.uci)

    scorer = LexicographicRolloutScorer(
        _first_legal,
        mating_opponent,
        RolloutConfig(rollouts=3, max_plies=2, seed=5),
    )
    original_fen = board.fen(en_passant="fen")
    original_stack = tuple(board.move_stack)

    summaries = scorer.evaluate_moves(board, _context())
    scores = scorer.score_moves(board, _context())
    ranked = rank_position(
        board,
        target_color=chess.WHITE,
        scorer=scorer,
        context=_context(),
        source_id="rollout-test",
        trajectory_id="fools-mate",
    )
    g4 = next(item for item in summaries if item.move.uci() == "g2g4")

    assert g4.selfmates == 3
    assert g4.selfmate_plies_sum == 6
    assert g4.ranking_score == max(item.ranking_score for item in summaries)
    assert list(scores) == sorted(board.legal_moves, key=chess.Move.uci)
    assert len(scores) == board.legal_moves.count()
    assert all(math.isfinite(score) for score in scores.values())
    assert chess.Move.from_uci("g2g4") in {
        decode_action(board, action) for action in ranked.best_actions
    }
    assert board.fen(en_passant="fen") == original_fen
    assert tuple(board.move_stack) == original_stack


def test_common_random_contexts_match_across_candidate_actions_and_roles() -> None:
    records: dict[str, dict[str, list[tuple[str, int, int]]]] = {
        "target": defaultdict(list),
        "opponent": defaultdict(list),
    }

    class RecordingAgent:
        def __init__(self, role: str) -> None:
            self.role = role

        def select_move(
            self, board: chess.Board, context: MoveContext
        ) -> chess.Move:
            is_target = board.turn == context.target_color
            assert is_target == (self.role == "target")
            candidate = board.move_stack[0].uci()
            records[self.role][candidate].append(
                (context.game_id, context.seed, context.ply)
            )
            return min(board.legal_moves, key=chess.Move.uci)

    board = chess.Board()
    scorer = LexicographicRolloutScorer(
        RecordingAgent("target"),
        RecordingAgent("opponent"),
        RolloutConfig(rollouts=3, max_plies=3, seed=7),
    )

    scorer.score_moves(board, _context())

    for role_records in records.values():
        sequences = list(role_records.values())
        assert len(sequences) == board.legal_moves.count()
        assert all(sequence == sequences[0] for sequence in sequences[1:])
        assert len(sequences[0]) == 3


def test_selector_mutation_is_isolated_and_illegal_moves_are_rejected() -> None:
    board = chess.Board()
    original = board.fen()

    def mutating_selector(
        selection: chess.Board, context: MoveContext
    ) -> chess.Move:
        del context
        move = min(selection.legal_moves, key=chess.Move.uci)
        selection.push(move)
        return move

    scorer = LexicographicRolloutScorer(
        mutating_selector,
        mutating_selector,
        RolloutConfig(rollouts=1, max_plies=3),
    )
    scorer.score_moves(board, _context())
    assert board.fen() == original

    def illegal_selector(
        selection: chess.Board, context: MoveContext
    ) -> chess.Move:
        del selection, context
        return chess.Move.from_uci("a1a8")

    bad = LexicographicRolloutScorer(
        _first_legal,
        illegal_selector,
        RolloutConfig(rollouts=1, max_plies=2),
    )
    with pytest.raises(AgentError, match="opponent.*illegal"):
        bad.score_moves(board, _context())


def test_scorer_rejects_wrong_role_and_terminal_positions() -> None:
    scorer = LexicographicRolloutScorer(
        _first_legal,
        _first_legal,
        RolloutConfig(rollouts=1, max_plies=1),
    )

    with pytest.raises(AgentError, match="target's turn"):
        scorer.score_moves(chess.Board(), _context(chess.BLACK))
    terminal = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    with pytest.raises(AgentError, match="terminal"):
        scorer.score_moves(terminal, _context(chess.BLACK))


def test_immediate_target_checkmate_is_counted_as_target_win() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    scorer = LexicographicRolloutScorer(
        _first_legal,
        _first_legal,
        RolloutConfig(rollouts=2, max_plies=1),
    )

    summaries = scorer.evaluate_moves(board, _context())
    mating = [item for item in summaries if item.target_wins == 2]

    assert mating
    assert all(item.selfmates == 0 and item.draws == 0 for item in mating)
