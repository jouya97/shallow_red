from __future__ import annotations

from dataclasses import replace

import chess
import pytest

from worst_chess.agents.base import MoveContext
from worst_chess.evaluation.match import (
    MatchConfig,
    ReplayError,
    play_match,
    replay_match,
)


class ScriptedAgent:
    def __init__(self, name: str, moves: list[str]) -> None:
        self._name = name
        self._moves = iter(moves)

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci(next(self._moves))


class IllegalAgent:
    @property
    def name(self) -> str:
        return "illegal"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del board, context
        return chess.Move.from_uci("a1a8")


class MutatingAgent:
    @property
    def name(self) -> str:
        return "mutator"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        del context
        move = next(iter(board.legal_moves))
        board.push(move)
        return move


def fools_mate(target_color: chess.Color = chess.WHITE):
    return play_match(
        ScriptedAgent("white-script", ["f2f3", "g2g4"]),
        ScriptedAgent("black-script", ["e7e5", "d8h4"]),
        MatchConfig(game_id="fools-mate", seed=7, target_color=target_color),
    )


def test_target_self_checkmate_is_success_and_replays() -> None:
    result = fools_mate(chess.WHITE)

    assert result.target_was_checkmated
    assert not result.target_won
    assert result.target_utility == 1.0
    assert result.winner == chess.BLACK
    assert result.termination == "checkmate"
    assert len(result.plies) == 4
    assert result.protocol_failure is None
    assert replay_match(result).fen() == result.final_fen
    assert '[Result "0-1"]' in result.pgn


def test_target_delivering_checkmate_is_failure_of_objective() -> None:
    result = fools_mate(chess.BLACK)

    assert result.target_won
    assert not result.target_was_checkmated
    assert result.target_utility == -1.0


def test_illegal_move_is_protocol_failure_not_a_successful_loss() -> None:
    result = play_match(
        IllegalAgent(),
        ScriptedAgent("unused", []),
        MatchConfig(game_id="illegal", seed=1, target_color=chess.WHITE),
    )

    assert result.protocol_failure is not None
    assert result.protocol_failure.kind == "illegal_move"
    assert result.target_utility == -1.0
    assert not result.target_was_checkmated


def test_agent_may_not_mutate_supplied_board() -> None:
    result = play_match(
        MutatingAgent(),
        ScriptedAgent("unused", []),
        MatchConfig(game_id="mutation", seed=1, target_color=chess.WHITE),
    )

    assert result.protocol_failure is not None
    assert result.protocol_failure.kind == "board_mutation"
    assert result.final_fen == chess.STARTING_FEN


def test_max_plies_is_a_truncation_not_a_draw() -> None:
    result = play_match(
        ScriptedAgent("white", ["e2e4"]),
        ScriptedAgent("black", ["e7e5"]),
        MatchConfig(
            game_id="truncated",
            seed=2,
            target_color=chess.WHITE,
            max_plies=2,
        ),
    )

    assert result.truncated
    assert result.termination == "max_plies"
    assert result.target_utility is None
    assert result.winner is None


def test_replay_rejects_tampered_move() -> None:
    result = fools_mate()
    bad_first = replace(result.plies[0], move_uci="e2e4")
    tampered = replace(result, plies=(bad_first, *result.plies[1:]))

    with pytest.raises(ReplayError, match="action mismatch"):
        replay_match(tampered)


def test_replay_rejects_tampered_terminal_utility() -> None:
    result = fools_mate()

    with pytest.raises(ReplayError, match="terminal outcome"):
        replay_match(replace(result, target_utility=-1.0))


def test_match_config_rejects_bad_limits_and_positions() -> None:
    with pytest.raises(ValueError, match="max_plies"):
        MatchConfig(game_id="bad", seed=0, target_color=chess.WHITE, max_plies=0)
    with pytest.raises(ValueError):
        MatchConfig(
            game_id="bad-fen",
            seed=0,
            target_color=chess.WHITE,
            initial_fen="not a fen",
        )
