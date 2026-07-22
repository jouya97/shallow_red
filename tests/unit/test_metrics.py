import chess

from worst_chess.agents.base import MoveContext
from worst_chess.evaluation.match import MatchConfig, play_match
from worst_chess.evaluation.metrics import paired_bootstrap_comparison, summarize


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


def play_fools_mate(target_color: chess.Color, game_id: str):
    return play_match(
        ScriptedAgent("white", ["f2f3", "g2g4"]),
        ScriptedAgent("black", ["e7e5", "d8h4"]),
        MatchConfig(game_id=game_id, seed=1, target_color=target_color),
    )


def test_summary_keeps_target_loss_and_win_distinct() -> None:
    selfmate = play_fools_mate(chess.WHITE, "selfmate")
    target_win = play_fools_mate(chess.BLACK, "target-win")

    summary = summarize([selfmate, target_win])

    assert summary.overall.games == 2
    assert summary.overall.self_checkmate_rate == 0.5
    assert summary.overall.target_win_rate == 0.5
    assert summary.overall.draw_rate == 0.0
    assert summary.mean_target_utility == 0.0
    assert summary.median_plies_to_self_checkmate == 4.0
    assert summary.by_target_color[0].name == "white"
    assert summary.by_target_color[0].self_checkmate_rate == 1.0
    assert summary.by_target_color[1].name == "black"
    assert summary.by_target_color[1].target_win_rate == 1.0
    assert summary.termination_counts == (("checkmate", 2),)


def test_empty_summary_has_defined_zero_rates() -> None:
    summary = summarize([])

    assert summary.overall.games == 0
    assert summary.overall.self_checkmate_rate == 0.0
    assert summary.mean_target_utility is None
    assert summary.median_plies_to_self_checkmate is None


def test_paired_speed_bootstrap_uses_positive_as_faster() -> None:
    candidate = play_fools_mate(chess.WHITE, "candidate")
    baseline = play_fools_mate(chess.WHITE, "baseline")

    comparison = paired_bootstrap_comparison(
        [candidate],
        [baseline],
        metric="speed",
        resamples=100,
    )

    assert comparison.pairs == 1
    assert comparison.mean_difference == 0.0
    assert comparison.confidence_low == 0.0
    assert comparison.confidence_high == 0.0
