from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import chess
import pytest

from worst_chess import cli
from worst_chess.agents.base import MoveContext
from worst_chess.agents.random import RandomAgent
from worst_chess.training.ranked_dataset import (
    RankedDatasetSplit,
    RankedPosition,
    rank_position,
    read_ranked_jsonl,
    write_ranked_jsonl,
)
from worst_chess.training.rollout_teacher import RolloutConfig


def _example(trajectory_id: str) -> RankedPosition:
    board = chess.Board()

    def scorer(
        scored_board: chess.Board, context: MoveContext
    ) -> dict[chess.Move, float]:
        del context
        return {move: 0.0 for move in scored_board.legal_moves}

    return rank_position(
        board,
        target_color=chess.WHITE,
        scorer=scorer,
        context=MoveContext(trajectory_id, 0, 1, chess.WHITE),
        source_id="placeholder",
        trajectory_id=trajectory_id,
    )


def test_train_ranked_groups_multiple_sources_inside_one_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = replace(
        _example("ordinary/trajectory-000000"),
        source_id="ordinary",
        trajectory_id="ordinary/trajectory-000000",
    )
    resistant = replace(
        _example("resistant/trajectory-000000"),
        source_id="resistant",
        trajectory_id="resistant/trajectory-000000",
    )
    observed: dict[str, bool] = {}

    monkeypatch.setattr(
        cli,
        "read_ranked_jsonl",
        lambda path: (ordinary, resistant),
    )

    def capture_split(
        positions: object,
        *,
        seed: int,
        group_matching_suffixes: bool,
    ) -> RankedDatasetSplit:
        del positions, seed
        observed["group_matching_suffixes"] = group_matching_suffixes
        return RankedDatasetSplit(train=(), validation=(), test=())

    monkeypatch.setattr(cli, "split_ranked_by_trajectory", capture_split)
    arguments = argparse.Namespace(dataset=[Path("combined.jsonl")], seed=5)

    with pytest.raises(ValueError, match="validation and test"):
        cli._run_train_ranked(arguments)

    assert observed == {"group_matching_suffixes": True}


def test_train_ranked_cli_exposes_optional_perspective_actions() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "train-ranked",
            "--dataset",
            "ranked.jsonl",
            "--checkpoint",
            "model.pt",
            "--perspective-actions",
        ]
    )

    assert arguments.perspective_actions is True


def test_train_ranked_cli_exposes_finetune_and_fixed_splits() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "train-ranked",
            "--dataset",
            "train.jsonl",
            "--validation-dataset",
            "validation.jsonl",
            "--test-dataset",
            "test.jsonl",
            "--initialize-from",
            "base.pt",
            "--checkpoint",
            "candidate.pt",
        ]
    )

    assert arguments.validation_dataset == [Path("validation.jsonl")]
    assert arguments.test_dataset == [Path("test.jsonl")]
    assert arguments.initialize_from == Path("base.pt")


def test_rerank_rollouts_cli_exposes_required_controls() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "rerank-rollouts",
            "--input",
            "input.jsonl",
            "--output",
            "output.jsonl",
            "--checkpoint",
            "model.pt",
            "--start",
            "3",
            "--positions",
            "7",
            "--rollouts",
            "3",
            "--rollout-plies",
            "40",
            "--target-continuation",
            "stalemate-aware",
            "--target-top-k",
            "6",
            "--rollout-opponent",
            "synthetic-loser-league",
            "--seed",
            "99",
            "--device",
            "cpu",
            "--workers",
            "2",
        ]
    )

    assert arguments.start == 3
    assert arguments.positions == 7
    assert arguments.rollouts == 3
    assert arguments.rollout_plies == 40
    assert arguments.target_continuation == "stalemate-aware"
    assert arguments.target_top_k == 6
    assert arguments.rollout_opponent == "synthetic-loser-league"
    assert arguments.seed == 99
    assert arguments.workers == 2


def test_build_rollout_target_can_match_deployed_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_policy = object()

    monkeypatch.setattr(
        cli.NeuralAgent,
        "from_checkpoint",
        classmethod(lambda cls, path, **kwargs: frozen_policy),
    )

    target = cli._build_rollout_target(
        "model.pt",
        "cpu",
        "stalemate-aware",
        5,
    )

    assert isinstance(target, cli.StalemateAwareRandomReplySearchAgent)
    assert target.policy is frozen_policy
    assert target.top_k == 5


def test_rerank_rollouts_is_deterministic_and_preserves_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "model.pt"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    checkpoint.write_bytes(b"frozen-checkpoint")
    positions = tuple(
        replace(
            _example(f"original/trajectory-{index:06d}"),
            source_id="original",
            trajectory_id=f"original/trajectory-{index:06d}",
            value_target=(1.0, 0.0, -1.0, None)[index],
        )
        for index in range(4)
    )
    write_ranked_jsonl(input_path, positions)
    frozen_target = object()
    loaded: list[tuple[Path, str, str]] = []
    constructed: list[tuple[object, object, RolloutConfig]] = []

    def fake_load(
        cls: type[object],
        path: Path,
        *,
        device: str,
        agent_name: str,
    ) -> object:
        del cls
        loaded.append((path, device, agent_name))
        return frozen_target

    class FakeScorer:
        def __init__(
            self,
            target: object,
            opponent: object,
            config: RolloutConfig,
        ) -> None:
            constructed.append((target, opponent, config))

        def __call__(
            self, board: chess.Board, context: MoveContext
        ) -> dict[chess.Move, float]:
            assert board.turn == context.target_color
            return {
                move: float(index)
                for index, move in enumerate(
                    sorted(board.legal_moves, key=chess.Move.uci)
                )
            }

    monkeypatch.setattr(
        cli.NeuralAgent,
        "from_checkpoint",
        classmethod(fake_load),
    )
    monkeypatch.setattr(cli, "LexicographicRolloutScorer", FakeScorer)

    def run(output: Path) -> None:
        assert (
            cli.main(
                [
                    "rerank-rollouts",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                    "--start",
                    "1",
                    "--positions",
                    "2",
                    "--rollouts",
                    "3",
                    "--rollout-plies",
                    "5",
                    "--seed",
                    "41",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )

    run(first_output)
    run(second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    actual = read_ranked_jsonl(first_output)
    expected = sorted(
        enumerate(positions),
        key=lambda item: cli._rerank_position_key(
            item[1], index=item[0], seed=41
        ),
    )[1:3]
    expected_by_trajectory = {
        position.trajectory_id: position for _, position in expected
    }
    assert len(actual) == 2
    for position in actual:
        original = expected_by_trajectory[position.trajectory_id]
        assert position.fen == original.fen
        assert position.target_color == original.target_color
        assert position.value_target == original.value_target
        assert position.source_id.startswith(
            "original/rollout-rerank-v1-target-frozen-neural-"
            "opponent-uniform-random-r3-h5"
        )
        assert "checkpoint" in position.source_id
        assert "input" in position.source_id
    assert loaded == [
        (checkpoint, "cpu", "rollout-future-target"),
        (checkpoint, "cpu", "rollout-future-target"),
    ]
    assert len(constructed) == 2
    assert all(item[0] is frozen_target for item in constructed)
    assert all(isinstance(item[1], RandomAgent) for item in constructed)
    assert all(item[2] == RolloutConfig(3, 5, 41) for item in constructed)


def test_rerank_rollouts_rejects_destructive_or_oversized_requests(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    write_ranked_jsonl(input_path, (_example("trajectory"),))

    common = argparse.Namespace(
        input=input_path,
        output=input_path,
        checkpoint=checkpoint,
        start=0,
        positions=1,
        rollouts=1,
        rollout_plies=1,
        seed=1,
        device="cpu",
        workers=1,
    )
    with pytest.raises(ValueError, match="different files"):
        cli._run_rerank_rollouts(common)

    common.output = tmp_path / "output.jsonl"
    common.positions = 2
    with pytest.raises(ValueError, match="contains only 1"):
        cli._run_rerank_rollouts(common)


def test_rerank_rollouts_parallel_loads_once_per_worker_and_keeps_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "model.pt"
    sequential_output = tmp_path / "sequential.jsonl"
    parallel_output = tmp_path / "parallel.jsonl"
    checkpoint.write_bytes(b"parallel-checkpoint")
    positions = tuple(
        replace(
            _example(f"source/trajectory-{index:06d}"),
            source_id="source",
            trajectory_id=f"source/trajectory-{index:06d}",
            value_target=float(index),
        )
        for index in range(2)
    )
    write_ranked_jsonl(input_path, positions)
    loads: list[tuple[object, str]] = []

    def fake_load(
        cls: type[object],
        path: object,
        *,
        device: str,
        agent_name: str,
    ) -> object:
        del cls, agent_name
        loads.append((path, device))
        return object()

    class FakeScorer:
        def __init__(
            self,
            target: object,
            opponent: object,
            config: RolloutConfig,
        ) -> None:
            del target, opponent, config

        def __call__(
            self, board: chess.Board, context: MoveContext
        ) -> dict[chess.Move, float]:
            del context
            return {
                move: float(index)
                for index, move in enumerate(
                    sorted(board.legal_moves, key=chess.Move.uci)
                )
            }

    executor_settings: dict[str, object] = {}

    class FakeExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            executor_settings["max_workers"] = max_workers
            executor_settings["start_method"] = mp_context.get_start_method()  # type: ignore[attr-defined]
            assert callable(initializer)
            for _ in range(max_workers):
                initializer(*initargs)

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def map(
            self,
            function: object,
            tasks: object,
            *,
            chunksize: int,
        ) -> list[RankedPosition]:
            assert callable(function)
            assert chunksize == 1
            materialized = list(tasks)  # type: ignore[arg-type]
            completed = {
                task.output_index: function(task)
                for task in reversed(materialized)
            }
            return [completed[task.output_index] for task in materialized]

    monkeypatch.setattr(
        cli.NeuralAgent,
        "from_checkpoint",
        classmethod(fake_load),
    )
    monkeypatch.setattr(cli, "LexicographicRolloutScorer", FakeScorer)

    common = [
        "rerank-rollouts",
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint),
        "--positions",
        "2",
        "--rollouts",
        "1",
        "--rollout-plies",
        "1",
        "--seed",
        "8",
    ]
    assert cli.main([*common, "--output", str(sequential_output)]) == 0
    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakeExecutor)
    assert (
        cli.main(
            [
                *common,
                "--output",
                str(parallel_output),
                "--workers",
                "2",
            ]
        )
        == 0
    )

    assert sequential_output.read_bytes() == parallel_output.read_bytes()
    assert executor_settings == {"max_workers": 2, "start_method": "spawn"}
    assert len(loads) == 3  # once sequentially, then once per fake worker


def test_rerank_rollouts_rejects_parallel_accelerator_device(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_ranked_jsonl(input_path, (_example("trajectory"),))

    with pytest.raises(ValueError, match="requires a CPU"):
        cli.main(
            [
                "rerank-rollouts",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--checkpoint",
                str(tmp_path / "unused.pt"),
                "--positions",
                "1",
                "--workers",
                "2",
                "--device",
                "mps",
            ]
        )


def test_smoke_cli_exposes_selfish_loser_opponent_and_pressure() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "smoke",
            "--target",
            "stalemate-aware",
            "--opponent",
            "selfish-loser",
            "--reply-pressure-scale",
            "4",
            "--reply-pressure-min-material",
            "2000",
            "--cycle-penalty",
            "1e12",
            "--tactical-mate-override",
        ]
    )

    assert arguments.opponent == "selfish-loser"
    assert arguments.reply_pressure_scale == 4.0
    assert arguments.reply_pressure_min_material == 2_000
    assert arguments.cycle_penalty == 1e12
    assert arguments.tactical_mate_override is True


def test_smoke_cli_exposes_trying_to_lose_population() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(
        ["smoke", "--opponent", "selfish-random-reply"]
    ).opponent == "selfish-random-reply"
    assert parser.parse_args(
        ["smoke", "--opponent", "selfish-portfolio"]
    ).opponent == "selfish-portfolio"


def test_ranked_cli_exposes_adversarial_population_and_deployed_target() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "generate-ranked",
            "--checkpoint",
            "model.pt",
            "--target-policy",
            "stalemate-aware",
            "--opponent",
            "selfish-portfolio",
        ]
    )

    assert arguments.target_policy == "stalemate-aware"
    assert arguments.opponent == "selfish-portfolio"


def test_selfish_population_contains_three_distinct_adversarial_roles() -> None:
    target = RandomAgent()

    population = cli._selfish_population_opponent(target, candidate_limit=7)

    assert isinstance(population, cli.RegimeSwitchingOpponentAgent)
    assert population.weights == (2, 2, 1)
    assert len(population.members) == 3
    assert isinstance(population.members[0], cli.SelfishLoserOpponentAgent)
    assert isinstance(population.members[1], cli.SelfishLoserOpponentAgent)
    assert isinstance(population.members[2], cli.FrozenTargetExploitOpponentAgent)
    assert population.members[2].candidate_limit == 7


def test_ranked_opponent_factory_builds_trying_to_lose_population() -> None:
    target = RandomAgent()

    with ExitStack() as stack:
        opponent = cli._ranked_opponent_agent(
            "selfish-portfolio", target, None, 1_000, stack
        )

    assert isinstance(opponent, cli.RegimeSwitchingOpponentAgent)
    assert opponent.members[0].loser is target


def test_target_factory_applies_stalemate_aware_cycle_penalty() -> None:
    with ExitStack() as stack:
        target = cli._target_agent(
            "stalemate-aware",
            None,
            64,
            stack,
            None,
            "cpu",
            cycle_penalty=123.0,
        )

    assert isinstance(target, cli.StalemateAwareRandomReplySearchAgent)
    assert target.cycle_penalty == 123.0


def test_target_factory_applies_tactical_mate_override() -> None:
    with ExitStack() as stack:
        target = cli._target_agent(
            "stalemate-aware",
            None,
            64,
            stack,
            None,
            "cpu",
            tactical_mate_override=True,
        )

    assert isinstance(target, cli.StalemateAwareRandomReplySearchAgent)
    assert target.tactical_mate_override is True


def test_smoke_cli_and_target_factory_apply_forced_mate_override() -> None:
    arguments = cli.build_parser().parse_args(
        ["smoke", "--target", "stalemate-aware", "--forced-mate-override"]
    )
    with ExitStack() as stack:
        target = cli._target_agent(
            arguments.target,
            None,
            64,
            stack,
            None,
            "cpu",
            forced_mate_override=arguments.forced_mate_override,
        )

    assert isinstance(target, cli.StalemateAwareRandomReplySearchAgent)
    assert target.forced_mate_override is True


def test_smoke_cli_rejects_both_mate_override_modes() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "smoke",
                "--tactical-mate-override",
                "--forced-mate-override",
            ]
        )


def test_smoke_cli_exposes_selfish_reverse_stockfish_adversary() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "smoke",
            "--target",
            "neural",
            "--checkpoint",
            "model.pt",
            "--opponent",
            "selfish-reverse-stockfish",
            "--opponent-nodes",
            "32",
        ]
    )

    assert arguments.opponent == "selfish-reverse-stockfish"
    assert arguments.opponent_nodes == 32


def test_smoke_cli_exposes_reproducible_target_exploration() -> None:
    arguments = cli.build_parser().parse_args(
        ["smoke", "--target-exploration", "0.25"]
    )
    base = cli.HeuristicAgent()

    target = cli._exploring_target(base, arguments.target_exploration, salt="test")

    assert isinstance(target, cli.ExploringLoserAgent)
    assert target.base is base
    assert target.exploration_probability == 0.25


def test_smoke_cli_exposes_frozen_target_exploit_adversary() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "smoke",
            "--target",
            "stalemate-aware",
            "--checkpoint",
            "model.pt",
            "--opponent",
            "frozen-target-exploit",
            "--exploit-candidates",
            "12",
        ]
    )

    assert arguments.opponent == "frozen-target-exploit"
    assert arguments.exploit_candidates == 12


def test_smoke_cli_builds_neural_shortlist_rollout_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenPolicy:
        def select_move(
            self, board: chess.Board, context: MoveContext
        ) -> chess.Move:
            del context
            return min(board.legal_moves, key=chess.Move.uci)

    frozen_policy = FrozenPolicy()
    monkeypatch.setattr(
        cli.NeuralAgent,
        "from_checkpoint",
        classmethod(lambda cls, path, **kwargs: frozen_policy),
    )
    arguments = cli.build_parser().parse_args(
        [
            "smoke",
            "--target",
            "rollout-search",
            "--checkpoint",
            "model.pt",
            "--search-top-k",
            "5",
            "--rollouts",
            "3",
            "--rollout-plies",
            "48",
            "--seed",
            "123",
        ]
    )

    with ExitStack() as stack:
        target = cli._target_agent(
            arguments.target,
            arguments.stockfish,
            arguments.target_nodes,
            stack,
            arguments.checkpoint,
            arguments.device,
            arguments.search_top_k,
            rollout_count=arguments.rollouts,
            rollout_plies=arguments.rollout_plies,
            rollout_seed=arguments.seed,
        )

    assert isinstance(target, cli.NeuralShortlistRolloutAgent)
    assert target.policy is frozen_policy
    assert target.top_k == 5
    assert target.config == RolloutConfig(rollouts=3, max_plies=48, seed=123)


def test_smoke_cli_builds_two_turn_random_reply_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_policy = object()
    monkeypatch.setattr(
        cli.NeuralAgent,
        "from_checkpoint",
        classmethod(lambda cls, path, **kwargs: frozen_policy),
    )
    arguments = cli.build_parser().parse_args(
        [
            "smoke",
            "--target",
            "two-turn-random-reply",
            "--checkpoint",
            "model.pt",
            "--search-top-k",
            "5",
            "--reply-samples",
            "7",
            "--seed",
            "123",
        ]
    )

    with ExitStack() as stack:
        target = cli._target_agent(
            arguments.target,
            arguments.stockfish,
            arguments.target_nodes,
            stack,
            arguments.checkpoint,
            arguments.device,
            arguments.search_top_k,
            rollout_seed=arguments.seed,
            reply_samples=arguments.reply_samples,
        )

    assert isinstance(target, cli.TwoTurnRandomReplyAgent)
    assert target.policy is frozen_policy
    assert target.top_k == 5
    assert target.config == cli.SampledExpectimaxConfig(
        reply_samples=7,
        seed=123,
    )
