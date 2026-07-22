"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import chess
import torch

from worst_chess import __version__
from worst_chess.agents.base import Agent
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.policy_search import PolicyGuidedReverseSearchAgent
from worst_chess.agents.random import RandomAgent
from worst_chess.agents.resistant import ResistantOpponentAgent
from worst_chess.agents.stockfish import ReverseStockfishAgent, StockfishAgent
from worst_chess.evaluation.metrics import summarize
from worst_chess.evaluation.openings import generate_random_openings
from worst_chess.evaluation.report import write_report
from worst_chess.evaluation.tournament import TournamentConfig, run_paired_tournament
from worst_chess.training.dataset import (
    generate_labeled_positions,
    read_jsonl,
    split_by_trajectory,
    write_jsonl,
)
from worst_chess.training.model import (
    ModelConfig,
    PolicyValueNetwork,
    save_checkpoint,
)
from worst_chess.training.ranked_dataset import (
    generate_ranked_trajectories,
    read_ranked_jsonl,
    split_ranked_by_trajectory,
    write_ranked_jsonl,
)
from worst_chess.training.ranked_trainer import (
    RankedTrainingConfig,
    evaluate_ranked,
    train_ranked,
)
from worst_chess.training.trainer import TrainingConfig, evaluate_policy, train_policy
from worst_chess.uci import run_uci
from worst_chess.verification import verify_action_roundtrips


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worst-chess",
        description="Train and evaluate an orthodox chess engine that tries to lose.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    smoke = subparsers.add_parser(
        "smoke", description="Run a small color-paired CPU baseline evaluation."
    )
    smoke.add_argument(
        "--target",
        choices=(
            "random",
            "heuristic",
            "reverse-stockfish",
            "neural",
            "policy-guided",
        ),
        default="heuristic",
    )
    smoke.add_argument(
        "--opponent",
        choices=("stockfish", "resistant", "random"),
        default="stockfish",
    )
    smoke.add_argument("--pairs", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=20260721)
    smoke.add_argument("--max-plies", type=int, default=300)
    smoke.add_argument("--target-nodes", type=int, default=64)
    smoke.add_argument("--search-top-k", type=int, default=8)
    smoke.add_argument("--opponent-nodes", type=int, default=1_000)
    smoke.add_argument(
        "--openings",
        type=int,
        default=1,
        help="number of deterministic random opening positions",
    )
    smoke.add_argument("--opening-plies", type=int, default=6)
    smoke.add_argument("--stockfish", default=shutil.which("stockfish"))
    smoke.add_argument("--checkpoint", type=Path)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    verify = subparsers.add_parser(
        "verify-actions",
        description="Run high-volume legal move/action round-trip verification.",
    )
    verify.add_argument("--transitions", type=int, default=1_000_000)
    verify.add_argument("--seed", type=int, default=20260721)
    generate = subparsers.add_parser(
        "generate-dataset",
        description="Generate deterministic supervised losing-move labels.",
    )
    generate.add_argument(
        "--labeler",
        choices=("heuristic", "reverse-stockfish"),
        default="heuristic",
    )
    generate.add_argument("--trajectories", type=int, default=100)
    generate.add_argument("--positions-per-trajectory", type=int, default=20)
    generate.add_argument("--opening-plies", type=int, default=4)
    generate.add_argument("--seed", type=int, default=20260721)
    generate.add_argument("--stockfish", default=shutil.which("stockfish"))
    generate.add_argument("--nodes", type=int, default=64)
    generate.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/datasets/pilot.jsonl"),
    )
    train = subparsers.add_parser(
        "train",
        description="Train a compact masked policy network from a JSONL dataset.",
    )
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=20260721)
    train.add_argument("--device", default="auto")
    train.add_argument("--channels", type=int, default=32)
    train.add_argument("--residual-blocks", type=int, default=2)
    ranked = subparsers.add_parser(
        "generate-ranked",
        description="Generate all-legal-move reverse-search labels on-policy.",
    )
    ranked.add_argument("--checkpoint", type=Path, required=True)
    ranked.add_argument(
        "--opponent",
        choices=("stockfish", "resistant", "random"),
        default="stockfish",
    )
    ranked.add_argument("--trajectories", type=int, default=100)
    ranked.add_argument("--positions-per-trajectory", type=int, default=20)
    ranked.add_argument("--max-plies", type=int, default=300)
    ranked.add_argument("--opening-plies", type=int, default=6)
    ranked.add_argument("--seed", type=int, default=20260721)
    ranked.add_argument("--stockfish", default=shutil.which("stockfish"))
    ranked.add_argument("--teacher-nodes", type=int, default=64)
    ranked.add_argument("--opponent-nodes", type=int, default=1_000)
    ranked.add_argument("--device", default="cpu")
    ranked.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/datasets/ranked-on-policy.jsonl"),
    )
    train_ranked_parser = subparsers.add_parser(
        "train-ranked",
        description="Train policy ranks and loser value from a ranked dataset.",
    )
    train_ranked_parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="one or more ranked JSONL datasets",
    )
    train_ranked_parser.add_argument("--checkpoint", type=Path, required=True)
    train_ranked_parser.add_argument("--epochs", type=int, default=10)
    train_ranked_parser.add_argument("--batch-size", type=int, default=128)
    train_ranked_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_ranked_parser.add_argument("--rank-temperature", type=float, default=2.0)
    train_ranked_parser.add_argument("--value-loss-weight", type=float, default=0.25)
    train_ranked_parser.add_argument("--seed", type=int, default=20260721)
    train_ranked_parser.add_argument("--device", default="auto")
    train_ranked_parser.add_argument("--channels", type=int, default=32)
    train_ranked_parser.add_argument("--residual-blocks", type=int, default=4)
    uci = subparsers.add_parser(
        "uci",
        description="Run a trained neural checkpoint as a UCI chess engine.",
    )
    uci.add_argument("--checkpoint", type=Path, required=True)
    uci.add_argument("--device", default="cpu")
    uci.add_argument(
        "--search-stockfish",
        help="optional Stockfish binary for neural top-k reverse search",
    )
    uci.add_argument("--search-nodes", type=int, default=64)
    uci.add_argument("--search-top-k", type=int, default=8)
    return parser


def _target_agent(
    name: str,
    stockfish: str | None,
    target_nodes: int,
    stack: ExitStack,
    checkpoint: Path | None,
    device: str,
    search_top_k: int = 8,
) -> Agent:
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    if name == "neural":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for a neural target")
        return NeuralAgent.from_checkpoint(checkpoint, device=device)
    if name == "policy-guided":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for a policy-guided target")
        if stockfish is None:
            raise ValueError("--stockfish is required for a policy-guided target")
        policy = NeuralAgent.from_checkpoint(checkpoint, device=device)
        evaluator = stack.enter_context(
            ReverseStockfishAgent(stockfish, nodes=target_nodes)
        )
        return PolicyGuidedReverseSearchAgent(
            policy, evaluator, top_k=search_top_k
        )
    if stockfish is None:
        raise ValueError("--stockfish is required for a reverse-stockfish target")
    return stack.enter_context(
        ReverseStockfishAgent(stockfish, nodes=target_nodes)
    )


def _opponent_agent(
    name: str,
    stockfish: str | None,
    nodes: int,
    stack: ExitStack,
) -> Agent:
    if name == "resistant":
        return ResistantOpponentAgent()
    if name == "random":
        return RandomAgent()
    if stockfish is None:
        raise ValueError("Stockfish was not found; pass --stockfish /path/to/stockfish")
    return stack.enter_context(StockfishAgent(stockfish, nodes=nodes))


def _run_smoke(arguments: argparse.Namespace) -> int:
    opening_fens = (
        generate_random_openings(
            arguments.openings,
            arguments.opening_plies,
            arguments.seed,
        )
        if arguments.openings > 1 or arguments.opening_plies > 0
        else (chess.STARTING_FEN,)
    )
    tournament = TournamentConfig(
        tournament_id=f"smoke-{arguments.target}",
        pairs=arguments.pairs,
        base_seed=arguments.seed,
        opening_fens=opening_fens,
        max_plies=arguments.max_plies,
    )
    with ExitStack() as stack:
        target = _target_agent(
            arguments.target,
            arguments.stockfish,
            arguments.target_nodes,
            stack,
            arguments.checkpoint,
            arguments.device,
            arguments.search_top_k,
        )
        opponent = _opponent_agent(
            arguments.opponent,
            arguments.stockfish,
            arguments.opponent_nodes,
            stack,
        )
        results = run_paired_tournament(target, opponent, tournament)
    report_path, pgn_path = write_report(arguments.output, tournament, results)
    summary = summarize(results)
    print(f"games: {summary.overall.games}")
    print(f"self-checkmate rate: {summary.overall.self_checkmate_rate:.1%}")
    print(f"draw rate: {summary.overall.draw_rate:.1%}")
    print(f"target-win rate: {summary.overall.target_win_rate:.1%}")
    print(f"protocol-failure rate: {summary.overall.protocol_failure_rate:.1%}")
    print(f"report: {report_path}")
    print(f"pgn: {pgn_path}")
    return 0


def _run_generate_dataset(arguments: argparse.Namespace) -> int:
    with ExitStack() as stack:
        if arguments.labeler == "heuristic":
            labeler: Agent = HeuristicAgent()
        else:
            if arguments.stockfish is None:
                raise ValueError(
                    "Stockfish was not found; pass --stockfish /path/to/stockfish"
                )
            labeler = stack.enter_context(
                ReverseStockfishAgent(arguments.stockfish, nodes=arguments.nodes)
            )
        positions = generate_labeled_positions(
            labeler.select_move,
            trajectory_count=arguments.trajectories,
            positions_per_trajectory=arguments.positions_per_trajectory,
            seed=arguments.seed,
            source_id=f"{arguments.labeler}-seed-{arguments.seed}",
            opening_plies=arguments.opening_plies,
        )
    write_jsonl(arguments.output, positions)
    print(f"positions: {len(positions)}")
    print(f"dataset: {arguments.output}")
    return 0


def _run_train(arguments: argparse.Namespace) -> int:
    positions = read_jsonl(arguments.dataset)
    split = split_by_trajectory(positions, seed=arguments.seed)
    if not split.validation:
        raise ValueError("dataset split has no validation examples")
    if not split.test:
        raise ValueError("dataset split has no test examples")
    torch.manual_seed(arguments.seed)
    model = PolicyValueNetwork(
        ModelConfig(
            channels=arguments.channels,
            residual_blocks=arguments.residual_blocks,
        )
    )
    initial_loss, initial_top1 = evaluate_policy(
        model,
        split.validation,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    training = train_policy(
        model,
        split.train,
        split.validation,
        config=TrainingConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            device=arguments.device,
        ),
    )
    test_loss, test_top1 = evaluate_policy(
        model,
        split.test,
        batch_size=arguments.batch_size,
        device=training.device,
    )
    metadata = {
        "dataset": str(arguments.dataset),
        "seed": arguments.seed,
        "initial_validation_loss": initial_loss,
        "initial_validation_top1": initial_top1,
        "test_loss": test_loss,
        "test_top1": test_top1,
        "training": asdict(training),
    }
    save_checkpoint(model, arguments.checkpoint, metadata=metadata)
    metrics_path = arguments.checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"device: {training.device}")
    print(f"training examples: {training.training_examples}")
    print(f"initial validation top-1: {initial_top1:.1%}")
    print(f"final validation top-1: {training.epochs[-1].validation_top1:.1%}")
    print(f"test top-1: {test_top1:.1%}")
    print(f"checkpoint: {arguments.checkpoint}")
    print(f"metrics: {metrics_path}")
    return 0


def _run_generate_ranked(arguments: argparse.Namespace) -> int:
    if arguments.stockfish is None:
        raise ValueError("Stockfish was not found; pass --stockfish /path/to/stockfish")
    opening_fens = (
        generate_random_openings(
            arguments.trajectories,
            arguments.opening_plies,
            arguments.seed,
        )
        if arguments.opening_plies > 0
        else (chess.STARTING_FEN,)
    )
    with ExitStack() as stack:
        target = NeuralAgent.from_checkpoint(
            arguments.checkpoint, device=arguments.device
        )
        teacher = stack.enter_context(
            ReverseStockfishAgent(
                arguments.stockfish, nodes=arguments.teacher_nodes
            )
        )
        opponent = _opponent_agent(
            arguments.opponent,
            arguments.stockfish,
            arguments.opponent_nodes,
            stack,
        )
        positions = generate_ranked_trajectories(
            teacher.score_moves,
            target.select_move,
            opponent.select_move,
            trajectory_count=arguments.trajectories,
            positions_per_trajectory=arguments.positions_per_trajectory,
            max_plies=arguments.max_plies,
            seed=arguments.seed,
            source_id=(
                f"ranked-{arguments.opponent}-teacher-n"
                f"{arguments.teacher_nodes}-seed-{arguments.seed}"
            ),
            starting_fens=opening_fens,
        )
    write_ranked_jsonl(arguments.output, positions)
    valued = sum(position.value_target is not None for position in positions)
    print(f"positions: {len(positions)}")
    print(f"positions with terminal values: {valued}")
    print(f"dataset: {arguments.output}")
    return 0


def _run_train_ranked(arguments: argparse.Namespace) -> int:
    datasets = tuple(arguments.dataset)
    positions = tuple(
        position
        for dataset in datasets
        for position in read_ranked_jsonl(dataset)
    )
    split = split_ranked_by_trajectory(
        positions,
        seed=arguments.seed,
        group_matching_suffixes=len(datasets) > 1,
    )
    if not split.validation or not split.test:
        raise ValueError("ranked dataset must have validation and test examples")
    torch.manual_seed(arguments.seed)
    model = PolicyValueNetwork(
        ModelConfig(
            channels=arguments.channels,
            residual_blocks=arguments.residual_blocks,
        )
    )
    evaluation_options = {
        "batch_size": arguments.batch_size,
        "rank_temperature": arguments.rank_temperature,
        "value_loss_weight": arguments.value_loss_weight,
    }
    initial = evaluate_ranked(
        model, split.validation, device=arguments.device, **evaluation_options
    )
    training = train_ranked(
        model,
        split.train,
        split.validation,
        config=RankedTrainingConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            rank_temperature=arguments.rank_temperature,
            value_loss_weight=arguments.value_loss_weight,
            seed=arguments.seed,
            device=arguments.device,
        ),
    )
    test = evaluate_ranked(
        model, split.test, device=training.device, **evaluation_options
    )
    metadata = {
        "datasets": [str(dataset) for dataset in datasets],
        "group_matching_trajectory_suffixes": len(datasets) > 1,
        "seed": arguments.seed,
        "initial_validation": asdict(initial),
        "test": asdict(test),
        "training": asdict(training),
    }
    save_checkpoint(model, arguments.checkpoint, metadata=metadata)
    metrics_path = arguments.checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_validation = training.epochs[-1].validation
    if final_validation is None:
        raise RuntimeError("ranked training produced no validation metrics")
    print(f"device: {training.device}")
    print(f"training examples: {training.training_examples}")
    print(f"initial validation rank-1: {initial.rank_one_accuracy:.1%}")
    print(f"final validation rank-1: {final_validation.rank_one_accuracy:.1%}")
    print(f"test rank-1: {test.rank_one_accuracy:.1%}")
    print(f"test mean reciprocal rank: {test.mean_reciprocal_rank:.3f}")
    print(f"checkpoint: {arguments.checkpoint}")
    print(f"metrics: {metrics_path}")
    return 0


def _run_uci(arguments: argparse.Namespace) -> int:
    policy = NeuralAgent.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        agent_name="worst-chess-neural",
    )
    with ExitStack() as stack:
        agent: Agent = policy
        if arguments.search_stockfish is not None:
            evaluator = stack.enter_context(
                ReverseStockfishAgent(
                    arguments.search_stockfish, nodes=arguments.search_nodes
                )
            )
            agent = PolicyGuidedReverseSearchAgent(
                policy, evaluator, top_k=arguments.search_top_k
            )
        run_uci(agent, sys.stdin, sys.stdout)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "smoke":
        return _run_smoke(arguments)
    if arguments.command == "verify-actions":
        result = verify_action_roundtrips(arguments.transitions, seed=arguments.seed)
        print(f"verified transitions: {result.verified_transitions}")
        print(f"positions: {result.positions}")
        print(f"completed games: {result.completed_games}")
        print(f"elapsed seconds: {result.elapsed_seconds:.3f}")
        print(f"transitions/second: {result.transitions_per_second:.0f}")
        return 0
    if arguments.command == "generate-dataset":
        return _run_generate_dataset(arguments)
    if arguments.command == "train":
        return _run_train(arguments)
    if arguments.command == "generate-ranked":
        return _run_generate_ranked(arguments)
    if arguments.command == "train-ranked":
        return _run_train_ranked(arguments)
    if arguments.command == "uci":
        return _run_uci(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
