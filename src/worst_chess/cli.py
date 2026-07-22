"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch

from worst_chess import __version__
from worst_chess.agents.adapters import SelfishLoserOpponentAgent
from worst_chess.agents.base import Agent, MoveContext
from worst_chess.agents.exploit import FrozenTargetExploitOpponentAgent
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.opponent_model import (
    OpportunisticHybridAgent,
    RandomReplySearchAgent,
    SampledExpectimaxConfig,
    StalemateAwareRandomReplySearchAgent,
    TwoTurnRandomReplyAgent,
)
from worst_chess.agents.policy_search import PolicyGuidedReverseSearchAgent
from worst_chess.agents.portfolio import RegimeSwitchingOpponentAgent
from worst_chess.agents.random import RandomAgent
from worst_chess.agents.resistant import ResistantOpponentAgent
from worst_chess.agents.rollout_search import NeuralShortlistRolloutAgent
from worst_chess.agents.stockfish import (
    LimitedStrengthStockfishAgent,
    ReverseStockfishAgent,
    StockfishAgent,
)
from worst_chess.agents.tablebase import SyzygyLosingAgent
from worst_chess.agents.weak import (
    CaptureFirstOpponentAgent,
    MaterialOpponentAgent,
    NoisyOpponentAgent,
)
from worst_chess.chess.neural_actions import (
    ABSOLUTE_ACTION_ORIENTATION,
    ACTION_ORIENTATION_METADATA_KEY,
    PERSPECTIVE_ACTION_ORIENTATION,
)
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
    RankedPosition,
    generate_ranked_trajectories,
    rank_position,
    read_ranked_jsonl,
    split_ranked_by_trajectory,
    write_ranked_jsonl,
)
from worst_chess.training.ranked_trainer import (
    RankedTrainingConfig,
    evaluate_ranked,
    train_ranked,
)
from worst_chess.training.rollout_teacher import (
    LexicographicRolloutScorer,
    RolloutConfig,
)
from worst_chess.training.trainer import TrainingConfig, evaluate_policy, train_policy
from worst_chess.uci import run_uci
from worst_chess.verification import verify_action_roundtrips

OPPONENT_CHOICES = (
    "stockfish",
    "stockfish-skill0",
    "stockfish-elo1320",
    "material",
    "capture-first",
    "noisy-material-25",
    "noisy-material-50",
    "noisy-material-75",
    "noisy-material-90",
    "noisy-material-95",
    "noisy-material-99",
    "weak-portfolio",
    "stress-portfolio",
    "random",
    "resistant",
)
SMOKE_OPPONENT_CHOICES = (
    *OPPONENT_CHOICES,
    "selfish-loser",
    "selfish-reverse-stockfish",
    "frozen-target-exploit",
)


@dataclass(frozen=True, slots=True)
class _RolloutRerankTask:
    position: RankedPosition
    input_index: int
    output_index: int
    source_id: str
    seed: int


_ROLLOUT_WORKER_SCORER: LexicographicRolloutScorer | None = None


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
            "random-reply",
            "stalemate-aware",
            "two-turn-random-reply",
            "opportunistic",
            "rollout-search",
        ),
        default="heuristic",
    )
    smoke.add_argument(
        "--opponent",
        choices=SMOKE_OPPONENT_CHOICES,
        default="stockfish",
    )
    smoke.add_argument("--pairs", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=20260721)
    smoke.add_argument("--max-plies", type=int, default=300)
    smoke.add_argument("--target-nodes", type=int, default=64)
    smoke.add_argument("--search-top-k", type=int, default=8)
    smoke.add_argument(
        "--reply-samples",
        type=int,
        default=4,
        help="common-random opponent replies per root for two-turn search",
    )
    smoke.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="counterfactual samples per root move for rollout-search",
    )
    smoke.add_argument(
        "--rollout-plies",
        type=int,
        default=80,
        help="maximum plies per counterfactual rollout",
    )
    smoke.add_argument("--reply-pressure-scale", type=float, default=1.0)
    smoke.add_argument("--reply-pressure-min-material", type=int, default=0)
    smoke.add_argument(
        "--cycle-penalty",
        type=float,
        default=0.0,
        help="stalemate-aware penalty for recreating a prior position",
    )
    mate_override = smoke.add_mutually_exclusive_group()
    mate_override.add_argument(
        "--tactical-mate-override",
        action="store_true",
        help="scan all legal target moves for immediate self-mate replies",
    )
    mate_override.add_argument(
        "--forced-mate-override",
        action="store_true",
        help="override only when every legal opponent reply is immediate mate",
    )
    smoke.add_argument("--opponent-nodes", type=int, default=1_000)
    smoke.add_argument(
        "--exploit-candidates",
        type=int,
        default=24,
        help="target-policy lookahead candidates for frozen-target-exploit",
    )
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
    smoke.add_argument(
        "--tablebase",
        type=Path,
        help="optional local Syzygy directory for exact endgame guidance",
    )
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
        "--teacher",
        choices=("reverse-stockfish", "random-reply"),
        default="reverse-stockfish",
    )
    ranked.add_argument(
        "--target-policy",
        choices=("neural", "random-reply"),
        default="neural",
    )
    ranked.add_argument("--target-top-k", type=int, default=12)
    ranked.add_argument(
        "--opponent",
        choices=OPPONENT_CHOICES,
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
    rerank = subparsers.add_parser(
        "rerank-rollouts",
        description=(
            "Rerank a deterministic ranked-position subset with "
            "lexicographic neural-versus-random rollouts."
        ),
    )
    rerank.add_argument("--input", type=Path, required=True)
    rerank.add_argument("--output", type=Path, required=True)
    rerank.add_argument("--checkpoint", type=Path, required=True)
    rerank.add_argument("--positions", type=int, default=100)
    rerank.add_argument("--rollouts", type=int, default=8)
    rerank.add_argument("--rollout-plies", type=int, default=160)
    rerank.add_argument(
        "--target-continuation",
        choices=("neural", "stalemate-aware"),
        default="neural",
        help="policy used for future target turns inside each rollout",
    )
    rerank.add_argument(
        "--target-top-k",
        type=int,
        default=4,
        help="neural candidates searched by stalemate-aware continuation",
    )
    rerank.add_argument("--seed", type=int, default=20260721)
    rerank.add_argument("--device", default="cpu")
    rerank.add_argument("--workers", type=int, default=1)
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
    train_ranked_parser.add_argument(
        "--perspective-actions",
        action="store_true",
        help="vertically mirror Black policy actions to match observations",
    )
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
    reply_pressure_scale: float = 1.0,
    reply_pressure_min_material: int = 0,
    cycle_penalty: float = 0.0,
    tactical_mate_override: bool = False,
    forced_mate_override: bool = False,
    rollout_count: int = 2,
    rollout_plies: int = 80,
    rollout_seed: int = 20260721,
    reply_samples: int = 4,
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
    if name == "random-reply":
        random_policy = (
            NeuralAgent.from_checkpoint(checkpoint, device=device)
            if checkpoint is not None
            else None
        )
        return RandomReplySearchAgent(random_policy, top_k=search_top_k)
    if name == "stalemate-aware":
        safe_policy = (
            NeuralAgent.from_checkpoint(checkpoint, device=device)
            if checkpoint is not None
            else None
        )
        return StalemateAwareRandomReplySearchAgent(
            safe_policy,
            top_k=search_top_k,
            pressure_scale=reply_pressure_scale,
            pressure_min_material=reply_pressure_min_material,
            cycle_penalty=cycle_penalty,
            tactical_mate_override=tactical_mate_override,
            forced_mate_override=forced_mate_override,
        )
    if name == "two-turn-random-reply":
        if checkpoint is None:
            raise ValueError(
                "--checkpoint is required for a two-turn-random-reply target"
            )
        two_turn_policy = NeuralAgent.from_checkpoint(checkpoint, device=device)
        return TwoTurnRandomReplyAgent(
            two_turn_policy,
            top_k=search_top_k,
            config=SampledExpectimaxConfig(
                reply_samples=reply_samples,
                seed=rollout_seed,
            ),
        )
    if name == "opportunistic":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for an opportunistic target")
        if stockfish is None:
            raise ValueError("--stockfish is required for an opportunistic target")
        opportunistic_policy = NeuralAgent.from_checkpoint(
            checkpoint, device=device
        )
        opportunistic_evaluator = stack.enter_context(
            ReverseStockfishAgent(stockfish, nodes=target_nodes)
        )
        return OpportunisticHybridAgent(
            opportunistic_policy,
            opportunistic_evaluator,
            policy_top_k=search_top_k,
            reverse_top_k=min(8, search_top_k),
        )
    if name == "rollout-search":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for a rollout-search target")
        policy = NeuralAgent.from_checkpoint(
            checkpoint,
            device=device,
            agent_name="rollout-search-continuation",
        )
        return NeuralShortlistRolloutAgent(
            policy,
            top_k=search_top_k,
            config=RolloutConfig(
                rollouts=rollout_count,
                max_plies=rollout_plies,
                seed=rollout_seed,
            ),
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
    if name == "material":
        return MaterialOpponentAgent()
    if name == "capture-first":
        return CaptureFirstOpponentAgent()
    if name == "weak-portfolio":
        return RegimeSwitchingOpponentAgent(
            (
                MaterialOpponentAgent(),
                CaptureFirstOpponentAgent(),
                NoisyOpponentAgent(
                    MaterialOpponentAgent(),
                    random_move_probability=0.5,
                    salt="weak-portfolio-noise",
                ),
            ),
            regime_plies=8,
            salt="weak-portfolio-v1",
        )
    if name == "stress-portfolio":
        return RegimeSwitchingOpponentAgent(
            (
                MaterialOpponentAgent(),
                CaptureFirstOpponentAgent(),
                ResistantOpponentAgent(),
            ),
            weights=(2, 2, 1),
            regime_plies=8,
            salt="stress-portfolio-v1",
        )
    if name.startswith("noisy-material-"):
        probability = int(name.rsplit("-", maxsplit=1)[-1]) / 100
        return NoisyOpponentAgent(
            MaterialOpponentAgent(),
            random_move_probability=probability,
            salt=name,
        )
    if stockfish is None:
        raise ValueError("Stockfish was not found; pass --stockfish /path/to/stockfish")
    if name == "stockfish-skill0":
        return stack.enter_context(
            LimitedStrengthStockfishAgent(
                stockfish, skill_level=0, nodes=nodes
            )
        )
    if name == "stockfish-elo1320":
        return stack.enter_context(
            LimitedStrengthStockfishAgent(stockfish, elo=1320, nodes=nodes)
        )
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
            arguments.reply_pressure_scale,
            arguments.reply_pressure_min_material,
            arguments.cycle_penalty,
            arguments.tactical_mate_override,
            arguments.forced_mate_override,
            arguments.rollouts,
            arguments.rollout_plies,
            arguments.seed,
            arguments.reply_samples,
        )
        if arguments.tablebase is not None:
            target = stack.enter_context(
                SyzygyLosingAgent(arguments.tablebase, target)
            )
        opponent: Agent
        if arguments.opponent == "frozen-target-exploit":
            frozen_target = _target_agent(
                arguments.target,
                arguments.stockfish,
                arguments.target_nodes,
                stack,
                arguments.checkpoint,
                arguments.device,
                arguments.search_top_k,
                arguments.reply_pressure_scale,
                arguments.reply_pressure_min_material,
                arguments.cycle_penalty,
                arguments.tactical_mate_override,
                arguments.forced_mate_override,
                arguments.rollouts,
                arguments.rollout_plies,
                arguments.seed,
                arguments.reply_samples,
            )
            if arguments.tablebase is not None:
                frozen_target = stack.enter_context(
                    SyzygyLosingAgent(arguments.tablebase, frozen_target)
                )
            opponent = FrozenTargetExploitOpponentAgent(
                frozen_target,
                candidate_limit=arguments.exploit_candidates,
            )
        elif arguments.opponent in {
            "selfish-loser",
            "selfish-reverse-stockfish",
        }:
            selfish_policy_name = (
                arguments.target
                if arguments.opponent == "selfish-loser"
                else "reverse-stockfish"
            )
            selfish_policy_nodes = (
                arguments.target_nodes
                if arguments.opponent == "selfish-loser"
                else arguments.opponent_nodes
            )
            selfish_policy = _target_agent(
                selfish_policy_name,
                arguments.stockfish,
                selfish_policy_nodes,
                stack,
                arguments.checkpoint,
                arguments.device,
                arguments.search_top_k,
                arguments.reply_pressure_scale,
                arguments.reply_pressure_min_material,
                arguments.cycle_penalty,
                arguments.tactical_mate_override,
                arguments.forced_mate_override,
                arguments.rollouts,
                arguments.rollout_plies,
                arguments.seed,
                arguments.reply_samples,
            )
            opponent = SelfishLoserOpponentAgent(selfish_policy)
        else:
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
    stockfish_opponents = {
        "stockfish",
        "stockfish-skill0",
        "stockfish-elo1320",
    }
    needs_stockfish = (
        arguments.teacher == "reverse-stockfish"
        or arguments.opponent in stockfish_opponents
    )
    if needs_stockfish and arguments.stockfish is None:
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
        base_target = NeuralAgent.from_checkpoint(
            arguments.checkpoint, device=arguments.device
        )
        target: Agent
        if arguments.target_policy == "random-reply":
            target = RandomReplySearchAgent(
                base_target, top_k=arguments.target_top_k
            )
        else:
            target = base_target
        if arguments.teacher == "reverse-stockfish":
            assert arguments.stockfish is not None
            reverse_teacher = stack.enter_context(
                ReverseStockfishAgent(
                    arguments.stockfish, nodes=arguments.teacher_nodes
                )
            )
            scorer = reverse_teacher.score_moves
        else:
            scorer = RandomReplySearchAgent().score_moves
        opponent = _opponent_agent(
            arguments.opponent,
            arguments.stockfish,
            arguments.opponent_nodes,
            stack,
        )
        positions = generate_ranked_trajectories(
            scorer,
            target.select_move,
            opponent.select_move,
            trajectory_count=arguments.trajectories,
            positions_per_trajectory=arguments.positions_per_trajectory,
            max_plies=arguments.max_plies,
            seed=arguments.seed,
            source_id=(
                f"ranked-{arguments.opponent}-opponent-n{arguments.opponent_nodes}-"
                f"teacher-{arguments.teacher}-n{arguments.teacher_nodes}-"
                f"target-{arguments.target_policy}-k{arguments.target_top_k}-"
                f"opening{arguments.opening_plies}-max{arguments.max_plies}-"
                f"checkpoint-{arguments.checkpoint.stem}-seed-{arguments.seed}"
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
    source_count = len({position.source_id for position in positions})
    group_matching_suffixes = source_count > 1
    split = split_ranked_by_trajectory(
        positions,
        seed=arguments.seed,
        group_matching_suffixes=group_matching_suffixes,
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
        "perspective_actions": arguments.perspective_actions,
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
            perspective_actions=arguments.perspective_actions,
        ),
    )
    test = evaluate_ranked(
        model, split.test, device=training.device, **evaluation_options
    )
    metadata = {
        "datasets": [str(dataset) for dataset in datasets],
        "group_matching_trajectory_suffixes": group_matching_suffixes,
        ACTION_ORIENTATION_METADATA_KEY: (
            PERSPECTIVE_ACTION_ORIENTATION
            if arguments.perspective_actions
            else ABSOLUTE_ACTION_ORIENTATION
        ),
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


def _run_rerank_rollouts(arguments: argparse.Namespace) -> int:
    if arguments.input.resolve() == arguments.output.resolve():
        raise ValueError("--input and --output must be different files")
    positions = read_ranked_jsonl(arguments.input)
    if not positions:
        raise ValueError("ranked input dataset must not be empty")
    if type(arguments.positions) is not int or arguments.positions <= 0:
        raise ValueError("--positions must be a positive integer")
    if arguments.positions > len(positions):
        raise ValueError(
            f"--positions requests {arguments.positions} records, but input "
            f"contains only {len(positions)}"
        )
    if type(arguments.workers) is not int or arguments.workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if type(arguments.target_top_k) is not int or arguments.target_top_k <= 0:
        raise ValueError("--target-top-k must be a positive integer")
    if arguments.workers > 1:
        try:
            device_type = torch.device(arguments.device).type
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"invalid --device {arguments.device!r}") from error
        if device_type != "cpu":
            raise ValueError("--workers greater than 1 requires a CPU device")

    config = RolloutConfig(
        rollouts=arguments.rollouts,
        max_plies=arguments.rollout_plies,
        seed=arguments.seed,
    )
    input_digest = _file_sha256(arguments.input)[:12]
    checkpoint_digest = _file_sha256(arguments.checkpoint)[:12]

    ordered = sorted(
        enumerate(positions),
        key=lambda item: _rerank_position_key(
            item[1], index=item[0], seed=arguments.seed
        ),
    )[: arguments.positions]
    tasks: list[_RolloutRerankTask] = []
    for output_index, (input_index, position) in enumerate(ordered):
        continuation_id = (
            "target-frozen-neural"
            if arguments.target_continuation == "neural"
            else f"target-stalemate-aware-top{arguments.target_top_k}"
        )
        source_id = (
            f"{position.source_id}/rollout-rerank-v1"
            f"-{continuation_id}-opponent-uniform-random"
            f"-r{config.rollouts}-h{config.max_plies}-seed{config.seed}"
            f"-checkpoint{checkpoint_digest}-input{input_digest}"
        )
        tasks.append(
            _RolloutRerankTask(
                position=position,
                input_index=input_index,
                output_index=output_index,
                source_id=source_id,
                seed=arguments.seed,
            )
        )

    if arguments.workers == 1:
        target = _build_rollout_target(
            arguments.checkpoint,
            arguments.device,
            arguments.target_continuation,
            arguments.target_top_k,
        )
        scorer = LexicographicRolloutScorer(target, RandomAgent(), config)
        reranked = [_rerank_rollout_task(task, scorer) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=arguments.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_rollout_worker,
            initargs=(
                str(arguments.checkpoint),
                arguments.device,
                config,
                arguments.target_continuation,
                arguments.target_top_k,
            ),
        ) as executor:
            reranked = list(
                executor.map(
                    _run_rollout_worker_task,
                    tasks,
                    chunksize=1,
                )
            )

    write_ranked_jsonl(arguments.output, reranked)
    print(f"input positions: {len(positions)}")
    print(f"reranked positions: {len(reranked)}")
    print(f"rollouts per legal action: {config.rollouts}")
    print(f"rollout horizon plies: {config.max_plies}")
    print(f"workers: {arguments.workers}")
    print(f"dataset: {arguments.output}")
    return 0


def _initialize_rollout_worker(
    checkpoint: str,
    device: str,
    config: RolloutConfig,
    target_continuation: str = "neural",
    target_top_k: int = 4,
) -> None:
    global _ROLLOUT_WORKER_SCORER
    target = _build_rollout_target(
        checkpoint,
        device,
        target_continuation,
        target_top_k,
    )
    _ROLLOUT_WORKER_SCORER = LexicographicRolloutScorer(
        target,
        RandomAgent(),
        config,
    )


def _build_rollout_target(
    checkpoint: str | Path,
    device: str,
    continuation: str,
    top_k: int,
) -> Agent:
    policy = NeuralAgent.from_checkpoint(
        checkpoint,
        device=device,
        agent_name="rollout-future-target",
    )
    if continuation == "neural":
        return policy
    if continuation == "stalemate-aware":
        return StalemateAwareRandomReplySearchAgent(policy, top_k=top_k)
    raise ValueError(f"unknown rollout target continuation {continuation!r}")


def _run_rollout_worker_task(task: _RolloutRerankTask) -> RankedPosition:
    if _ROLLOUT_WORKER_SCORER is None:
        raise RuntimeError("rollout worker was not initialized")
    return _rerank_rollout_task(task, _ROLLOUT_WORKER_SCORER)


def _rerank_rollout_task(
    task: _RolloutRerankTask,
    scorer: LexicographicRolloutScorer,
) -> RankedPosition:
    position = task.position
    board = position.board()
    context = MoveContext(
        game_id=(
            f"{position.trajectory_id}/rollout-rerank-{task.output_index:06d}"
        ),
        ply=board.ply(),
        seed=int.from_bytes(
            _rerank_position_key(
                position, index=task.input_index, seed=task.seed
            )[:8],
            "big",
        ),
        target_color=position.target_color,
    )
    return rank_position(
        board,
        target_color=position.target_color,
        scorer=scorer,
        context=context,
        source_id=task.source_id,
        trajectory_id=position.trajectory_id,
        value_target=position.value_target,
    )


def _rerank_position_key(
    position: RankedPosition,
    *,
    index: int,
    seed: int,
) -> bytes:
    payload = json.dumps(
        [
            "rollout-rerank-subset-v1",
            seed,
            index,
            position.source_id,
            position.trajectory_id,
            position.fen,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    if arguments.command == "rerank-rollouts":
        return _run_rerank_rollouts(arguments)
    if arguments.command == "train-ranked":
        return _run_train_ranked(arguments)
    if arguments.command == "uci":
        return _run_uci(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
