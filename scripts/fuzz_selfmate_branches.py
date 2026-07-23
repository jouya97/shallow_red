"""Branch promising positions against a varied synthetic loser population."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import Agent, AgentError, MoveContext
from worst_chess.agents.heuristic import HeuristicAgent
from worst_chess.agents.neural import NeuralAgent
from worst_chess.agents.opponent_model import (
    RandomReplySearchAgent,
    StalemateAwareRandomReplySearchAgent,
)
from worst_chess.agents.synthetic_loser import (
    ExploringLoserAgent,
    build_synthetic_loser_league,
)
from worst_chess.chess.actions import decode_action
from worst_chess.evaluation.match import MatchConfig, MatchResult, play_match
from worst_chess.training.ranked_dataset import read_ranked_jsonl


@dataclass(frozen=True, slots=True)
class FuzzNode:
    fen: str
    target_color: chess.Color
    root_id: str
    lineage: str
    generation: int
    pressure: float
    immediate_mate_probability: float
    sampled_target_wins: int = 0
    sampled_losses: int = 0
    preferred_moves: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FuzzConfig:
    checkpoint: Path
    branch_moves: int
    samples_per_move: int
    segment_plies: int
    target_top_k: int
    target_exploration: float
    seed: int


@dataclass(frozen=True, slots=True)
class BranchResult:
    record: dict[str, Any]
    child: FuzzNode | None
    pgn: str | None


class ForcedFirstMoveAgent:
    """Play a supplied root move once, then delegate future target turns."""

    def __init__(self, move: chess.Move, fallback: Agent) -> None:
        self.move = move
        self.fallback = fallback
        self.used = False

    @property
    def name(self) -> str:
        return f"forced_{self.move.uci()}_then_{self.fallback.name}"

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("ForcedFirstMoveAgent must act for the target")
        if not self.used:
            if self.move not in board.legal_moves:
                raise AgentError(f"forced root move is illegal: {self.move.uci()}")
            self.used = True
            return self.move
        return self.fallback.select_move(board, context)


_WORKER_CONFIG: FuzzConfig | None = None
_WORKER_NEURAL: NeuralAgent | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path)
    inputs.add_argument("--frontier", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--branch-moves", type=int, default=3)
    parser.add_argument("--min-legal-moves", type=int, default=2)
    parser.add_argument("--samples-per-move", type=int, default=2)
    parser.add_argument("--segment-plies", type=int, default=16)
    parser.add_argument("--target-top-k", type=int, default=12)
    parser.add_argument("--target-exploration", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--beam-objective",
        choices=("safety-first", "pressure"),
        default="safety-first",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_seed_nodes(
    dataset: Path,
    *,
    branch_moves: int,
    min_legal_moves: int,
    mirror: bool,
) -> list[FuzzNode]:
    if branch_moves < 1 or min_legal_moves < 1:
        raise ValueError("branch_moves and min_legal_moves must be positive")
    nodes: list[FuzzNode] = []
    for index, position in enumerate(read_ranked_jsonl(dataset)):
        board = position.board()
        if board.legal_moves.count() < min_legal_moves:
            continue
        ordered = sorted(
            position.move_targets,
            key=lambda target: (target.rank, target.action),
        )
        moves = tuple(
            decode_action(board, target.action).uci()
            for target in ordered[:branch_moves]
        )
        root_id = f"seed-{index:04d}"
        nodes.append(
            FuzzNode(
                fen=position.fen,
                target_color=position.target_color,
                root_id=root_id,
                lineage=root_id,
                generation=0,
                pressure=0.0,
                immediate_mate_probability=0.0,
                preferred_moves=moves,
            )
        )
        if mirror:
            mirrored = board.mirror()
            mirrored_moves = tuple(
                _mirror_move(chess.Move.from_uci(move)).uci() for move in moves
            )
            nodes.append(
                FuzzNode(
                    fen=mirrored.fen(en_passant="fen"),
                    target_color=not position.target_color,
                    root_id=f"{root_id}-mirror",
                    lineage=f"{root_id}-mirror",
                    generation=0,
                    pressure=0.0,
                    immediate_mate_probability=0.0,
                    preferred_moves=mirrored_moves,
                )
            )
    return nodes


def load_frontier_nodes(path: Path) -> list[FuzzNode]:
    """Resume a prior fuzzer frontier without inventing new root families."""

    nodes: list[FuzzNode] = []
    seen: set[tuple[str, chess.Color]] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line:
            continue
        record = json.loads(line)
        color_name = record.get("target_color")
        if color_name not in {"white", "black"}:
            raise ValueError(f"frontier record {index} has invalid target_color")
        target_color = color_name == "white"
        fen = record.get("fen")
        if not isinstance(fen, str):
            raise ValueError(f"frontier record {index} must contain a FEN")
        board = chess.Board(fen)
        if not board.is_valid() or board.turn != target_color:
            raise ValueError(f"frontier record {index} is not target-to-move")
        key = (_position_key(fen), target_color)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(
            FuzzNode(
                fen=board.fen(en_passant="fen"),
                target_color=target_color,
                root_id=str(record["root_id"]),
                lineage=str(record["lineage"]),
                generation=int(record["generation"]),
                pressure=float(record["pressure"]),
                immediate_mate_probability=float(record["immediate_mate_probability"]),
                sampled_target_wins=int(record.get("sampled_target_wins", 0)),
                sampled_losses=int(record.get("sampled_losses", 0)),
            )
        )
    if not nodes:
        raise ValueError("frontier must contain at least one valid node")
    return nodes


def _initialize_worker(config: FuzzConfig) -> None:
    global _WORKER_CONFIG, _WORKER_NEURAL
    _WORKER_CONFIG = config
    _WORKER_NEURAL = NeuralAgent.from_checkpoint(config.checkpoint, device="cpu")


def _expand_node(node: FuzzNode) -> list[BranchResult]:
    if _WORKER_CONFIG is None or _WORKER_NEURAL is None:
        raise RuntimeError("fuzzer worker was not initialized")
    config = _WORKER_CONFIG
    neural = _WORKER_NEURAL
    board = chess.Board(node.fen)
    context = MoveContext(
        game_id=f"fuzzer-candidates/{node.lineage}",
        ply=board.ply(),
        seed=config.seed + node.generation,
        target_color=node.target_color,
    )
    candidates = (
        tuple(chess.Move.from_uci(move) for move in node.preferred_moves)
        if node.preferred_moves
        else branch_candidates(
            board,
            context,
            neural,
            count=config.branch_moves,
        )
    )
    safe_moves = set(safe_root_moves(board))
    candidates = tuple(move for move in candidates if move in safe_moves)
    if len(candidates) < config.branch_moves:
        supplements = branch_candidates(
            board,
            context,
            neural,
            count=config.branch_moves,
        )
        candidates = tuple(dict.fromkeys((*candidates, *supplements)))[
            : config.branch_moves
        ]
    results: list[BranchResult] = []
    for candidate_index, move in enumerate(candidates[: config.branch_moves]):
        if move not in board.legal_moves:
            raise ValueError(f"fuzzer candidate is illegal: {move.uci()}")
        for sample in range(config.samples_per_move):
            branch_seed = _branch_seed(
                config.seed,
                node.lineage,
                node.generation,
                candidate_index,
                sample,
            )
            game_id = (
                f"{node.lineage}/g{node.generation:02d}/{move.uci()}-s{sample:02d}"
            )
            target_base = StalemateAwareRandomReplySearchAgent(
                neural,
                top_k=config.target_top_k,
            )
            target = ForcedFirstMoveAgent(
                move,
                ExploringLoserAgent(
                    target_base,
                    exploration_probability=config.target_exploration,
                    salt=f"selfmate-fuzzer-target/{game_id}",
                ),
            )
            opponent = build_synthetic_loser_league(
                neural,
                salt=f"selfmate-fuzzer-opponent/{game_id}",
            )
            match = play_match(
                target if node.target_color == chess.WHITE else opponent,
                opponent if node.target_color == chess.WHITE else target,
                MatchConfig(
                    game_id=game_id,
                    seed=branch_seed,
                    target_color=node.target_color,
                    initial_fen=node.fen,
                    max_plies=config.segment_plies,
                ),
            )
            results.append(
                _branch_result(
                    node,
                    move=move,
                    sample=sample,
                    match=match,
                )
            )
    return results


def branch_candidates(
    board: chess.Board,
    context: MoveContext,
    neural: NeuralAgent,
    *,
    count: int,
) -> tuple[chess.Move, ...]:
    """Build a stable, diverse shortlist for a newly reached position."""

    if count < 1:
        raise ValueError("count must be positive")
    if board.turn != context.target_color:
        raise ValueError("branch candidates require the target to move")
    legal = safe_root_moves(board)
    if not legal:
        raise ValueError("branch candidates require a nonterminal position")
    proposals: list[chess.Move] = []

    random_reply = RandomReplySearchAgent(None, top_k=64)
    random_best = max(
        legal,
        key=lambda move: (
            random_reply.score_move(board, move, context.target_color),
            _reverse_uci(move),
        ),
    )
    proposals.append(random_best)

    heuristic = HeuristicAgent()
    heuristic_best = max(
        legal,
        key=lambda move: (
            heuristic.score_move(board, move, context.target_color),
            _reverse_uci(move),
        ),
    )
    proposals.append(heuristic_best)
    proposals.extend(
        item.move for item in neural.rank_moves(board, context, top_k=count)
    )
    deduplicated: list[chess.Move] = []
    legal_set = set(legal)
    for move in proposals:
        if move in legal_set and move not in deduplicated:
            deduplicated.append(move)
        if len(deduplicated) == count:
            break
    return tuple(deduplicated)


def safe_root_moves(board: chess.Board) -> tuple[chess.Move, ...]:
    """Reject avoidable immediate wins, terminal draws, and repetitions."""

    candidates: list[tuple[chess.Move, chess.Board]] = []
    for move in sorted(board.legal_moves, key=chess.Move.uci):
        position = board.copy(stack=True)
        position.push(move)
        candidates.append((move, position))
    if not candidates:
        return ()

    def retain(predicate: Callable[[chess.Board], bool]) -> None:
        nonlocal candidates
        filtered = [item for item in candidates if predicate(item[1])]
        if filtered:
            candidates = filtered

    retain(lambda position: not position.is_checkmate())
    retain(lambda position: not position.is_game_over(claim_draw=False))
    retain(lambda position: not position.is_repetition(2))
    return tuple(move for move, _ in candidates)


def _branch_result(
    parent: FuzzNode,
    *,
    move: chess.Move,
    sample: int,
    match: MatchResult,
) -> BranchResult:
    if match.protocol_failure is not None:
        outcome = "protocol_failure"
    elif match.target_was_checkmated:
        outcome = "loss"
    elif match.target_won:
        outcome = "win"
    elif match.truncated:
        outcome = "frontier"
    else:
        outcome = "draw"
    child: FuzzNode | None = None
    pressure = 0.0
    mate_probability = 0.0
    if outcome == "frontier":
        child_board = chess.Board(match.final_fen)
        if child_board.turn != parent.target_color:
            raise RuntimeError("even segment must return the target to move")
        pressure, mate_probability = _position_pressure(
            child_board,
            parent.target_color,
        )
        child = FuzzNode(
            fen=child_board.fen(en_passant="fen"),
            target_color=parent.target_color,
            root_id=parent.root_id,
            lineage=f"{match.config.game_id}/frontier",
            generation=parent.generation + 1,
            pressure=pressure,
            immediate_mate_probability=mate_probability,
        )
    record = {
        "game_id": match.config.game_id,
        "root_id": parent.root_id,
        "generation": parent.generation,
        "parent_fen": parent.fen,
        "forced_move": move.uci(),
        "sample": sample,
        "outcome": outcome,
        "plies": len(match.plies),
        "termination": match.termination,
        "final_fen": match.final_fen,
        "pressure": pressure,
        "immediate_mate_probability": mate_probability,
    }
    return BranchResult(
        record=record,
        child=child,
        pgn=match.pgn if outcome in {"loss", "win"} else None,
    )


def _position_pressure(
    board: chess.Board,
    target_color: chess.Color,
) -> tuple[float, float]:
    search = RandomReplySearchAgent(None, top_k=64)
    best_score = float("-inf")
    best_mate_probability = 0.0
    for move in board.legal_moves:
        evaluation = search.evaluate_move(board, move, target_color)
        if (evaluation.immediate_mate_probability, evaluation.score) > (
            best_mate_probability,
            best_score,
        ):
            best_mate_probability = evaluation.immediate_mate_probability
            best_score = evaluation.score
    return best_score, best_mate_probability


def select_novel_beam(
    children: list[FuzzNode],
    *,
    beam_width: int,
    safety_first: bool = True,
) -> list[FuzzNode]:
    """Deduplicate positions and select round-robin across root families."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    unique: dict[tuple[str, chess.Color], FuzzNode] = {}
    for child in children:
        key = (_position_key(child.fen), child.target_color)
        incumbent = unique.get(key)
        if incumbent is None or _node_sort_key(
            child,
            safety_first=safety_first,
        ) < _node_sort_key(incumbent, safety_first=safety_first):
            unique[key] = child
    groups: dict[str, list[FuzzNode]] = defaultdict(list)
    for child in unique.values():
        groups[child.root_id].append(child)
    for group in groups.values():
        group.sort(key=lambda node: _node_sort_key(node, safety_first=safety_first))

    selected: list[FuzzNode] = []
    depth = 0
    while len(selected) < beam_width:
        added = False
        for root_id in sorted(groups):
            group = groups[root_id]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == beam_width:
                    break
        if not added:
            break
        depth += 1
    return selected


def annotate_sibling_outcomes(results: list[BranchResult]) -> list[FuzzNode]:
    """Attach action-level sampled wins and losses to surviving siblings."""

    outcomes: Counter[tuple[str, str, str, str]] = Counter()
    for result in results:
        record = result.record
        outcome = record["outcome"]
        if outcome not in {"loss", "win"}:
            continue
        key = (record["root_id"], record["parent_fen"], record["forced_move"])
        outcomes[(*key, outcome)] += 1
    children: list[FuzzNode] = []
    for result in results:
        if result.child is None:
            continue
        record = result.record
        key = (record["root_id"], record["parent_fen"], record["forced_move"])
        children.append(
            replace(
                result.child,
                sampled_target_wins=outcomes[(*key, "win")],
                sampled_losses=outcomes[(*key, "loss")],
            )
        )
    return children


def run_fuzzer(
    seeds: list[FuzzNode],
    *,
    config: FuzzConfig,
    generations: int,
    beam_width: int,
    workers: int,
    beam_objective: str = "safety-first",
) -> dict[str, Any]:
    if generations < 1 or workers < 1:
        raise ValueError("generations and workers must be positive")
    if beam_objective not in {"safety-first", "pressure"}:
        raise ValueError("unknown beam objective")
    frontier = seeds
    all_records: list[dict[str, Any]] = []
    decisive_pgns: list[str] = []
    generation_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        for generation in range(generations):
            expanded = list(executor.map(_expand_node, frontier, chunksize=1))
            results = [result for group in expanded for result in group]
            records = [result.record for result in results]
            all_records.extend(records)
            decisive_pgns.extend(
                result.pgn for result in results if result.pgn is not None
            )
            children = annotate_sibling_outcomes(results)
            frontier = select_novel_beam(
                children,
                beam_width=beam_width,
                safety_first=beam_objective == "safety-first",
            )
            counts = Counter(record["outcome"] for record in records)
            generation_report = {
                "generation": generation,
                "input_nodes": len(expanded),
                "branches": len(results),
                "losses": counts["loss"],
                "wins": counts["win"],
                "draws": counts["draw"],
                "protocol_failures": counts["protocol_failure"],
                "frontier_children": len(children),
                "selected_children": len(frontier),
                "selected_root_families": len({child.root_id for child in frontier}),
                "selected_with_sampled_win_risk": sum(
                    child.sampled_target_wins > 0 for child in frontier
                ),
                "selected_with_sampled_loss": sum(
                    child.sampled_losses > 0 for child in frontier
                ),
                "max_immediate_mate_probability": max(
                    (child.immediate_mate_probability for child in frontier),
                    default=0.0,
                ),
            }
            generation_reports.append(generation_report)
            print(json.dumps(generation_report, sort_keys=True), flush=True)
            if not frontier:
                break
    counts = Counter(record["outcome"] for record in all_records)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "generations": generations,
            "beam_width": beam_width,
            "workers": workers,
            "beam_objective": beam_objective,
        },
        "summary": {
            "seed_nodes": len(seeds),
            "seed_root_families": len({seed.root_id for seed in seeds}),
            "branches": len(all_records),
            "losses": counts["loss"],
            "wins": counts["win"],
            "draws": counts["draw"],
            "protocol_failures": counts["protocol_failure"],
            "frontiers": counts["frontier"],
            "decisive_games": len(decisive_pgns),
            "final_frontier": len(frontier),
            "final_root_families": len({node.root_id for node in frontier}),
        },
        "generations": generation_reports,
        "records": all_records,
        "frontier": [
            {
                **asdict(node),
                "target_color": "white" if node.target_color else "black",
            }
            for node in frontier
        ],
        "decisive_pgns": decisive_pgns,
    }


def _node_sort_key(
    node: FuzzNode,
    *,
    safety_first: bool,
) -> tuple[float | str, ...]:
    tactical: tuple[float | str, ...] = (
        -node.immediate_mate_probability,
        -node.pressure,
        node.lineage,
    )
    if not safety_first:
        return tactical
    return (
        float(node.sampled_target_wins),
        float(-node.sampled_losses),
        *tactical,
    )


def _position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def _reverse_uci(move: chess.Move) -> tuple[int, ...]:
    return tuple(-ord(character) for character in move.uci())


def _branch_seed(*parts: object) -> int:
    import hashlib

    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def main() -> int:
    arguments = build_parser().parse_args()
    positive_names = (
        "generations",
        "beam_width",
        "branch_moves",
        "min_legal_moves",
        "samples_per_move",
        "segment_plies",
        "target_top_k",
        "workers",
    )
    for name in positive_names:
        if getattr(arguments, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if arguments.segment_plies % 2:
        raise ValueError("segment-plies must be even")
    if not 0.0 <= arguments.target_exploration <= 1.0:
        raise ValueError("target-exploration must be in [0, 1]")
    config = FuzzConfig(
        checkpoint=arguments.checkpoint,
        branch_moves=arguments.branch_moves,
        samples_per_move=arguments.samples_per_move,
        segment_plies=arguments.segment_plies,
        target_top_k=arguments.target_top_k,
        target_exploration=arguments.target_exploration,
        seed=arguments.seed,
    )
    if arguments.dataset is not None:
        seeds = load_seed_nodes(
            arguments.dataset,
            branch_moves=arguments.branch_moves,
            min_legal_moves=arguments.min_legal_moves,
            mirror=arguments.mirror,
        )
    else:
        seeds = load_frontier_nodes(arguments.frontier)
    result = run_fuzzer(
        seeds,
        config=config,
        generations=arguments.generations,
        beam_width=arguments.beam_width,
        workers=arguments.workers,
        beam_objective=arguments.beam_objective,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "decisive_pgns"}
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.output / "decisive-games.pgn").write_text(
        "\n\n".join(result["decisive_pgns"]) + "\n",
        encoding="utf-8",
    )
    (arguments.output / "frontier.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in report["frontier"]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
