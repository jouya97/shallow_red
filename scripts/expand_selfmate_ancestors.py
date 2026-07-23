"""Search quiet legal predecessors of independently proven selfmates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from worst_chess.objective.proof_search import (
    ProofSearchConfig,
    ProofStatus,
    prove_forced_selfmate_after_move,
)


@dataclass(frozen=True, slots=True)
class QuietPredecessor:
    board: chess.Board
    forward_move: chess.Move


@dataclass(frozen=True, slots=True)
class TwoPlyAncestor:
    board: chess.Board
    forward_moves: tuple[chess.Move, chess.Move]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-report", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-candidates-per-seed", type=int, default=100)
    parser.add_argument("--max-extended-per-seed", type=int, default=10)
    parser.add_argument("--node-budget", type=int, default=250_000)
    parser.add_argument("--shuffle-seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def quiet_predecessors(child: chess.Board) -> Iterator[QuietPredecessor]:
    """Yield positions one reversible, non-capturing move before ``child``.

    Castling, promotion and en-passant history are intentionally not invented.
    Every returned move is replayed with python-chess and must recreate the
    child's piece placement, side to move, castling rights and en-passant state.
    """

    mover = not child.turn
    child_key = _position_state(child)
    occupied = child.occupied
    for to_square in chess.scan_forward(child.occupied_co[mover]):
        piece = child.piece_at(to_square)
        if piece is None:
            continue
        for from_square in chess.SQUARES:
            if occupied & chess.BB_SQUARES[from_square]:
                continue
            if piece.piece_type == chess.PAWN and chess.square_rank(from_square) in {
                0,
                7,
            }:
                continue
            predecessor = child.copy(stack=False)
            predecessor.turn = mover
            predecessor.ep_square = None
            predecessor.castling_rights = child.castling_rights
            predecessor.halfmove_clock = 0
            predecessor.fullmove_number = 1
            predecessor.remove_piece_at(to_square)
            predecessor.set_piece_at(from_square, piece)
            if not predecessor.is_valid():
                continue
            move = chess.Move(from_square, to_square)
            if move not in predecessor.legal_moves or predecessor.is_capture(move):
                continue
            replayed = predecessor.copy(stack=False)
            replayed.push(move)
            if _position_state(replayed) != child_key:
                continue
            yield QuietPredecessor(predecessor, move)


def two_ply_ancestors(seed: chess.Board) -> list[TwoPlyAncestor]:
    """Generate deduplicated same-side-to-move ancestors of ``seed``."""

    ancestors: dict[str, TwoPlyAncestor] = {}
    for previous_reply in quiet_predecessors(seed):
        for previous_target in quiet_predecessors(previous_reply.board):
            ancestor = previous_target.board
            key = ancestor.fen(en_passant="fen")
            candidate = TwoPlyAncestor(
                ancestor,
                (previous_target.forward_move, previous_reply.forward_move),
            )
            incumbent = ancestors.get(key)
            if incumbent is None or _move_key(candidate.forward_moves) < _move_key(
                incumbent.forward_moves
            ):
                ancestors[key] = candidate
    return list(ancestors.values())


def expand_report(
    report: dict[str, Any],
    *,
    start: int,
    count: int | None,
    max_candidates_per_seed: int,
    max_extended_per_seed: int,
    node_budget: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must not be negative")
    for name, value in (
        ("max_candidates_per_seed", max_candidates_per_seed),
        ("max_extended_per_seed", max_extended_per_seed),
        ("node_budget", node_budget),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if count is not None and count < 1:
        raise ValueError("count must be positive")

    raw_records = report.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("proof report records must be a list")
    seeds = [record for record in raw_records if record.get("status") == "proven"]
    stop = len(seeds) if count is None else min(len(seeds), start + count)
    selected = seeds[start:stop]
    output_records: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed_offset, seed_record in enumerate(selected):
        board = chess.Board(seed_record["fen"])
        target_color = seed_record["target_color"] == "white"
        if board.turn != target_color:
            raise ValueError("proven seed must have target side to move")
        forced_plies = _proven_plies(seed_record)
        ancestors = two_ply_ancestors(board)
        ancestors.sort(
            key=lambda candidate: _stable_order_key(
                candidate,
                source_id=seed_record.get("source_id", seed_offset + start),
                shuffle_seed=shuffle_seed,
            )
        )
        selected_ancestors = ancestors[:max_candidates_per_seed]
        seed_records: list[dict[str, Any]] = []
        extended_for_seed = 0
        for ancestor in selected_ancestors:
            result = prove_forced_selfmate_after_move(
                ancestor.board,
                target_color,
                ancestor.forward_moves[0],
                ProofSearchConfig(
                    max_plies=forced_plies + 2,
                    node_budget=node_budget,
                ),
            )
            distance_gain = (
                result.plies - forced_plies
                if result.status is ProofStatus.PROVEN and result.plies is not None
                else None
            )
            lower_horizon_status: str | None = None
            distance_gain_verified = False
            if distance_gain is not None and distance_gain > 0:
                lower_result = prove_forced_selfmate_after_move(
                    ancestor.board,
                    target_color,
                    ancestor.forward_moves[0],
                    ProofSearchConfig(
                        max_plies=forced_plies,
                        node_budget=node_budget,
                    ),
                )
                lower_horizon_status = lower_result.status.value
                distance_gain_verified = lower_result.status is ProofStatus.REFUTED

            previous_generation = seed_record.get("generation")
            if not isinstance(previous_generation, int):
                previous_generation = int(
                    seed_record.get("source") == "quiet-retro-expansion"
                )
            record = {
                "fen": ancestor.board.fen(en_passant="fen"),
                "target_color": seed_record["target_color"],
                "source": "quiet-retro-expansion",
                "root_source": seed_record.get(
                    "root_source", seed_record.get("source")
                ),
                "root_source_id": seed_record.get(
                    "root_source_id", seed_record.get("source_id")
                ),
                "generation": previous_generation + 1,
                "parent_fen": seed_record["fen"],
                "prelude": [move.uci() for move in ancestor.forward_moves],
                "status": result.status.value,
                "forced_plies": result.plies,
                "max_plies": forced_plies + 2,
                "nodes": result.nodes,
                "cache_hits": result.cache_hits,
                "principal_variation": [
                    move.uci() for move in result.principal_variation
                ],
                "distance_gain": distance_gain,
                "lower_horizon_status": lower_horizon_status,
                "distance_gain_verified": distance_gain_verified,
            }
            seed_records.append(record)
            output_records.append(record)
            if result.status is ProofStatus.PROVEN and result.plies is not None:
                if distance_gain_verified:
                    extended_for_seed += 1
                if extended_for_seed >= max_extended_per_seed:
                    break
        counts = Counter(record["status"] for record in seed_records)
        proven_extended = sum(
            record["status"] == "proven" and record["distance_gain_verified"]
            for record in seed_records
        )
        seed_summaries.append(
            {
                "seed_index": start + seed_offset,
                "source_id": seed_record.get("source_id"),
                "generated_ancestors": len(ancestors),
                "searched": len(seed_records),
                "proven": counts["proven"],
                "proven_extended": proven_extended,
                "refuted": counts["refuted"],
                "unknown": counts["unknown"],
            }
        )
        print(
            f"seeds={seed_offset + 1} generated={len(ancestors)} "
            f"searched={len(seed_records)} proven={counts['proven']} "
            f"extended={proven_extended} unknown={counts['unknown']}",
            flush=True,
        )

    counts = Counter(record["status"] for record in output_records)
    return {
        "seed_count": len(selected),
        "max_candidates_per_seed": max_candidates_per_seed,
        "max_extended_per_seed": max_extended_per_seed,
        "node_budget": node_budget,
        "shuffle_seed": shuffle_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "generated_ancestors": sum(
                summary["generated_ancestors"] for summary in seed_summaries
            ),
            "searched": len(output_records),
            "proven": counts["proven"],
            "proven_extended": sum(
                record["status"] == "proven" and record["distance_gain_verified"]
                for record in output_records
            ),
            "refuted": counts["refuted"],
            "unknown": counts["unknown"],
        },
        "seeds": seed_summaries,
        "records": output_records,
    }


def _position_state(board: chess.Board) -> tuple[str, bool, int, int | None]:
    return (
        board.board_fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
    )


def _move_key(moves: tuple[chess.Move, chess.Move]) -> tuple[str, str]:
    return moves[0].uci(), moves[1].uci()


def _stable_order_key(
    candidate: TwoPlyAncestor,
    *,
    source_id: object,
    shuffle_seed: int,
) -> bytes:
    value = (
        f"{shuffle_seed}|{source_id}|{candidate.board.fen(en_passant='fen')}|"
        f"{candidate.forward_moves[0].uci()}|{candidate.forward_moves[1].uci()}"
    )
    return hashlib.sha256(value.encode()).digest()


def _proven_plies(record: dict[str, Any]) -> int:
    direct = record.get("forced_plies")
    if isinstance(direct, int):
        return direct
    attempts = record.get("attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("seed attempts must be a list")
    for attempt in attempts:
        if attempt.get("status") == "proven" and isinstance(
            attempt.get("forced_plies"), int
        ):
            return attempt["forced_plies"]
    raise ValueError("proven seed has no finite proven attempt")


def main() -> int:
    arguments = build_parser().parse_args()
    report = json.loads(arguments.proof_report.read_text(encoding="utf-8"))
    result = expand_report(
        report,
        start=arguments.start,
        count=arguments.count,
        max_candidates_per_seed=arguments.max_candidates_per_seed,
        max_extended_per_seed=arguments.max_extended_per_seed,
        node_budget=arguments.node_budget,
        shuffle_seed=arguments.shuffle_seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
