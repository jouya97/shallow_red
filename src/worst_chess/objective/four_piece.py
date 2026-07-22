"""Exact symmetry-reduced KBvKR forced-selfmate pilot.

White is the designated target and owns king plus bishop. Black resists and
owns king plus rook. The history-free graph omits castling, en passant,
halfmove clocks, optional claims, and repetition history.
"""

from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from array import array
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import chess

from worst_chess.objective.retrograde import RetrogradeSolution, solve_forced_selfmate
from worst_chess.objective.three_piece import solve_three_piece_class

_TURN_BIT = 24
_SQUARE_MASK = 0x3F
_RAW_SYMMETRY_REDUCED_STATES = 3_813_096
_KING_LEGAL_SYMMETRY_REDUCED_STATES = 3_415_776
_ESTIMATED_BYTES_PER_STATE = 438
_ESTIMATED_BYTES_PER_EDGE = 72
_MEMORY_SAFETY_FACTOR = 2.0
_MEMORY_BASELINE_BYTES = 1024**3


@dataclass(frozen=True, slots=True, order=True)
class FourPieceState:
    """Canonical KBvKR placement with White fixed as the target."""

    target_king: chess.Square
    target_bishop: chess.Square
    opponent_king: chess.Square
    opponent_rook: chess.Square
    target_turn: bool


@dataclass(frozen=True, slots=True)
class FourPieceProjection:
    """Deterministic sampled state, edge, memory, and runtime projection."""

    sample_size: int
    seed: int
    legal_samples: int
    terminal_samples: int
    raw_symmetry_reduced_states: int
    king_legal_symmetry_reduced_states: int
    projected_legal_states: int
    projected_legal_states_low_95: int
    projected_legal_states_high_95: int
    average_unique_successors: float
    projected_edges: int
    projected_peak_ram_bytes: int
    projected_graph_build_seconds: float


@dataclass(frozen=True, slots=True)
class FourPieceResult:
    """Exact history-free forced-selfmate summary and distances for KBvKR."""

    state_count: int
    edge_count: int
    successful_terminals: int
    solution: RetrogradeSolution

    @property
    def forced_nonterminal_count(self) -> int:
        return sum(
            forced and plies != 0
            for forced, plies in zip(
                self.solution.forced_selfmate,
                self.solution.plies,
                strict=True,
            )
        )


class CsrRow(Sequence[int]):
    """Zero-copy row view into packed CSR children."""

    __slots__ = ("_children", "_start", "_stop")

    def __init__(self, children: array[int], start: int, stop: int) -> None:
        self._children = children
        self._start = start
        self._stop = stop

    def __len__(self) -> int:
        return self._stop - self._start

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._children[self._start + index]

    def __iter__(self) -> Iterator[int]:
        for position in range(self._start, self._stop):
            yield self._children[position]


class CsrGraph(Sequence[Sequence[int]]):
    """Packed successor graph accepted by the generic retrograde solver."""

    __slots__ = ("children", "offsets")

    def __init__(self, offsets: array[int], children: array[int]) -> None:
        if not offsets or offsets[0] != 0 or offsets[-1] != len(children):
            raise ValueError("invalid CSR offsets")
        self.offsets = offsets
        self.children = children

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @overload
    def __getitem__(self, index: int) -> Sequence[int]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[int]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Sequence[int] | Sequence[Sequence[int]]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return CsrRow(self.children, self.offsets[index], self.offsets[index + 1])


def project_kbvkr(
    *,
    sample_size: int = 50_000,
    seed: int = 20260722,
) -> FourPieceProjection:
    """Project exact graph size from uniform labeled placements.

    The raw orbit count is exact by Burnside's lemma. Legality and branching
    are sampled deterministically. The RAM model intentionally doubles a
    conservative packed-graph plus Python-predecessor accounting.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    generator = random.Random(seed)
    legal = 0
    terminal = 0
    total_successors = 0
    started = time.perf_counter()
    for _ in range(sample_size):
        squares = generator.sample(range(64), 4)
        state = canonical_four_piece_state(
            target_king=squares[0],
            target_bishop=squares[1],
            opponent_king=squares[2],
            opponent_rook=squares[3],
            target_turn=bool(generator.getrandbits(1)),
        )
        board = four_piece_board(state)
        if not board.is_valid():
            continue
        legal += 1
        if board.is_game_over(claim_draw=False):
            terminal += 1
        else:
            total_successors += _unique_successor_count(board)
    elapsed = time.perf_counter() - started
    proportion = legal / sample_size
    low, high = _wilson_interval(legal, sample_size)
    projected_states = round(_RAW_SYMMETRY_REDUCED_STATES * proportion)
    projected_low = round(_RAW_SYMMETRY_REDUCED_STATES * low)
    projected_high = round(_RAW_SYMMETRY_REDUCED_STATES * high)
    average_successors = total_successors / legal if legal else 0.0
    projected_edges = math.ceil(projected_high * average_successors)
    projected_ram = math.ceil(
        _MEMORY_BASELINE_BYTES
        + _MEMORY_SAFETY_FACTOR
        * (
            projected_high * _ESTIMATED_BYTES_PER_STATE
            + projected_edges * _ESTIMATED_BYTES_PER_EDGE
        )
    )
    return FourPieceProjection(
        sample_size=sample_size,
        seed=seed,
        legal_samples=legal,
        terminal_samples=terminal,
        raw_symmetry_reduced_states=_RAW_SYMMETRY_REDUCED_STATES,
        king_legal_symmetry_reduced_states=(
            _KING_LEGAL_SYMMETRY_REDUCED_STATES
        ),
        projected_legal_states=projected_states,
        projected_legal_states_low_95=projected_low,
        projected_legal_states_high_95=projected_high,
        average_unique_successors=average_successors,
        projected_edges=projected_edges,
        projected_peak_ram_bytes=projected_ram,
        projected_graph_build_seconds=(elapsed / sample_size) * projected_high,
    )


def solve_kbvkr(
    *,
    maximum_ram_bytes: int = 16 * 1024**3,
    progress: Callable[[dict[str, int | str]], None] | None = None,
) -> FourPieceResult:
    """Enumerate and exactly solve the closed KBvKR reachability game."""

    if maximum_ram_bytes <= 0:
        raise ValueError("maximum_ram_bytes must be positive")

    def checkpoint(phase: str, completed: int, edges: int = 0) -> None:
        rss = _current_rss_bytes()
        update: dict[str, int | str] = {
            "phase": phase,
            "completed": completed,
            "edges": edges,
            "rss_bytes": rss,
        }
        if progress is not None:
            progress(update)
        if rss > maximum_ram_bytes:
            raise MemoryError(
                f"KBvKR RSS {rss} exceeded hard guard {maximum_ram_bytes}"
            )

    # Capturing White's bishop enters KvKR. Its exhaustive solution has only
    # already-checkmated target terminals; no nonterminal continuation is
    # forced. Capturing Black's rook enters dead KBvK.
    reduced_rook = solve_three_piece_class(chess.ROOK, extra_is_target=False)
    if reduced_rook.forced_nonterminal_count:
        raise RuntimeError("KvKR closure unexpectedly gained forced continuations")

    state_ids = array("I")
    for packed in _enumerate_legal_state_ids():
        state_ids.append(packed)
        if len(state_ids) % 250_000 == 0:
            checkpoint("enumerate_states", len(state_ids))
    checkpoint("enumerate_states_complete", len(state_ids))
    index = {packed: state_index for state_index, packed in enumerate(state_ids)}
    checkpoint("build_index_complete", len(state_ids))
    state_count = len(state_ids)
    failure_sink = state_count
    success_sink = state_count + 1
    offsets = array("Q", [0])
    children = array("I")
    target_turn: list[bool] = []
    successful: set[int] = set()

    for state_index, packed in enumerate(state_ids):
        state = unpack_four_piece_state(packed)
        board = four_piece_board(state)
        target_turn.append(state.target_turn)
        if board.is_game_over(claim_draw=False):
            if board.turn == chess.WHITE and board.is_checkmate():
                successful.add(state_index)
            offsets.append(len(children))
            continue
        child_indexes: set[int] = set()
        for move in board.legal_moves:
            position = board.copy(stack=False)
            position.push(move)
            non_kings = [
                (square, piece)
                for square, piece in position.piece_map().items()
                if piece.piece_type != chess.KING
            ]
            if len(non_kings) == 2:
                child = canonical_four_piece_state(
                    target_king=_required_king(position, chess.WHITE),
                    target_bishop=_required_piece(
                        position, chess.WHITE, chess.BISHOP
                    ),
                    opponent_king=_required_king(position, chess.BLACK),
                    opponent_rook=_required_piece(position, chess.BLACK, chess.ROOK),
                    target_turn=position.turn == chess.WHITE,
                )
                try:
                    child_indexes.add(index[pack_four_piece_state(child)])
                except KeyError as error:
                    raise RuntimeError(
                        f"missing canonical successor {child}"
                    ) from error
            elif len(non_kings) == 1:
                # Every reduced nonterminal is non-forced by the exact
                # three-piece proof. A capture that immediately mates White is
                # represented by the shared success terminal.
                if position.turn == chess.WHITE and position.is_checkmate():
                    child_indexes.add(success_sink)
                else:
                    child_indexes.add(failure_sink)
            else:
                raise RuntimeError("KBvKR transition changed material unexpectedly")
        children.extend(sorted(child_indexes))
        offsets.append(len(children))
        if (state_index + 1) % 100_000 == 0:
            checkpoint("build_graph", state_index + 1, len(children))

    offsets.extend((len(children), len(children)))
    target_turn.extend((False, True))
    del index
    del state_ids
    gc.collect()
    checkpoint("solve_start", state_count, len(children))
    graph = CsrGraph(offsets, children)
    solution = solve_forced_selfmate(
        graph,
        target_turn,
        successful | {success_sink},
        checkpoint=lambda phase, completed: checkpoint(
            f"retrograde_{phase}", completed, len(children)
        ),
    )
    return FourPieceResult(
        state_count=state_count,
        edge_count=len(children),
        successful_terminals=len(successful),
        solution=RetrogradeSolution(
            solution.forced_selfmate[:state_count],
            solution.plies[:state_count],
        ),
    )


def canonical_four_piece_state(
    *,
    target_king: chess.Square,
    target_bishop: chess.Square,
    opponent_king: chess.Square,
    opponent_rook: chess.Square,
    target_turn: bool,
) -> FourPieceState:
    placement = min(
        (
            _D4_SQUARES[symmetry][target_king],
            _D4_SQUARES[symmetry][target_bishop],
            _D4_SQUARES[symmetry][opponent_king],
            _D4_SQUARES[symmetry][opponent_rook],
        )
        for symmetry in range(8)
    )
    return FourPieceState(*placement, target_turn=target_turn)


def four_piece_board(state: FourPieceState) -> chess.Board:
    """Reconstruct a history-free KBvKR board."""

    board = chess.Board.empty()
    board.turn = chess.WHITE if state.target_turn else chess.BLACK
    board.set_piece_at(state.target_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(state.target_bishop, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(state.opponent_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(state.opponent_rook, chess.Piece(chess.ROOK, chess.BLACK))
    board.clear_stack()
    return board


def pack_four_piece_state(state: FourPieceState) -> int:
    return (
        state.target_king
        | (state.target_bishop << 6)
        | (state.opponent_king << 12)
        | (state.opponent_rook << 18)
        | (int(state.target_turn) << _TURN_BIT)
    )


def unpack_four_piece_state(packed: int) -> FourPieceState:
    return FourPieceState(
        target_king=packed & _SQUARE_MASK,
        target_bishop=(packed >> 6) & _SQUARE_MASK,
        opponent_king=(packed >> 12) & _SQUARE_MASK,
        opponent_rook=(packed >> 18) & _SQUARE_MASK,
        target_turn=bool((packed >> _TURN_BIT) & 1),
    )


def _enumerate_legal_state_ids() -> Iterator[int]:
    target_king_representatives = tuple(
        square
        for square in chess.SQUARES
        if square == min(_D4_SQUARES[symmetry][square] for symmetry in range(8))
    )
    for target_king in target_king_representatives:
        for target_bishop in chess.SQUARES:
            if target_bishop == target_king:
                continue
            for opponent_king in chess.SQUARES:
                if opponent_king in (target_king, target_bishop):
                    continue
                if chess.square_distance(target_king, opponent_king) <= 1:
                    continue
                for opponent_rook in chess.SQUARES:
                    if opponent_rook in (
                        target_king,
                        target_bishop,
                        opponent_king,
                    ):
                        continue
                    placement = canonical_four_piece_state(
                        target_king=target_king,
                        target_bishop=target_bishop,
                        opponent_king=opponent_king,
                        opponent_rook=opponent_rook,
                        target_turn=False,
                    )
                    if (
                        placement.target_king,
                        placement.target_bishop,
                        placement.opponent_king,
                        placement.opponent_rook,
                    ) != (
                        target_king,
                        target_bishop,
                        opponent_king,
                        opponent_rook,
                    ):
                        continue
                    for target_turn in (False, True):
                        state = FourPieceState(
                            target_king,
                            target_bishop,
                            opponent_king,
                            opponent_rook,
                            target_turn,
                        )
                        if four_piece_board(state).is_valid():
                            yield pack_four_piece_state(state)


def _unique_successor_count(board: chess.Board) -> int:
    children: set[tuple[int, ...] | str] = set()
    for move in board.legal_moves:
        position = board.copy(stack=False)
        position.push(move)
        non_kings = [
            piece
            for piece in position.piece_map().values()
            if piece.piece_type != chess.KING
        ]
        if len(non_kings) == 2:
            child = canonical_four_piece_state(
                target_king=_required_king(position, chess.WHITE),
                target_bishop=_required_piece(position, chess.WHITE, chess.BISHOP),
                opponent_king=_required_king(position, chess.BLACK),
                opponent_rook=_required_piece(position, chess.BLACK, chess.ROOK),
                target_turn=position.turn == chess.WHITE,
            )
            children.add(
                (
                    child.target_king,
                    child.target_bishop,
                    child.opponent_king,
                    child.opponent_rook,
                    int(child.target_turn),
                )
            )
        elif position.turn == chess.WHITE and position.is_checkmate():
            children.add("success")
        else:
            children.add("failure")
    return len(children)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def print_progress(update: dict[str, int | str]) -> None:
    """Emit one machine-readable progress record to stderr."""

    print(json.dumps(update, sort_keys=True), file=sys.stderr, flush=True)


def _required_king(board: chess.Board, color: chess.Color) -> chess.Square:
    square = board.king(color)
    if square is None:
        raise RuntimeError("legal transition removed a king")
    return square


def _required_piece(
    board: chess.Board,
    color: chess.Color,
    piece_type: chess.PieceType,
) -> chess.Square:
    squares = board.pieces(piece_type, color)
    if len(squares) != 1:
        raise RuntimeError("legal transition changed expected material")
    return next(iter(squares))


def _build_d4_squares() -> tuple[tuple[chess.Square, ...], ...]:
    transforms: list[tuple[chess.Square, ...]] = []
    for symmetry in range(8):
        transformed: list[chess.Square] = []
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            coordinates = (
                (file, rank),
                (7 - file, rank),
                (file, 7 - rank),
                (7 - file, 7 - rank),
                (rank, file),
                (7 - rank, file),
                (rank, 7 - file),
                (7 - rank, 7 - file),
            )
            transformed_file, transformed_rank = coordinates[symmetry]
            transformed.append(chess.square(transformed_file, transformed_rank))
        transforms.append(tuple(transformed))
    return tuple(transforms)


_D4_SQUARES = _build_d4_squares()


__all__ = [
    "CsrGraph",
    "FourPieceProjection",
    "FourPieceResult",
    "FourPieceState",
    "canonical_four_piece_state",
    "four_piece_board",
    "pack_four_piece_state",
    "print_progress",
    "project_kbvkr",
    "solve_kbvkr",
    "unpack_four_piece_state",
]
