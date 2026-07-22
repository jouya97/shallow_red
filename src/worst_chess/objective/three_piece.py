"""Exhaustive symmetry-reduced three-piece forced-selfmate state graphs."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from worst_chess.objective.retrograde import RetrogradeSolution, solve_forced_selfmate

_SUPPORTED_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


@dataclass(frozen=True, slots=True, order=True)
class ThreePieceState:
    """Canonical placement with White fixed as the designated target."""

    target_king: chess.Square
    opponent_king: chess.Square
    extra_square: chess.Square
    target_turn: bool


@dataclass(frozen=True, slots=True)
class ThreePieceResult:
    """Exact history-free forced-selfmate result for one material class."""

    piece_type: chess.PieceType
    extra_is_target: bool
    states: tuple[ThreePieceState, ...]
    solution: RetrogradeSolution
    successful_terminals: int

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


def solve_three_piece_class(
    piece_type: chess.PieceType,
    *,
    extra_is_target: bool,
) -> ThreePieceResult:
    """Enumerate and solve a closed KXvK class for Q/R/B/N.

    Capturing the extra piece enters bare-kings dead position and is represented
    by one non-winning sink.  Halfmove clocks, the 50/75-move rules, optional
    claims, and repetition history are omitted.  Cycles outside the selfmate
    attractor are therefore draws/non-forced outcomes.
    """

    if piece_type not in _SUPPORTED_PIECES:
        raise ValueError(
            "three-piece exact classes currently support queen, rook, bishop, "
            "and knight; pawns require promotion-closure across material classes"
        )
    candidates = enumerate_three_piece_states(
        piece_type,
        extra_is_target=extra_is_target,
    )
    states = _legal_states(
        candidates,
        piece_type=piece_type,
        extra_is_target=extra_is_target,
    )
    successful = {
        state_index
        for state_index, state in enumerate(states)
        if _is_target_checkmate(
            state,
            piece_type=piece_type,
            extra_is_target=extra_is_target,
        )
    }
    if not successful:
        return ThreePieceResult(
            piece_type=piece_type,
            extra_is_target=extra_is_target,
            states=states,
            solution=RetrogradeSolution(
                forced_selfmate=(False,) * len(states),
                plies=(None,) * len(states),
            ),
            successful_terminals=0,
        )

    index = {state: state_index for state_index, state in enumerate(states)}
    sink = len(states)
    successors: list[tuple[int, ...]] = []
    target_turn: list[bool] = []
    for state in states:
        board = state_board(
            state,
            piece_type=piece_type,
            extra_is_target=extra_is_target,
        )
        target_turn.append(state.target_turn)
        if board.is_game_over(claim_draw=False):
            successors.append(())
            continue
        children: set[int] = set()
        for move in board.legal_moves:
            position = board.copy(stack=False)
            position.push(move)
            extra = [
                (square, piece)
                for square, piece in position.piece_map().items()
                if piece.piece_type != chess.KING
            ]
            if not extra:
                children.add(sink)
                continue
            if len(extra) != 1 or extra[0][1].piece_type != piece_type:
                raise RuntimeError("non-pawn three-piece class was not closed")
            child = canonical_state(
                target_king=_required_king(position, chess.WHITE),
                opponent_king=_required_king(position, chess.BLACK),
                extra_square=extra[0][0],
                target_turn=position.turn == chess.WHITE,
            )
            try:
                children.add(index[child])
            except KeyError as error:
                raise RuntimeError(f"missing canonical successor {child}") from error
        successors.append(tuple(sorted(children)))

    # Bare kings are an automatic dead-position draw and never a successful
    # selfmate. It is intentionally a zero-successor failure sink.
    successors.append(())
    target_turn.append(False)
    solution = solve_forced_selfmate(successors, target_turn, successful)
    # The sink is an implementation detail, not a material-class state.
    trimmed = RetrogradeSolution(
        solution.forced_selfmate[:-1],
        solution.plies[:-1],
    )
    return ThreePieceResult(
        piece_type=piece_type,
        extra_is_target=extra_is_target,
        states=states,
        solution=trimmed,
        successful_terminals=len(successful),
    )


def solve_pawn_three_piece_class(*, extra_is_target: bool) -> ThreePieceResult:
    """Solve KPvK with promotion links to exact KQ/KR/KB/KN-v-K results.

    The pawn is always Black when ``extra_is_target`` is false and White
    otherwise. Only file reflection is a valid symmetry because pawn movement
    has a fixed rank direction. Promotion successors are classified using the
    exhaustive non-pawn solutions rather than being assumed terminal.
    """

    promoted_results = {
        promoted_type: solve_three_piece_class(
            promoted_type,
            extra_is_target=extra_is_target,
        )
        for promoted_type in _SUPPORTED_PIECES
    }
    states = _legal_states(
        enumerate_pawn_three_piece_states(extra_is_target=extra_is_target),
        piece_type=chess.PAWN,
        extra_is_target=extra_is_target,
    )
    index = {state: state_index for state_index, state in enumerate(states)}
    successful = {
        state_index
        for state_index, state in enumerate(states)
        if _is_target_checkmate(
            state,
            piece_type=chess.PAWN,
            extra_is_target=extra_is_target,
        )
    }
    if not successful and all(
        result.solution.forced_count == 0 for result in promoted_results.values()
    ):
        return _zero_result(
            chess.PAWN,
            extra_is_target=extra_is_target,
            states=states,
        )

    promoted_indexes = {
        promoted_type: {
            state: state_index for state_index, state in enumerate(result.states)
        }
        for promoted_type, result in promoted_results.items()
    }

    failure_sink = len(states)
    success_sink = failure_sink + 1
    successors: list[tuple[int, ...]] = []
    target_turn: list[bool] = []
    for state in states:
        board = state_board(
            state,
            piece_type=chess.PAWN,
            extra_is_target=extra_is_target,
        )
        target_turn.append(state.target_turn)
        if board.is_game_over(claim_draw=False):
            successors.append(())
            continue
        children: set[int] = set()
        for move in board.legal_moves:
            position = board.copy(stack=False)
            position.push(move)
            extra = [
                (square, piece)
                for square, piece in position.piece_map().items()
                if piece.piece_type != chess.KING
            ]
            if not extra:
                children.add(failure_sink)
                continue
            if len(extra) != 1:
                raise RuntimeError("three-piece pawn transition changed piece count")
            extra_square, extra_piece = extra[0]
            if extra_piece.piece_type == chess.PAWN:
                child = canonical_pawn_state(
                    target_king=_required_king(position, chess.WHITE),
                    opponent_king=_required_king(position, chess.BLACK),
                    extra_square=extra_square,
                    target_turn=position.turn == chess.WHITE,
                )
                try:
                    children.add(index[child])
                except KeyError as error:
                    raise RuntimeError(
                        f"missing canonical pawn successor {child}"
                    ) from error
                continue
            promoted_type = extra_piece.piece_type
            if promoted_type not in promoted_results:
                raise RuntimeError("pawn promoted to unsupported piece")
            promoted_state = canonical_state(
                target_king=_required_king(position, chess.WHITE),
                opponent_king=_required_king(position, chess.BLACK),
                extra_square=extra_square,
                target_turn=position.turn == chess.WHITE,
            )
            try:
                promoted_index = promoted_indexes[promoted_type][promoted_state]
            except KeyError as error:
                raise RuntimeError(
                    f"missing promoted successor {promoted_state}"
                ) from error
            promoted_solution = promoted_results[promoted_type].solution
            promoted_plies = promoted_solution.plies[promoted_index]
            if promoted_plies is None:
                children.add(failure_sink)
            elif promoted_plies == 0:
                children.add(success_sink)
            else:
                raise RuntimeError(
                    "promotion closure requires embedding a nonterminal "
                    "non-pawn forced state"
                )
        successors.append(tuple(sorted(children)))

    # Bare kings and non-forced promoted positions are collapsed into one
    # failure sink. Exact promoted checkmates share one success terminal.
    successors.extend(((), ()))
    target_turn.extend((False, True))
    solution = solve_forced_selfmate(
        successors,
        target_turn,
        successful | {success_sink},
    )
    return ThreePieceResult(
        piece_type=chess.PAWN,
        extra_is_target=extra_is_target,
        states=states,
        solution=RetrogradeSolution(
            solution.forced_selfmate[: len(states)],
            solution.plies[: len(states)],
        ),
        successful_terminals=len(successful),
    )


def enumerate_three_piece_states(
    piece_type: chess.PieceType,
    *,
    extra_is_target: bool,
) -> tuple[ThreePieceState, ...]:
    """Enumerate every legal placement modulo all eight board symmetries."""

    if piece_type not in _SUPPORTED_PIECES:
        raise ValueError("unsupported non-pawn three-piece material")
    # Ownership is accepted here so callers can use one material-class API;
    # legality is filtered separately because opposite-check validity depends
    # on which side owns the extra piece.
    del extra_is_target
    states: list[ThreePieceState] = []
    for target_king in chess.SQUARES:
        for opponent_king in chess.SQUARES:
            if target_king == opponent_king:
                continue
            if chess.square_distance(target_king, opponent_king) <= 1:
                continue
            for extra_square in chess.SQUARES:
                if extra_square in (target_king, opponent_king):
                    continue
                placement = canonical_placement(
                    target_king,
                    opponent_king,
                    extra_square,
                )
                if placement != (target_king, opponent_king, extra_square):
                    continue
                for turn in (False, True):
                    state = ThreePieceState(*placement, target_turn=turn)
                    states.append(state)
    return tuple(states)


def enumerate_pawn_three_piece_states(
    *,
    extra_is_target: bool,
) -> tuple[ThreePieceState, ...]:
    """Enumerate KPvK placements modulo file reflection only."""

    del extra_is_target
    states: list[ThreePieceState] = []
    pawn_squares = tuple(
        square for square in chess.SQUARES if chess.square_rank(square) not in (0, 7)
    )
    for target_king in chess.SQUARES:
        for opponent_king in chess.SQUARES:
            if target_king == opponent_king:
                continue
            if chess.square_distance(target_king, opponent_king) <= 1:
                continue
            for extra_square in pawn_squares:
                if extra_square in (target_king, opponent_king):
                    continue
                placement = canonical_pawn_placement(
                    target_king,
                    opponent_king,
                    extra_square,
                )
                if placement != (target_king, opponent_king, extra_square):
                    continue
                for turn in (False, True):
                    states.append(ThreePieceState(*placement, target_turn=turn))
    return tuple(states)


def _legal_states(
    states: tuple[ThreePieceState, ...],
    *,
    piece_type: chess.PieceType,
    extra_is_target: bool,
) -> tuple[ThreePieceState, ...]:
    return tuple(
        state
        for state in states
        if state_board(
            state,
            piece_type=piece_type,
            extra_is_target=extra_is_target,
        ).is_valid()
    )


def _is_target_checkmate(
    state: ThreePieceState,
    *,
    piece_type: chess.PieceType,
    extra_is_target: bool,
) -> bool:
    board = state_board(
        state,
        piece_type=piece_type,
        extra_is_target=extra_is_target,
    )
    return board.turn == chess.WHITE and board.is_checkmate()


def state_board(
    state: ThreePieceState,
    *,
    piece_type: chess.PieceType,
    extra_is_target: bool,
) -> chess.Board:
    """Reconstruct a history-free board for a canonical state."""

    board = chess.Board.empty()
    board.turn = chess.WHITE if state.target_turn else chess.BLACK
    board.set_piece_at(state.target_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(state.opponent_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(
        state.extra_square,
        chess.Piece(piece_type, chess.WHITE if extra_is_target else chess.BLACK),
    )
    board.clear_stack()
    return board


def canonical_state(
    *,
    target_king: chess.Square,
    opponent_king: chess.Square,
    extra_square: chess.Square,
    target_turn: bool,
) -> ThreePieceState:
    return ThreePieceState(
        *canonical_placement(target_king, opponent_king, extra_square),
        target_turn=target_turn,
    )


def canonical_pawn_state(
    *,
    target_king: chess.Square,
    opponent_king: chess.Square,
    extra_square: chess.Square,
    target_turn: bool,
) -> ThreePieceState:
    return ThreePieceState(
        *canonical_pawn_placement(target_king, opponent_king, extra_square),
        target_turn=target_turn,
    )


def canonical_placement(
    target_king: chess.Square,
    opponent_king: chess.Square,
    extra_square: chess.Square,
) -> tuple[chess.Square, chess.Square, chess.Square]:
    """Return the lexicographically smallest D4-equivalent placement."""

    return min(
        (
            _transform_square(target_king, symmetry),
            _transform_square(opponent_king, symmetry),
            _transform_square(extra_square, symmetry),
        )
        for symmetry in range(8)
    )


def canonical_pawn_placement(
    target_king: chess.Square,
    opponent_king: chess.Square,
    extra_square: chess.Square,
) -> tuple[chess.Square, chess.Square, chess.Square]:
    """Return the smaller of a pawn placement and its file reflection."""

    placement = (target_king, opponent_king, extra_square)
    reflected = (
        chess.square(
            7 - chess.square_file(target_king),
            chess.square_rank(target_king),
        ),
        chess.square(
            7 - chess.square_file(opponent_king),
            chess.square_rank(opponent_king),
        ),
        chess.square(
            7 - chess.square_file(extra_square),
            chess.square_rank(extra_square),
        ),
    )
    return min(placement, reflected)


def _zero_result(
    piece_type: chess.PieceType,
    *,
    extra_is_target: bool,
    states: tuple[ThreePieceState, ...],
) -> ThreePieceResult:
    return ThreePieceResult(
        piece_type=piece_type,
        extra_is_target=extra_is_target,
        states=states,
        solution=RetrogradeSolution(
            forced_selfmate=(False,) * len(states),
            plies=(None,) * len(states),
        ),
        successful_terminals=0,
    )


def _transform_square(square: chess.Square, symmetry: int) -> chess.Square:
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
    return chess.square(transformed_file, transformed_rank)


def _required_king(board: chess.Board, color: chess.Color) -> chess.Square:
    square = board.king(color)
    if square is None:
        raise RuntimeError("legal chess transition removed a king")
    return square


__all__ = [
    "ThreePieceResult",
    "ThreePieceState",
    "canonical_pawn_placement",
    "canonical_pawn_state",
    "canonical_placement",
    "canonical_state",
    "enumerate_pawn_three_piece_states",
    "enumerate_three_piece_states",
    "solve_pawn_three_piece_class",
    "solve_three_piece_class",
    "state_board",
]
