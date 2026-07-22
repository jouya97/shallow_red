"""Deterministic, rules-verified execution of an orthodox chess game."""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.pgn

from worst_chess.agents.base import Agent, MoveContext
from worst_chess.chess.actions import encode_move
from worst_chess.chess.outcomes import DrawPolicy, TargetOutcome, adjudicate
from worst_chess.objective.rewards import TARGET_WIN_UTILITY


@dataclass(frozen=True)
class MatchConfig:
    """Reproducible inputs for a single game."""

    game_id: str
    seed: int
    target_color: chess.Color
    initial_fen: str = chess.STARTING_FEN
    draw_policy: DrawPolicy = DrawPolicy.NEVER_CLAIM
    max_plies: int = 1_024

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")
        board = chess.Board(self.initial_fen)
        if not board.is_valid():
            raise ValueError(
                "initial_fen is not a valid standard position: "
                f"{self.initial_fen}"
            )


@dataclass(frozen=True)
class PlyRecord:
    """Everything needed to audit and replay one accepted move."""

    ply: int
    actor_color: chess.Color
    actor_name: str
    pre_fen: str
    legal_moves_uci: tuple[str, ...]
    legal_actions: tuple[int, ...]
    move_uci: str
    action: int
    latency_ns: int


@dataclass(frozen=True)
class ProtocolFailure:
    """An agent failure that must never count as successful losing play."""

    ply: int
    actor_color: chess.Color
    actor_name: str
    kind: str
    detail: str


@dataclass(frozen=True)
class MatchResult:
    """Immutable game artifact and target-relative result."""

    config: MatchConfig
    white_agent: str
    black_agent: str
    plies: tuple[PlyRecord, ...]
    pgn: str
    final_fen: str
    winner: chess.Color | None
    termination: str
    target_utility: float | None
    target_was_checkmated: bool
    target_won: bool
    protocol_failure: ProtocolFailure | None = None
    truncated: bool = False


class ReplayError(ValueError):
    """Raised when a stored match artifact cannot be reproduced exactly."""


def _safe_agent_name(agent: Agent) -> str:
    try:
        name = agent.name
    except Exception as error:  # pragma: no cover - defensive protocol boundary.
        return f"<name-error:{type(error).__name__}>"
    return name or "<unnamed>"


def _make_pgn(
    board: chess.Board,
    config: MatchConfig,
    white_name: str,
    black_name: str,
    result: str,
    termination: str,
) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Worst Chess Ever evaluation"
    game.headers["Site"] = "local"
    game.headers["Round"] = config.game_id
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["Target"] = "white" if config.target_color else "black"
    game.headers["Seed"] = str(config.seed)
    return str(game)


def _terminal_result(
    *,
    board: chess.Board,
    config: MatchConfig,
    white_name: str,
    black_name: str,
    plies: list[PlyRecord],
    outcome: TargetOutcome,
) -> MatchResult:
    if not outcome.terminal or outcome.termination is None:
        raise ValueError("_terminal_result requires a terminal outcome")
    result = board.result(
        claim_draw=config.draw_policy is DrawPolicy.CLAIM_AVAILABLE
    )
    termination = outcome.termination.name.lower()
    return MatchResult(
        config=config,
        white_agent=white_name,
        black_agent=black_name,
        plies=tuple(plies),
        pgn=_make_pgn(
            board,
            config,
            white_name,
            black_name,
            result,
            termination,
        ),
        final_fen=board.fen(),
        winner=outcome.winner,
        termination=termination,
        target_utility=outcome.utility,
        target_was_checkmated=outcome.target_was_checkmated,
        target_won=outcome.target_won,
    )


def _failed_result(
    *,
    board: chess.Board,
    config: MatchConfig,
    white_name: str,
    black_name: str,
    plies: list[PlyRecord],
    failure: ProtocolFailure,
) -> MatchResult:
    termination = f"protocol_failure:{failure.kind}"
    return MatchResult(
        config=config,
        white_agent=white_name,
        black_agent=black_name,
        plies=tuple(plies),
        pgn=_make_pgn(
            board,
            config,
            white_name,
            black_name,
            "*",
            termination,
        ),
        final_fen=board.fen(),
        winner=None,
        termination=termination,
        # A protocol failure is never rewarded as a successful loss, regardless
        # of which external policy caused it. Tournament code may separately
        # rerun or exclude opponent-infrastructure failures.
        target_utility=TARGET_WIN_UTILITY,
        target_was_checkmated=False,
        target_won=False,
        protocol_failure=failure,
    )


def play_match(
    white: Agent,
    black: Agent,
    config: MatchConfig,
) -> MatchResult:
    """Play one game and reject mutations, illegal moves, and agent failures."""

    board = chess.Board(config.initial_fen)
    white_name = _safe_agent_name(white)
    black_name = _safe_agent_name(black)
    records: list[PlyRecord] = []

    while True:
        outcome = adjudicate(board, config.target_color, config.draw_policy)
        if outcome.terminal:
            return _terminal_result(
                board=board,
                config=config,
                white_name=white_name,
                black_name=black_name,
                plies=records,
                outcome=outcome,
            )

        if len(records) >= config.max_plies:
            termination = "max_plies"
            return MatchResult(
                config=config,
                white_agent=white_name,
                black_agent=black_name,
                plies=tuple(records),
                pgn=_make_pgn(
                    board,
                    config,
                    white_name,
                    black_name,
                    "*",
                    termination,
                ),
                final_fen=board.fen(),
                winner=None,
                termination=termination,
                target_utility=None,
                target_was_checkmated=False,
                target_won=False,
                truncated=True,
            )

        agent = white if board.turn == chess.WHITE else black
        agent_name = white_name if board.turn == chess.WHITE else black_name
        actor_color = board.turn
        legal_moves = tuple(board.legal_moves)
        legal_uci = tuple(sorted(move.uci() for move in legal_moves))
        legal_actions = tuple(sorted(encode_move(board, move) for move in legal_moves))
        policy_board = board.copy(stack=True)
        before_fen = policy_board.fen()
        before_history = tuple(policy_board.move_stack)
        context = MoveContext(
            game_id=config.game_id,
            ply=len(records),
            seed=config.seed,
            target_color=config.target_color,
        )

        started = time.perf_counter_ns()
        try:
            move = agent.select_move(policy_board, context)
        except Exception as error:
            latency = time.perf_counter_ns() - started
            failure = ProtocolFailure(
                ply=len(records),
                actor_color=actor_color,
                actor_name=agent_name,
                kind="exception",
                detail=f"{type(error).__name__}: {error} (after {latency} ns)",
            )
            return _failed_result(
                board=board,
                config=config,
                white_name=white_name,
                black_name=black_name,
                plies=records,
                failure=failure,
            )
        latency = time.perf_counter_ns() - started

        if (
            policy_board.fen() != before_fen
            or tuple(policy_board.move_stack) != before_history
        ):
            failure = ProtocolFailure(
                ply=len(records),
                actor_color=actor_color,
                actor_name=agent_name,
                kind="board_mutation",
                detail="agent changed the supplied board or its move history",
            )
            return _failed_result(
                board=board,
                config=config,
                white_name=white_name,
                black_name=black_name,
                plies=records,
                failure=failure,
            )

        if not isinstance(move, chess.Move) or move not in legal_moves:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            failure = ProtocolFailure(
                ply=len(records),
                actor_color=actor_color,
                actor_name=agent_name,
                kind="illegal_move",
                detail=f"agent returned {detail}",
            )
            return _failed_result(
                board=board,
                config=config,
                white_name=white_name,
                black_name=black_name,
                plies=records,
                failure=failure,
            )

        action = encode_move(board, move)
        records.append(
            PlyRecord(
                ply=len(records),
                actor_color=actor_color,
                actor_name=agent_name,
                pre_fen=board.fen(),
                legal_moves_uci=legal_uci,
                legal_actions=legal_actions,
                move_uci=move.uci(),
                action=action,
                latency_ns=latency,
            )
        )
        board.push(move)


def replay_match(result: MatchResult) -> chess.Board:
    """Replay an artifact, raising if any stored position or action diverges."""

    board = chess.Board(result.config.initial_fen)
    for expected_ply, record in enumerate(result.plies):
        if record.ply != expected_ply:
            raise ReplayError(
                f"non-contiguous ply index: expected {expected_ply}, got {record.ply}"
            )
        if board.fen() != record.pre_fen:
            raise ReplayError(f"FEN mismatch before ply {record.ply}")
        legal_moves = tuple(board.legal_moves)
        legal_uci = tuple(sorted(move.uci() for move in legal_moves))
        legal_actions = tuple(sorted(encode_move(board, move) for move in legal_moves))
        if legal_uci != record.legal_moves_uci:
            raise ReplayError(f"legal move set mismatch at ply {record.ply}")
        if legal_actions != record.legal_actions:
            raise ReplayError(f"legal action set mismatch at ply {record.ply}")
        try:
            move = chess.Move.from_uci(record.move_uci)
        except ValueError as error:
            raise ReplayError(f"invalid stored UCI at ply {record.ply}") from error
        if move not in legal_moves:
            raise ReplayError(f"stored move is illegal at ply {record.ply}")
        if encode_move(board, move) != record.action:
            raise ReplayError(f"stored action mismatch at ply {record.ply}")
        board.push(move)

    if board.fen() != result.final_fen:
        raise ReplayError("final FEN mismatch")
    outcome = adjudicate(
        board,
        result.config.target_color,
        result.config.draw_policy,
    )
    if result.protocol_failure is not None:
        if outcome.terminal:
            raise ReplayError("protocol failure artifact ends in a board terminal")
        if result.target_utility != TARGET_WIN_UTILITY:
            raise ReplayError("protocol failure must have pessimistic target utility")
    elif result.truncated:
        if outcome.terminal:
            raise ReplayError("truncated artifact ends in a board terminal")
        if len(result.plies) != result.config.max_plies:
            raise ReplayError("truncated artifact does not end at max_plies")
        if result.target_utility is not None:
            raise ReplayError("truncation must not have terminal utility")
    else:
        if not outcome.terminal:
            raise ReplayError("completed artifact does not end in a board terminal")
        if (
            result.winner != outcome.winner
            or result.target_utility != outcome.utility
            or result.target_was_checkmated != outcome.target_was_checkmated
            or result.target_won != outcome.target_won
        ):
            raise ReplayError("stored terminal outcome does not match final board")
    return board


__all__ = [
    "MatchConfig",
    "MatchResult",
    "PlyRecord",
    "ProtocolFailure",
    "ReplayError",
    "play_match",
    "replay_match",
]
