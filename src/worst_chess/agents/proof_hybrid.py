"""Exact selfmate-book and bounded proof-search overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from worst_chess.agents.base import Agent, AgentError, MoveContext
from worst_chess.objective.proof_search import (
    ProofSearchConfig,
    ProofStatus,
    prove_forced_selfmate,
)


@dataclass(frozen=True, slots=True)
class SelfmateBookEntry:
    """One shortest known move from an independently proven position."""

    move: chess.Move
    forced_plies: int
    source_id: str


@dataclass(frozen=True, slots=True)
class ProofHybridStats:
    """Decision counters for auditing whether the hybrid changes play."""

    book_hits: int
    search_hits: int
    search_refutations: int
    search_unknowns: int
    fallbacks: int
    search_nodes: int
    decision_cache_hits: int


class SelfmateProofBook:
    """Exact-position lookup built from auditable proof-search reports.

    The book is deliberately not a fuzzy opening book. A move is returned only
    when the board's rule-relevant FEN fields exactly match a proven record.
    Color-mirrored records are safe because :meth:`chess.Board.mirror` swaps
    colors, piece placement, turn, castling rights, and en-passant state.
    """

    def __init__(self, entries: dict[str, SelfmateBookEntry] | None = None) -> None:
        self._entries = dict(entries or {})

    @classmethod
    def from_reports(
        cls,
        paths: tuple[str | Path, ...],
        *,
        include_color_mirrors: bool = True,
    ) -> SelfmateProofBook:
        entries: dict[str, SelfmateBookEntry] = {}
        for path in paths:
            source = Path(path)
            try:
                raw: object = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"could not read proof report {source}: {error}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(f"proof report {source} must be a JSON object")
            records = raw.get("records")
            if not isinstance(records, list):
                raise ValueError(f"proof report {source} must contain a records list")
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"proof report {source} record {index} must be an object"
                    )
                if record.get("status") != ProofStatus.PROVEN.value:
                    continue
                board, entry, variation = _validated_record(source, index, record)
                _store_shortest(entries, board, entry)
                if include_color_mirrors:
                    mirrored = board.mirror()
                    mirrored_variation = tuple(_mirror_move(move) for move in variation)
                    mirrored_entry = SelfmateBookEntry(
                        move=mirrored_variation[0],
                        forced_plies=entry.forced_plies,
                        source_id=f"{entry.source_id}/color-mirror",
                    )
                    _validate_variation(
                        mirrored,
                        mirrored_variation,
                        target_color=mirrored.turn,
                        label=f"{source} record {index} color mirror",
                    )
                    _store_shortest(entries, mirrored, mirrored_entry)
        return cls(entries)

    def lookup(self, board: chess.Board) -> SelfmateBookEntry | None:
        """Return an exact proven entry without mutating ``board``."""

        return self._entries.get(_position_key(board))

    def __len__(self) -> int:
        return len(self._entries)


class ProofGuidedSelfmateAgent:
    """Use exact proofs first, bounded live proof search second, then fallback.

    Only ``PROVEN`` live-search results can override the fallback. A completed
    refutation or an exhausted node budget therefore cannot turn an uncertain
    tactical guess into a supposedly forced selfmate.
    """

    def __init__(
        self,
        fallback: Agent,
        *,
        book: SelfmateProofBook | None = None,
        search_config: ProofSearchConfig | None = None,
    ) -> None:
        self.fallback = fallback
        self.book = book or SelfmateProofBook()
        self.search_config = search_config
        self._decision_cache: dict[tuple[str, chess.Color], chess.Move | None] = {}
        self._book_hits = 0
        self._search_hits = 0
        self._search_refutations = 0
        self._search_unknowns = 0
        self._fallbacks = 0
        self._search_nodes = 0
        self._decision_cache_hits = 0

    @property
    def name(self) -> str:
        search = (
            "no_live_search"
            if self.search_config is None
            else (
                f"search_{self.search_config.max_plies}_plies_"
                f"{self.search_config.node_budget}_nodes"
            )
        )
        return (
            f"proof_guided_book_{len(self.book)}_{search}_"
            f"with_{self.fallback.name}"
        )

    @property
    def stats(self) -> ProofHybridStats:
        return ProofHybridStats(
            book_hits=self._book_hits,
            search_hits=self._search_hits,
            search_refutations=self._search_refutations,
            search_unknowns=self._search_unknowns,
            fallbacks=self._fallbacks,
            search_nodes=self._search_nodes,
            decision_cache_hits=self._decision_cache_hits,
        )

    def select_move(self, board: chess.Board, context: MoveContext) -> chess.Move:
        if board.turn != context.target_color:
            raise AgentError("ProofGuidedSelfmateAgent must act for the target color")
        if board.is_game_over(claim_draw=False):
            raise AgentError(
                "ProofGuidedSelfmateAgent cannot move from a terminal position"
            )

        entry = self.book.lookup(board)
        if entry is not None:
            if entry.move not in board.legal_moves:
                raise AgentError(
                    f"proof book returned illegal move {entry.move.uci()}"
                )
            self._book_hits += 1
            return entry.move

        cache_key = (_position_key(board), context.target_color)
        if cache_key in self._decision_cache:
            self._decision_cache_hits += 1
            cached = self._decision_cache[cache_key]
            if cached is not None:
                self._search_hits += 1
                return cached
            return self._fallback_move(board, context)

        if self.search_config is not None:
            result = prove_forced_selfmate(
                board,
                context.target_color,
                self.search_config,
            )
            self._search_nodes += result.nodes
            if result.status is ProofStatus.PROVEN:
                move = result.first_move
                if move is None or move not in board.legal_moves:
                    raise AgentError(
                        "proven live selfmate search returned no legal move"
                    )
                self._decision_cache[cache_key] = move
                self._search_hits += 1
                return move
            if result.status is ProofStatus.REFUTED:
                self._search_refutations += 1
            else:
                self._search_unknowns += 1

        self._decision_cache[cache_key] = None
        return self._fallback_move(board, context)

    def _fallback_move(
        self,
        board: chess.Board,
        context: MoveContext,
    ) -> chess.Move:
        move = self.fallback.select_move(board.copy(stack=True), context)
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            detail = move.uci() if isinstance(move, chess.Move) else repr(move)
            raise AgentError(f"proof hybrid fallback returned illegal move {detail}")
        self._fallbacks += 1
        return move


def _validated_record(
    source: Path,
    index: int,
    record: dict[str, Any],
) -> tuple[chess.Board, SelfmateBookEntry, tuple[chess.Move, ...]]:
    label = f"{source} record {index}"
    fen = record.get("fen")
    target_name = record.get("target_color")
    forced_plies = record.get("forced_plies")
    raw_variation = record.get("principal_variation")
    if not isinstance(fen, str):
        raise ValueError(f"{label} must contain a FEN string")
    if target_name not in {"white", "black"}:
        raise ValueError(f"{label} must contain target_color white or black")
    if type(forced_plies) is not int or forced_plies < 1:
        raise ValueError(f"{label} must contain positive forced_plies")
    if not isinstance(raw_variation, list) or not all(
        isinstance(move, str) for move in raw_variation
    ):
        raise ValueError(f"{label} must contain a UCI principal_variation list")
    if len(raw_variation) != forced_plies:
        raise ValueError(f"{label} principal variation length must equal forced_plies")
    try:
        board = chess.Board(fen)
        variation = tuple(chess.Move.from_uci(move) for move in raw_variation)
    except ValueError as error:
        raise ValueError(f"{label} contains invalid chess data: {error}") from error
    if not board.is_valid():
        raise ValueError(f"{label} contains an invalid orthodox position")
    target_color = target_name == "white"
    if board.turn != target_color:
        raise ValueError(f"{label} target must be the side to move")
    _validate_variation(board, variation, target_color=target_color, label=label)
    source_id = str(record.get("root_source_id", record.get("source_id", index)))
    return (
        board,
        SelfmateBookEntry(
            move=variation[0],
            forced_plies=forced_plies,
            source_id=source_id,
        ),
        variation,
    )


def _validate_variation(
    board: chess.Board,
    variation: tuple[chess.Move, ...],
    *,
    target_color: chess.Color,
    label: str,
) -> None:
    position = board.copy(stack=False)
    for move in variation:
        if move not in position.legal_moves:
            raise ValueError(
                f"{label} principal variation has illegal move {move.uci()}"
            )
        position.push(move)
    if not position.is_checkmate() or position.turn != target_color:
        raise ValueError(f"{label} principal variation does not selfmate the target")


def _store_shortest(
    entries: dict[str, SelfmateBookEntry],
    board: chess.Board,
    candidate: SelfmateBookEntry,
) -> None:
    key = _position_key(board)
    current = entries.get(key)
    if current is None or (
        candidate.forced_plies,
        candidate.move.uci(),
        candidate.source_id,
    ) < (
        current.forced_plies,
        current.move.uci(),
        current.source_id,
    ):
        entries[key] = candidate


def _position_key(board: chess.Board) -> str:
    return " ".join(board.fen(en_passant="legal").split()[:5])


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


__all__ = [
    "ProofGuidedSelfmateAgent",
    "ProofHybridStats",
    "SelfmateBookEntry",
    "SelfmateProofBook",
]
