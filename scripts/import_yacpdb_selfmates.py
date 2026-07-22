"""Import a conservative research sample of orthodox YACPDB selfmates.

YACPDB does not currently advertise a bulk-data license. This importer therefore
fetches attributed records into the gitignored ``artifacts`` tree; it does not
vendor database contents or treat published solutions as training labels.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import chess

DEFAULT_QUERY = 'Stip("s#[1-4]") AND NOT Fairy'
GATEWAY_URL = "https://www.yacpdb.org/gateway/ql"
_STIPULATION = re.compile(r"^s#([1-4])$")
_PIECE = re.compile(r"^([KQRSBP])([a-h][1-8])$")
_PIECE_TYPES = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "S": chess.KNIGHT,
    "P": chess.PAWN,
}
_BAD_ANNOTATIONS = ("cook", "no solution", "unsound")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def diagram_to_board(algebraic: object) -> chess.Board:
    """Convert a standard YACPDB algebraic diagram to an orthodox board."""

    if not isinstance(algebraic, dict):
        raise ValueError("algebraic diagram must be an object")
    board = chess.Board.empty()
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        tokens = algebraic.get(color_name)
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError(f"{color_name} pieces must be a string list")
        for token in tokens:
            match = _PIECE.fullmatch(token)
            if match is None:
                raise ValueError(f"unsupported piece token: {token}")
            piece_code, square_name = match.groups()
            square = chess.parse_square(square_name)
            if board.piece_at(square) is not None:
                raise ValueError(f"duplicate occupied square: {square_name}")
            board.set_piece_at(square, chess.Piece(_PIECE_TYPES[piece_code], color))

    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    board.clear_stack()
    if not board.is_valid():
        raise ValueError(f"invalid orthodox position: {board.status()}")
    return board


def normalize_entry(entry: object) -> dict[str, Any] | None:
    """Return one conservative training candidate, or reject unsupported data."""

    if not isinstance(entry, dict):
        return None
    stipulation = entry.get("stipulation")
    if not isinstance(stipulation, str):
        return None
    stipulation_match = _STIPULATION.fullmatch(stipulation.strip())
    if stipulation_match is None:
        return None
    if entry.get("twins") or entry.get("options"):
        return None

    annotation_parts: list[str] = []
    for key in ("comments", "keywords"):
        value = entry.get(key, [])
        if isinstance(value, list):
            annotation_parts.extend(str(item).lower() for item in value)
    annotations = " ".join(annotation_parts)
    if any(marker in annotations for marker in _BAD_ANNOTATIONS):
        return None

    try:
        board = diagram_to_board(entry.get("algebraic"))
    except ValueError:
        return None

    entry_id = entry.get("id")
    if not isinstance(entry_id, int):
        return None
    source = entry.get("source")
    authors = entry.get("authors", [])
    return {
        "fen": board.fen(en_passant="fen"),
        "target_color": "white",
        "source": "yacpdb",
        "source_id": entry_id,
        "source_url": f"https://www.yacpdb.org/#id/{entry_id}",
        "authors": authors if isinstance(authors, list) else [],
        "publication": source if isinstance(source, dict) else {},
        "stipulation": stipulation.strip(),
        "selfmate_moves": int(stipulation_match.group(1)),
    }


def fetch_page(
    query: str,
    page: int,
    *,
    timeout_seconds: float,
) -> tuple[int, list[dict[str, Any]]]:
    params = urllib.parse.urlencode({"q": query, "p": page})
    request = urllib.request.Request(
        f"{GATEWAY_URL}?{params}",
        headers={"User-Agent": "Shallow-Red-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError("YACPDB query failed")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("YACPDB response has no result object")
    count = result.get("count")
    entries = result.get("entries")
    if not isinstance(count, int) or not isinstance(entries, list):
        raise RuntimeError("YACPDB result has an unexpected shape")
    return count, [entry for entry in entries if isinstance(entry, dict)]


def import_pages(
    *,
    query: str,
    start_page: int,
    pages: int,
    limit: int | None,
    delay_seconds: float,
    timeout_seconds: float,
) -> tuple[int, list[dict[str, Any]], int]:
    if start_page < 1 or pages < 1:
        raise ValueError("start_page and pages must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if delay_seconds < 0 or timeout_seconds <= 0:
        raise ValueError("delay must be nonnegative and timeout must be positive")

    imported: list[dict[str, Any]] = []
    rejected = 0
    total_matches = 0
    for page_offset in range(pages):
        if limit is not None and len(imported) >= limit:
            break
        total_matches, entries = fetch_page(
            query,
            start_page + page_offset,
            timeout_seconds=timeout_seconds,
        )
        for entry in entries:
            normalized = normalize_entry(entry)
            if normalized is None:
                rejected += 1
                continue
            imported.append(normalized)
            if limit is not None and len(imported) >= limit:
                break
        if page_offset + 1 < pages and delay_seconds:
            time.sleep(delay_seconds)
    return total_matches, imported, rejected


def main() -> int:
    arguments = build_parser().parse_args()
    total_matches, imported, rejected = import_pages(
        query=arguments.query,
        start_page=arguments.start_page,
        pages=arguments.pages,
        limit=arguments.limit,
        delay_seconds=arguments.delay_seconds,
        timeout_seconds=arguments.timeout_seconds,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in imported),
        encoding="utf-8",
    )
    print(f"query matches: {total_matches}")
    print(f"imported: {len(imported)}")
    print(f"rejected: {rejected}")
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
