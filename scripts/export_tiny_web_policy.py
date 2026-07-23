#!/usr/bin/env python3
"""Export a policy-only int8 checkpoint and deterministic browser parity cases."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from worst_chess.chess.actions import ACTION_PLANES, ACTION_SPACE_SIZE
from worst_chess.chess.neural_actions import (
    ACTION_ORIENTATION_METADATA_KEY,
    PERSPECTIVE_ACTION_ORIENTATION,
    neural_legal_action_mask,
)
from worst_chess.chess.observations import OBSERVATION_SHAPE, encode_observation
from worst_chess.training.model import ACTION_LAYOUT, load_checkpoint

FORMAT = "shallow-red.policy-int8"
FORMAT_VERSION = 1
MAGIC = b"SRPOLICY"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    return parser


def _policy_parameters(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head.")
    ]


def _quantize_policy(
    model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], bytes, torch.nn.Module]:
    quantized_model = copy.deepcopy(model)
    tensors: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0

    with torch.no_grad():
        for name, parameter in _policy_parameters(quantized_model):
            values = parameter.detach().cpu().numpy().astype(np.float32)
            peak = float(np.max(np.abs(values)))
            scale = np.float32(peak / 127.0 if peak else 1.0)
            quantized = np.rint(values / scale).clip(-127, 127).astype(np.int8)
            dequantized = quantized.astype(np.float32) * scale
            parameter.copy_(torch.from_numpy(dequantized))
            payload = quantized.tobytes()
            tensors.append(
                {
                    "length": len(payload),
                    "name": name,
                    "offset": offset,
                    "scale": float(scale),
                    "shape": list(values.shape),
                }
            )
            chunks.append(payload)
            offset += len(payload)

    return tensors, b"".join(chunks), quantized_model


def _build_model_bytes(
    model: torch.nn.Module,
    *,
    metadata: dict[str, Any],
) -> tuple[bytes, torch.nn.Module]:
    orientation = metadata.get(ACTION_ORIENTATION_METADATA_KEY)
    if orientation != PERSPECTIVE_ACTION_ORIENTATION:
        raise ValueError(
            "tiny web policy requires perspective-aligned action coordinates"
        )
    tensors, payload, quantized_model = _quantize_policy(model)
    config = model.config
    header = {
        "actionLayout": ACTION_LAYOUT,
        "actionPlanes": ACTION_PLANES,
        "actionSpaceSize": ACTION_SPACE_SIZE,
        "channels": config.channels,
        "format": FORMAT,
        "observationShape": list(OBSERVATION_SHAPE),
        "orientation": orientation,
        "payloadBytes": len(payload),
        "policyParameters": len(payload),
        "residualBlocks": config.residual_blocks,
        "tensors": tensors,
        "version": FORMAT_VERSION,
    }
    header_bytes = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    model_bytes = (
        MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload
    )
    return model_bytes, quantized_model


def _fixture_boards() -> tuple[tuple[str, chess.Board], ...]:
    cases: list[tuple[str, chess.Board]] = [
        ("starting-white", chess.Board()),
        ("after-e4-black", chess.Board()),
        ("en-passant-white", chess.Board()),
        ("castled-midgame-black", chess.Board()),
        ("repetition-white", chess.Board()),
        (
            "low-material-white",
            chess.Board("8/8/8/3k4/8/4K3/4P3/7r w - - 12 70"),
        ),
        (
            "promotion-white",
            chess.Board("8/P7/8/3k4/8/4K3/8/7r w - - 0 1"),
        ),
        (
            "promotion-black",
            chess.Board("7R/8/4k3/8/3K4/8/p7/8 b - - 0 1"),
        ),
    ]
    for move in ("e2e4",):
        cases[1][1].push_uci(move)
    for move in ("e2e4", "c7c5", "e4e5", "d7d5"):
        cases[2][1].push_uci(move)
    for move in (
        "d2d4",
        "g8f6",
        "c2c4",
        "e7e6",
        "b1c3",
        "f8b4",
        "e2e3",
        "e8g8",
        "f1d3",
        "d7d5",
        "g1f3",
        "c7c5",
        "e1g1",
    ):
        cases[3][1].push_uci(move)
    for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        cases[4][1].push_uci(move)
    return tuple(cases)


def _float32_base64(values: np.ndarray) -> str:
    raw = np.asarray(values, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _build_fixtures(
    model: torch.nn.Module,
    *,
    model_bytes: bytes,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for name, board in _fixture_boards():
            perspective = board.turn
            observation = encode_observation(board, perspective)
            logits, _ = model(torch.from_numpy(observation).unsqueeze(0))
            row = logits[0].detach().cpu().numpy().astype(np.float32)
            legal = np.flatnonzero(
                neural_legal_action_mask(
                    board,
                    PERSPECTIVE_ACTION_ORIENTATION,
                )
            ).tolist()
            top = sorted(legal, key=lambda action: (-float(row[action]), action))[:12]
            cases.append(
                {
                    "expectedLogitsF32Base64": _float32_base64(row),
                    "fen": board.fen(),
                    "initialFen": board.root().fen(),
                    "legalActions": legal,
                    "movesUci": [move.uci() for move in board.move_stack],
                    "name": name,
                    "observationF32Base64": _float32_base64(observation),
                    "targetColor": "white" if perspective == chess.WHITE else "black",
                    "top12Actions": top,
                }
            )
    return {
        "cases": cases,
        "format": "shallow-red.policy-parity",
        "modelSha256": hashlib.sha256(model_bytes).hexdigest(),
        "version": 1,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    model, metadata = load_checkpoint(arguments.checkpoint, device="cpu")
    model_bytes, quantized_model = _build_model_bytes(model, metadata=metadata)
    fixtures = _build_fixtures(quantized_model, model_bytes=model_bytes)

    arguments.model_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.fixture_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.model_output.write_bytes(model_bytes)
    arguments.fixture_output.write_text(
        json.dumps(fixtures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"policy parameters: {sum(p.numel() for _, p in _policy_parameters(model))}")
    print(f"model bytes: {len(model_bytes)}")
    print(f"model: {arguments.model_output}")
    print(f"fixtures: {arguments.fixture_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
