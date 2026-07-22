"""Project or exactly solve the closed KBvKR forced-selfmate graph."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from worst_chess.objective.four_piece import (
    print_progress,
    project_kbvkr,
    solve_kbvkr,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project or exactly solve KBvKR with White as selfmate target."
    )
    parser.add_argument("material", choices=("KBvKR",))
    parser.add_argument("--mode", choices=("project", "solve"), default="project")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--maximum-ram-gib",
        type=float,
        default=16.0,
        help="refuse solve when the projection exceeds this amount",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    projection = project_kbvkr(
        sample_size=arguments.sample_size,
        seed=arguments.seed,
    )
    output: dict[str, object] = {
        "material": arguments.material,
        "target": "white",
        "projection": asdict(projection),
        "projected_peak_ram_gib": (
            projection.projected_peak_ram_bytes / (1024**3)
        ),
        "rule_scope": {
            "stalemate": "terminal non-success",
            "dead_position": "terminal non-success",
            "cycles": "non-forced draw/failure",
            "omitted": [
                "50-move optional claim",
                "75-move automatic draw",
                "threefold optional claim",
                "fivefold automatic repetition",
            ],
            "capture_closure": (
                "KBvK and KvKR link to exact three-piece results; only an "
                "immediate target checkmate is successful"
            ),
        },
    }
    if arguments.mode == "solve":
        maximum_bytes = arguments.maximum_ram_gib * 1024**3
        if projection.projected_peak_ram_bytes > maximum_bytes:
            raise MemoryError(
                "projected peak RAM exceeds --maximum-ram-gib; refusing solve"
            )
        result = solve_kbvkr(
            maximum_ram_bytes=round(maximum_bytes),
            progress=print_progress,
        )
        finite_plies = tuple(
            plies for plies in result.solution.plies if plies is not None
        )
        output["solution"] = {
            "legal_symmetry_reduced_states": result.state_count,
            "edges": result.edge_count,
            "target_checkmate_terminals": result.successful_terminals,
            "forced_selfmate_states_including_terminals": (
                result.solution.forced_count
            ),
            "forced_selfmate_nonterminal_states": (
                result.forced_nonterminal_count
            ),
            "maximum_forced_selfmate_plies": (
                max(finite_plies) if finite_plies else None
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
