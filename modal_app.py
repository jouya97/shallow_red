"""Modal entrypoints for cloud evaluation and training workloads.

Run the zero-GPU connectivity check with::

    uv run --extra cloud modal run modal_app.py --mode smoke

CPU and GPU jobs execute the existing ``worst-chess`` CLI and mount the
``shallow-red-artifacts`` Volume at ``/artifacts``. Pass all data and output
paths beneath that mount so checkpoints survive container shutdown.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_MOUNT = PurePosixPath("/artifacts")

app = modal.App("shallow-red")
artifacts = modal.Volume.from_name(
    "shallow-red-artifacts",
    create_if_missing=True,
)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        uv_project_dir=str(PROJECT_ROOT),
        frozen=True,
    )
    .add_local_dir(PROJECT_ROOT / "src", "/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)
retrograde_image = base_image.add_local_file(
    PROJECT_ROOT / "scripts" / "three_piece_retrograde.py",
    "/root/three_piece_retrograde.py",
    copy=True,
)
four_piece_retrograde_image = base_image.add_local_file(
    PROJECT_ROOT / "scripts" / "four_piece_retrograde.py",
    "/root/four_piece_retrograde.py",
    copy=True,
)
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .uv_sync(
        uv_project_dir=str(PROJECT_ROOT),
        extras=["ml"],
        frozen=True,
    )
    .add_local_dir(PROJECT_ROOT / "src", "/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)
web_eval_image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.11")
    .uv_sync(
        uv_project_dir=str(PROJECT_ROOT),
        frozen=True,
    )
    .add_local_dir(PROJECT_ROOT / "src", "/root/src", copy=True)
    .add_local_file(
        PROJECT_ROOT / "web" / "evaluation" / "package.json",
        "/root/web/package.json",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "web" / "evaluation" / "package-lock.json",
        "/root/web/package-lock.json",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "web" / "lib" / "shallow-red.ts",
        "/root/web/lib/shallow-red.ts",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "web" / "lib" / "engine-protocol.ts",
        "/root/web/lib/engine-protocol.ts",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "web" / "scripts" / "engine-jsonl.ts",
        "/root/web/scripts/engine-jsonl.ts",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "scripts" / "evaluate_web_frozen.py",
        "/root/scripts/evaluate_web_frozen.py",
        copy=True,
    )
    .run_commands("cd /root/web && npm ci --ignore-scripts --no-audit --no-fund")
    .env({"PYTHONPATH": "/root/src"})
)
proof_search_image = base_image.add_local_file(
    PROJECT_ROOT / "scripts" / "mine_forced_selfmate.py",
    "/root/mine_forced_selfmate.py",
    copy=True,
)
retro_expand_image = base_image.add_local_file(
    PROJECT_ROOT / "scripts" / "expand_selfmate_ancestors.py",
    "/root/expand_selfmate_ancestors.py",
    copy=True,
)


@app.function(image=base_image, cpu=1.0, memory=512, timeout=300)
def smoke() -> dict[str, str]:
    """Verify that the locked project imports inside a Modal container."""

    import platform

    import worst_chess

    return {
        "status": "ok",
        "architecture": platform.machine(),
        "worst_chess_version": worst_chess.__version__,
    }


def _run_cli(arguments: list[str]) -> int:
    if not arguments:
        raise ValueError("a worst-chess subcommand is required")
    Path(ARTIFACTS_MOUNT).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "worst_chess.cli", *arguments],
        cwd=ARTIFACTS_MOUNT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    artifacts.commit()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=ml_image,
    cpu=4.0,
    memory=8_192,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_cpu(arguments: list[str]) -> int:
    """Run a CPU-heavy rollout or evaluation command."""

    return _run_cli(arguments)


@app.function(
    image=web_eval_image,
    cpu=1.0,
    memory=2_048,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_web_eval(arguments: list[str]) -> int:
    """Evaluate the exact TypeScript browser policy with the Python harness."""

    completed = subprocess.run(
        [
            sys.executable,
            "/root/scripts/evaluate_web_frozen.py",
            "--web-directory",
            "/root/web",
            *arguments,
        ],
        cwd=ARTIFACTS_MOUNT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    artifacts.commit()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=proof_search_image,
    cpu=1.0,
    memory=2_048,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_proof_search(arguments: list[str]) -> int:
    """Mine bounded forced-selfmate proofs on CPU."""

    completed = subprocess.run(
        [sys.executable, "/root/mine_forced_selfmate.py", *arguments],
        cwd=ARTIFACTS_MOUNT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    artifacts.commit()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=retro_expand_image,
    cpu=1.0,
    memory=2_048,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_retro_expand(arguments: list[str]) -> int:
    """Validate quiet two-ply ancestors of proven selfmates on CPU."""

    completed = subprocess.run(
        [sys.executable, "/root/expand_selfmate_ancestors.py", *arguments],
        cwd=ARTIFACTS_MOUNT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    artifacts.commit()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=base_image,
    cpu=8.0,
    memory=16_384,
    timeout=2 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_highmem(arguments: list[str]) -> int:
    """Run graph enumeration or retrograde analysis without a GPU."""

    return _run_cli(arguments)


@app.function(
    image=retrograde_image,
    cpu=8.0,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def run_retrograde(arguments: list[str]) -> int:
    """Run the exact three-piece retrograde script on high-memory CPU."""

    completed = subprocess.run(
        [sys.executable, "/root/three_piece_retrograde.py", *arguments],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=four_piece_retrograde_image,
    cpu=8.0,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def run_four_piece_retrograde(arguments: list[str]) -> int:
    """Run the projection-gated exact four-piece retrograde pilot."""

    completed = subprocess.run(
        [sys.executable, "/root/four_piece_retrograde.py", *arguments],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


@app.function(
    image=ml_image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_gpu(arguments: list[str]) -> int:
    """Run a single-L4 training command."""

    return _run_cli(arguments)


@app.local_entrypoint()
def main(mode: str = "smoke", command: str = "") -> None:
    """Dispatch a smoke, CPU, or GPU job from the Modal CLI."""

    if mode == "smoke":
        print(smoke.remote())
        return

    arguments = shlex.split(command)
    if mode == "cpu":
        run_cpu.remote(arguments)
        return
    if mode == "web-eval":
        run_web_eval.remote(arguments)
        return
    if mode == "web-frozen":
        shard_commands: list[list[str]] = []
        for label, pairs, seed in (
            ("primary", 100, 20261221),
            ("safety", 50, 20261321),
        ):
            for pair_start in range(0, pairs, 10):
                shard_commands.append(
                    [
                        "--pairs",
                        str(pairs),
                        "--pair-start",
                        str(pair_start),
                        "--pair-count",
                        str(min(10, pairs - pair_start)),
                        "--seed",
                        str(seed),
                        "--opening-plies",
                        "6",
                        "--max-plies",
                        "600",
                        "--tournament-id",
                        "smoke-stalemate-aware",
                        "--output",
                        f"evaluations/web-frozen-{label}-shard-{pair_start:03d}",
                    ]
                )
        results = list(
            run_web_eval.map(
                shard_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} web-evaluation shards failed")
        return
    if mode == "proof-search":
        run_proof_search.remote(arguments)
        return
    if mode == "proof-candidates":
        proof_commands = [
            [
                "search",
                "--input",
                "/artifacts/datasets/forced-selfmate-candidates-v1.jsonl",
                "--start",
                str(start),
                "--count",
                str(min(100, 1_132 - start)),
                "--max-plies",
                "2",
                "4",
                "--node-budget",
                "20000",
                "--output",
                f"/artifacts/evaluations/forced-proof-v1-shard-{start:04d}/report.json",
            ]
            for start in range(0, 1_132, 100)
        ]
        results = list(
            run_proof_search.map(
                proof_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} proof-search shards failed")
        return
    if mode == "retro-ancestors":
        total_seeds = 84
        shard_size = 7
        retro_commands = [
            [
                "--proof-report",
                "/artifacts/evaluations/yacpdb-pilot-proof/report.json",
                "--start",
                str(start),
                "--count",
                str(min(shard_size, total_seeds - start)),
                "--max-candidates-per-seed",
                "50",
                "--max-extended-per-seed",
                "5",
                "--node-budget",
                "100000",
                "--output",
                f"/artifacts/evaluations/yacpdb-retro-shard-{start:03d}/report.json",
            ]
            for start in range(0, total_seeds, shard_size)
        ]
        results = list(
            run_retro_expand.map(
                retro_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} retro-expansion shards failed")
        return
    if mode == "retro-ancestors-v2":
        total_seeds = 20
        shard_size = 5
        retro_commands = [
            [
                "--proof-report",
                "/artifacts/evaluations/yacpdb-retro-modal/merged-report.json",
                "--start",
                str(start),
                "--count",
                str(min(shard_size, total_seeds - start)),
                "--max-candidates-per-seed",
                "20",
                "--max-extended-per-seed",
                "3",
                "--node-budget",
                "100000",
                "--output",
                f"/artifacts/evaluations/yacpdb-retro-v2-shard-{start:03d}/report.json",
            ]
            for start in range(0, total_seeds, shard_size)
        ]
        results = list(
            run_retro_expand.map(
                retro_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} v2 retro-expansion shards failed")
        return
    if mode == "selfish-pilot":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        population_commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--opponent",
                opponent,
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                "10",
                "--openings",
                "10",
                "--opening-plies",
                "6",
                "--max-plies",
                "300",
                "--seed",
                "20263021",
                "--output",
                f"/artifacts/evaluations/{opponent}-pilot-20g",
            ]
            for opponent in ("selfish-random-reply", "selfish-portfolio")
        ]
        results = list(
            run_cpu.map(
                population_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} selfish-pilot jobs failed")
        return
    if mode == "highmem":
        run_highmem.remote(arguments)
        return
    if mode == "retrograde":
        run_retrograde.remote(arguments)
        return
    if mode == "four-piece-retrograde":
        run_four_piece_retrograde.remote(arguments)
        return
    if mode == "gpu":
        run_gpu.remote(arguments)
        return
    raise ValueError(
        "mode must be one of: smoke, cpu, web-eval, web-frozen, proof-search, "
        "proof-candidates, retro-ancestors, retro-ancestors-v2, selfish-pilot, "
        "highmem, retrograde, four-piece-retrograde, gpu"
    )
