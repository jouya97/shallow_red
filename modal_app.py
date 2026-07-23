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
proof_ranked_image = ml_image.add_local_file(
    PROJECT_ROOT / "scripts" / "build_proof_ranked_dataset.py",
    "/root/build_proof_ranked_dataset.py",
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
    image=ml_image,
    cpu=1.0,
    memory=2_048,
    timeout=2 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_game_cpu(arguments: list[str]) -> int:
    """Run an independently shardable neural gameplay batch."""

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
    image=proof_ranked_image,
    cpu=1.0,
    memory=2_048,
    timeout=6 * 60 * 60,
    volumes={ARTIFACTS_MOUNT: artifacts},
)
def run_proof_ranked(arguments: list[str]) -> int:
    """Build all-legal-move ranked labels from proven selfmates."""

    completed = subprocess.run(
        [sys.executable, "/root/build_proof_ranked_dataset.py", *arguments],
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
    if mode == "proof-ranked":
        total_positions = 138
        shard_size = 10
        label_commands = [
            [
                "--proof-report",
                "/artifacts/evaluations/yacpdb-retro-modal/merged-report.json",
                "--start",
                str(start),
                "--count",
                str(min(shard_size, total_positions - start)),
                "--node-budget",
                "100000",
                "--output",
                f"/artifacts/datasets/proof-ranked-shard-{start:03d}.jsonl",
                "--report",
                f"/artifacts/evaluations/proof-ranked-shard-{start:03d}/report.json",
            ]
            for start in range(0, total_positions, shard_size)
        ]
        results = list(
            run_proof_ranked.map(
                label_commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} proof-ranked shards failed")
        return
    if mode == "proof-finetune":
        base_train = "/artifacts/datasets/finetune-splits/base-train.jsonl"
        proof_train = "/artifacts/datasets/finetune-splits/proof-train.jsonl"
        initialization = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        commands: list[list[str]] = []
        for proof_weight in (1, 4, 8):
            commands.append(
                [
                    "train-ranked",
                    "--dataset",
                    base_train,
                    *([proof_train] * proof_weight),
                    "--validation-dataset",
                    "/artifacts/datasets/finetune-splits/base-validation.jsonl",
                    "/artifacts/datasets/finetune-splits/proof-validation.jsonl",
                    "--test-dataset",
                    "/artifacts/datasets/finetune-splits/base-test.jsonl",
                    "/artifacts/datasets/finetune-splits/proof-test.jsonl",
                    "--initialize-from",
                    initialization,
                    "--checkpoint",
                    (
                        "/artifacts/checkpoints/"
                        f"ranked-v06-proof-w{proof_weight}-seed-20264021.pt"
                    ),
                    "--epochs",
                    "5",
                    "--batch-size",
                    "128",
                    "--learning-rate",
                    "0.0001",
                    "--rank-temperature",
                    "2",
                    "--value-loss-weight",
                    "0",
                    "--seed",
                    "20264021",
                    "--device",
                    "cuda",
                    "--channels",
                    "32",
                    "--residual-blocks",
                    "4",
                    "--perspective-actions",
                ]
            )
        results = list(
            run_gpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} proof-finetune jobs failed")
        return
    if mode == "proof-finetune-sharp":
        base_train = "/artifacts/datasets/finetune-splits/base-train.jsonl"
        proof_train = "/artifacts/datasets/finetune-splits/proof-train.jsonl"
        initialization = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        commands = []
        for proof_weight in (4, 8):
            commands.append(
                [
                    "train-ranked",
                    "--dataset",
                    base_train,
                    *([proof_train] * proof_weight),
                    "--validation-dataset",
                    "/artifacts/datasets/finetune-splits/base-validation.jsonl",
                    "/artifacts/datasets/finetune-splits/proof-validation.jsonl",
                    "--test-dataset",
                    "/artifacts/datasets/finetune-splits/base-test.jsonl",
                    "/artifacts/datasets/finetune-splits/proof-test.jsonl",
                    "--initialize-from",
                    initialization,
                    "--checkpoint",
                    (
                        "/artifacts/checkpoints/"
                        f"ranked-v06s-proof-w{proof_weight}-seed-20264022.pt"
                    ),
                    "--epochs",
                    "10",
                    "--batch-size",
                    "128",
                    "--learning-rate",
                    "0.0001",
                    "--rank-temperature",
                    "0.25",
                    "--value-loss-weight",
                    "0",
                    "--seed",
                    "20264022",
                    "--device",
                    "cuda",
                    "--channels",
                    "32",
                    "--residual-blocks",
                    "4",
                    "--perspective-actions",
                ]
            )
        results = list(
            run_gpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} sharp fine-tune jobs failed")
        return
    if mode == "proof-finetune-safety":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v06s-proof-w8-seed-20264022.pt"
        )
        commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--opponent",
                "random",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                "100",
                "--openings",
                "100",
                "--opening-plies",
                "6",
                "--max-plies",
                "600",
                "--seed",
                "20261221",
                "--output",
                "/artifacts/evaluations/proof-v06s-w8-random-200",
            ],
            *[
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
                    f"/artifacts/evaluations/proof-v06s-w8-{opponent}-20",
                ]
                for opponent in ("selfish-random-reply", "selfish-portfolio")
            ],
        ]
        results = list(
            run_cpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} proof safety jobs failed")
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
    if mode == "synthetic-loser-league":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        total_pairs = 50
        shard_pairs = 5
        commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--opponent",
                "synthetic-loser-league",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                str(shard_pairs),
                "--openings",
                "1",
                "--opening-plies",
                "0",
                "--max-plies",
                "600",
                "--seed",
                str(20265020 + pair_start),
                "--output",
                f"/artifacts/evaluations/synthetic-league-shard-{pair_start:03d}",
            ]
            for pair_start in range(0, total_pairs, shard_pairs)
        ]
        results = list(
            run_game_cpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"{len(failures)} synthetic league jobs failed")
        return
    if mode == "synthetic-loser-exploration":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--target-exploration",
                "0.20",
                "--opponent",
                "synthetic-loser-league",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                "5",
                "--openings",
                "1",
                "--opening-plies",
                "0",
                "--max-plies",
                "600",
                "--seed",
                str(20265100 + pair_start),
                "--output",
                (
                    "/artifacts/evaluations/"
                    f"synthetic-exploration-shard-{pair_start:03d}"
                ),
            ]
            for pair_start in range(0, 25, 5)
        ]
        results = list(
            run_game_cpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} synthetic exploration jobs failed"
            )
        return
    if mode == "synthetic-exploration-sweep":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--target-exploration",
                str(probability),
                "--opponent",
                "synthetic-loser-league",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                "5",
                "--openings",
                "1",
                "--opening-plies",
                "0",
                "--max-plies",
                "600",
                "--seed",
                str(20265200 + probability_index * 100 + shard),
                "--output",
                (
                    "/artifacts/evaluations/"
                    f"synthetic-sweep-p{probability_index:02d}-shard-{shard:03d}"
                ),
            ]
            for probability_index, probability in enumerate((0.10, 0.35, 0.50))
            for shard in (0, 5)
        ]
        results = list(
            run_game_cpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} synthetic sweep jobs failed"
            )
        return
    if mode == "synthetic-exploration-scale":
        checkpoint = (
            "/artifacts/checkpoints/"
            "ranked-v03-perspective-random-seed-20261021.pt"
        )
        total_pairs = 100
        shard_pairs = 5
        commands = [
            [
                "smoke",
                "--target",
                "stalemate-aware",
                "--target-exploration",
                "0.20",
                "--opponent",
                "synthetic-loser-league",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
                "--search-top-k",
                "12",
                "--pairs",
                str(shard_pairs),
                "--openings",
                "1",
                "--opening-plies",
                "0",
                "--max-plies",
                "600",
                "--seed",
                str(20266000 + pair_start),
                "--output",
                (
                    "/artifacts/evaluations/"
                    f"synthetic-scale-shard-{pair_start:03d}"
                ),
            ]
            for pair_start in range(0, total_pairs, shard_pairs)
        ]
        results = list(
            run_game_cpu.map(
                commands,
                order_outputs=False,
                return_exceptions=True,
            )
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} synthetic scale jobs failed"
            )
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
        "proof-candidates, retro-ancestors, retro-ancestors-v2, proof-ranked, "
        "proof-finetune, proof-finetune-sharp, proof-finetune-safety, "
        "selfish-pilot, synthetic-loser-league, synthetic-loser-exploration, "
        "synthetic-exploration-sweep, synthetic-exploration-scale, highmem, "
        "retrograde, four-piece-retrograde, gpu"
    )
