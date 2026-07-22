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
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_MOUNT = Path("/artifacts")

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
    ARTIFACTS_MOUNT.mkdir(parents=True, exist_ok=True)
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
        "mode must be one of: smoke, cpu, highmem, retrograde, "
        "four-piece-retrograde, gpu"
    )
