"""Run the Hooke pair GPU preflight config and measure per-parameter weight drift.

Spawns one training run (same mechanism as ``plotting.py``), then reads the
per-step checkpoints under ``<run_dir>/checkpoints/step_*.pt`` and computes,
for every named model parameter, the L2 distance from its step-0 value:

    drift_t = ||theta_t - theta_0||_2
    relative drift_t = drift_t / (||theta_0||_2 + eps)

Prints a per-parameter summary (sorted by final relative drift, flagging
parameters that never moved) and saves drift curves to
``experiments/hooke/preflight/plots/<run_id>_weight_drift.png``.

Requires ``checkpoint.every_n_steps: 1`` (the preflight default) for a
per-step trajectory; sparser checkpoints still work but sample the curve.

Usage
-----
Run from the repository root inside the GPU environment::

    export UV_PROJECT_ENVIRONMENT=.venv-gpu
    uv run --extra cu130 python experiments/hooke/preflight/weight_drift.py

Extra OmegaConf dotlist overrides are forwarded to the run::

    uv run --extra cu130 python experiments/hooke/preflight/weight_drift.py training.max_steps=10
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import PLOTS_DIR, parse_args, spawn_run  # noqa: E402

# Guards against division by zero for parameters initialized exactly at zero.
EPS = 1.0e-12

# Parameters whose final relative drift is below this are reported as static.
STATIC_THRESHOLD = 1.0e-12


def load_weight_trajectory(run_dir: Path) -> tuple[list[int], dict[str, list[torch.Tensor]]]:
    """Return ``(steps, name -> per-step parameter tensors)`` from checkpoints.

    Checkpoints are loaded to CPU; only ``model_state_dict`` is used.
    """

    checkpoint_dir = run_dir / "checkpoints"
    step_files: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if match:
            step_files.append((int(match.group(1)), path))
    if len(step_files) < 2:
        raise RuntimeError(
            f"need at least two step_*.pt checkpoints in {checkpoint_dir} to measure "
            f"drift, found {len(step_files)}; run with checkpoint.every_n_steps=1"
        )
    step_files.sort()

    steps: list[int] = []
    trajectory: dict[str, list[torch.Tensor]] = {}
    for step, path in step_files:
        # weights_only=False: the payload also pickles sampler MCMC state.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        steps.append(step)
        for name, tensor in payload["model_state_dict"].items():
            trajectory.setdefault(name, []).append(tensor.detach().double())
    return steps, trajectory


def compute_drift(
    trajectory: dict[str, list[torch.Tensor]],
) -> dict[str, tuple[list[float], list[float]]]:
    """Map parameter name to per-step ``(absolute drift, relative drift)``."""

    drift: dict[str, tuple[list[float], list[float]]] = {}
    for name, tensors in trajectory.items():
        initial = tensors[0]
        initial_norm = float(initial.norm())
        absolute = [float((tensor - initial).norm()) for tensor in tensors]
        relative = [value / (initial_norm + EPS) for value in absolute]
        drift[name] = (absolute, relative)
    return drift


def print_summary(steps: list[int], drift: dict[str, tuple[list[float], list[float]]]) -> None:
    """Print final per-parameter drift, largest movers first."""

    rows = sorted(drift.items(), key=lambda item: item[1][1][-1], reverse=True)
    name_width = max(len(name) for name in drift)
    print(f"\n[weight_drift] drift from step {steps[0]} to step {steps[-1]}:")
    print(f"{'parameter'.ljust(name_width)}  {'abs drift':>12}  {'rel drift':>12}")
    for name, (absolute, relative) in rows:
        flag = "  <- static" if relative[-1] < STATIC_THRESHOLD else ""
        print(f"{name.ljust(name_width)}  {absolute[-1]:>12.4e}  {relative[-1]:>12.4e}{flag}")

    static = [name for name, (_, relative) in rows if relative[-1] < STATIC_THRESHOLD]
    moving = len(rows) - len(static)
    print(f"\n[weight_drift] {moving}/{len(rows)} parameters moved; {len(static)} static")


def plot_drift(
    steps: list[int],
    drift: dict[str, tuple[list[float], list[float]]],
    run_id: str,
    output_path: Path,
) -> None:
    """Plot global absolute drift and per-parameter relative drift curves."""

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Global drift: L2 norm over the concatenation of all parameter deltas.
    global_drift = [
        sum(absolute[index] ** 2 for absolute, _ in drift.values()) ** 0.5
        for index in range(len(steps))
    ]
    top.plot(steps, global_drift, marker="o", markersize=3)
    top.set_ylabel("global ||theta_t - theta_0||")
    top.grid(True, alpha=0.3)

    # Per-parameter relative drift; legend only labels the biggest movers.
    ranked = sorted(drift.items(), key=lambda item: item[1][1][-1], reverse=True)
    for rank, (name, (_, relative)) in enumerate(ranked):
        label = name if rank < 8 else None
        bottom.plot(steps, relative, linewidth=1, alpha=0.8, label=label)
    bottom.set_ylabel("relative drift per parameter")
    bottom.set_xlabel("step")
    bottom.set_yscale("log")
    bottom.grid(True, alpha=0.3)
    bottom.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"{run_id} weight drift", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Spawn the preflight run, analyze checkpoint drift, return the run's exit code."""

    args = parse_args(argv)
    exit_code, run_dir = spawn_run(args.config, args.overrides)

    steps, trajectory = load_weight_trajectory(run_dir)
    drift = compute_drift(trajectory)
    print_summary(steps, drift)

    run_id = run_dir.name
    output_path = PLOTS_DIR / f"weight-drift-{run_id}.png"
    plot_drift(steps, drift, run_id, output_path)
    print(f"[weight_drift] run_dir={run_dir}")
    print(f"[weight_drift] plot={output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
