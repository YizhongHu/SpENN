"""Run the Hooke pair GPU preflight training config and plot training curves.

Spawns one training run from ``experiments/hooke/configs/preflight/pair_train_gpu.yaml``
(in-process, through ``spenn.run.run_from_config``), then reads the run's
``metrics.jsonl`` and plots loss, energy, energy stderr, sampler acceptance,
and gradient norm against the training step. The figure is saved to
``experiments/hooke/preflight/plots/<run_id>.png``.

If the run fails partway, whatever metrics were logged are still plotted so
the curve up to the failure can be inspected; the script exits with the run's
exit code either way.

Usage
-----
Run from the repository root inside the GPU environment::

    export UV_PROJECT_ENVIRONMENT=.venv-gpu
    uv run --extra cu130 python experiments/hooke/preflight/plotting.py

Extra OmegaConf dotlist overrides are forwarded to the run::

    uv run --extra cu130 python experiments/hooke/preflight/plotting.py training.max_steps=10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from spenn.run import load_config, run_from_config  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments" / "hooke" / "configs" / "preflight" / "pair_train_gpu.yaml"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# (panel title, metrics.jsonl namespace, metric key) per subplot, in plot order.
PANELS = (
    ("loss", "train", "loss"),
    ("energy", "train", "energy"),
    ("energy stderr", "train", "energy_stderr"),
    ("acceptance rate", "train/sampler", "acceptance_rate"),
    ("gradient norm", "train", "grad_norm"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the config path and pass-through dotlist overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training YAML config (default: the GPU preflight config).",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args(argv)


def spawn_run(config: Path, overrides: Sequence[str]) -> tuple[int, Path]:
    """Run the config and return ``(exit_code, run_dir)``.

    The run directory is detected by diffing the experiment's output sector
    before and after the run, so it works regardless of how run setup
    generates the run id.
    """

    cfg = load_config(str(config), list(overrides))
    sector_dir = (
        Path(str(OmegaConf.select(cfg, "run.root", default="outputs")))
        / str(OmegaConf.select(cfg, "experiment.name", default="experiment"))
        / str(OmegaConf.select(cfg, "experiment.sector", default="default"))
    )
    before = set(sector_dir.glob("*")) if sector_dir.exists() else set()

    command = " ".join(["plotting.py", f"--config={config}", *overrides])
    exit_code = run_from_config(cfg, config_path=str(config), command=command)

    new_dirs = sorted(set(sector_dir.glob("*")) - before)
    if len(new_dirs) != 1:
        raise RuntimeError(f"expected one new run dir under {sector_dir}, found {new_dirs}")
    return exit_code, new_dirs[0]


def load_series(run_dir: Path) -> dict[tuple[str, str], tuple[list[int], list[float]]]:
    """Map ``(namespace, key)`` to parallel ``(steps, values)`` lists."""

    series: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    wanted = {(namespace, key) for _, namespace, key in PANELS}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = record.get("step")
        if step is None:
            continue
        for key, value in record.get("metrics", {}).items():
            if (record["namespace"], key) not in wanted or not isinstance(value, (int, float)):
                continue
            steps, values = series.setdefault((record["namespace"], key), ([], []))
            steps.append(int(step))
            values.append(float(value))
    return series


def plot_series(
    series: dict[tuple[str, str], tuple[list[int], list[float]]],
    run_id: str,
    output_path: Path,
) -> None:
    """Render one stacked panel per metric and save the figure."""

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8, 2.2 * len(PANELS)), sharex=True)
    for axis, (title, namespace, key) in zip(axes, PANELS):
        steps, values = series.get((namespace, key), ([], []))
        axis.plot(steps, values, marker="o", markersize=3, linewidth=1)
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
        if not steps:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)
    axes[-1].set_xlabel("step")
    fig.suptitle(run_id, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Spawn the preflight run, plot its curves, and return the run's exit code."""

    args = parse_args(argv)
    exit_code, run_dir = spawn_run(args.config, args.overrides)

    run_id = run_dir.name
    output_path = PLOTS_DIR / f"metrics-{run_id}.png"
    plot_series(load_series(run_dir), run_id, output_path)
    print(f"[plotting] run_dir={run_dir}")
    print(f"[plotting] plot={output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
