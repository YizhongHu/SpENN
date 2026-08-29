"""Instrument a FermiNet-family scaling arm without modifying FermiNet.

This runner is deliberately separate from :mod:`scaling_probe`: the latter is
backend-agnostic, while this module knows FermiNet's ``train_stats.csv`` and
the exact call where FermiNet consumes its per-device batch.  The instrumented
value is emitted at that call site; it is not reconstructed from the requested
GPU count by the probe.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

from experiments.baselines.statistics import blocking_stderr


class FermiNetScalingError(ValueError):
    """Raised when a completed FermiNet arm has unusable scientific output."""


def _load_config(path: Path) -> ModuleType:
    """Load an ordinary FermiNet config file by path."""

    spec = importlib.util.spec_from_file_location("scaling_probe_ferminet_config", path)
    if spec is None or spec.loader is None:
        raise FermiNetScalingError(f"cannot load FermiNet config {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "get_config", None)):
        raise FermiNetScalingError(f"FermiNet config {path} does not define get_config()")
    return module


def summarize_training_tail(run_dir: str | Path, *, first_step: int) -> tuple[float, float]:
    """Return a blocking-error energy estimate from a completed training tail."""

    if first_step < 0:
        raise FermiNetScalingError("first training-tail step cannot be negative")
    path = Path(run_dir) / "train_stats.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise FermiNetScalingError(f"cannot read FermiNet statistics {path}: {exc}") from exc
    try:
        energies = [float(row["energy"]) for row in rows if int(row["step"]) >= first_step]
    except (KeyError, TypeError, ValueError) as exc:
        raise FermiNetScalingError(f"invalid FermiNet train_stats.csv at {path}") from exc
    if len(energies) < 32:
        raise FermiNetScalingError(
            f"FermiNet training tail after step {first_step} has only {len(energies)} values; need at least 32 for blocking"
        )
    stderr, blocks = blocking_stderr(energies)
    if blocks is None or stderr <= 0:
        raise FermiNetScalingError("FermiNet training-tail error is not a usable blocking estimate")
    return statistics.fmean(energies), stderr


def run(config_path: str | Path, *, run_dir: str | Path, first_tail_step: int) -> tuple[float, float]:
    """Execute FermiNet and emit evidence required by ``scaling_probe``.

    Importing FermiNet is intentionally delayed to this execution-only
    function, keeping module import and its unit tests cluster-free.
    """

    try:
        from ferminet import base_config  # noqa: PLC0415
        from ferminet import train  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - requires the external runtime
        raise FermiNetScalingError(f"FermiNet runtime is unavailable: {exc}") from exc
    module = _load_config(Path(config_path))
    cfg: Any = base_config.resolve(module.get_config())
    expected_run_dir = Path(run_dir).resolve()
    configured_run_dir = Path(str(cfg.log.save_path)).resolve()
    if configured_run_dir != expected_run_dir:
        raise FermiNetScalingError(
            f"config save_path {configured_run_dir} does not match requested run directory {expected_run_dir}"
        )

    original = train.mcmc.make_mcmc_step

    def recorded_make_mcmc_step(*args: Any, **kwargs: Any) -> Any:
        # ``train.train`` has already validated divisibility and calculated this
        # value.  Capturing its actual argument avoids reimplementing that
        # calculation in the probe.
        batch = kwargs.get("batch_size", args[1] if len(args) > 1 else None)
        print(f"SCALING_PROBE device_batch_size={batch}", flush=True)
        return original(*args, **kwargs)

    train.mcmc.make_mcmc_step = recorded_make_mcmc_step
    try:
        train.train(cfg)
    finally:
        train.mcmc.make_mcmc_step = original
    energy, stderr = summarize_training_tail(expected_run_dir, first_step=first_tail_step)
    print(f"SCALING_PROBE energy={energy:.12g} stderr={stderr:.12g}", flush=True)
    return energy, stderr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--first-tail-step", type=int, default=400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entrypoint."""

    args = _parser().parse_args(argv)
    try:
        run(args.config, run_dir=args.run_dir, first_tail_step=args.first_tail_step)
    except FermiNetScalingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
