"""Instrument one DeepQMC scaling arm without modifying DeepQMC itself.

The generic scaling probe owns timestamping and comparison.  This small
backend runner supplies evidence from DeepQMC's actual execution: the
application's device report, sampler construction's per-device batch, and an
HDF5 write after every optimizer step.  Its imports are deliberately delayed
so the module and its unit-tested helpers remain usable without DeepQMC.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
from pathlib import Path
import sys
from typing import Any, Sequence

from experiments.baselines.statistics import blocking_stderr


class DeepQMCScalingError(ValueError):
    """Raised when a DeepQMC arm lacks usable scientific output."""


def per_device_batch(total_batch_size: int, device_count: int) -> int:
    """Return one device's share of a validated total electron batch."""

    if total_batch_size <= 0 or device_count <= 0:
        raise DeepQMCScalingError("batch size and device count must be positive")
    if total_batch_size % device_count:
        raise DeepQMCScalingError(
            f"electron batch size {total_batch_size} does not divide {device_count} devices"
        )
    return total_batch_size // device_count


def training_tail(energies: Sequence[float], *, first_tail_step: int) -> tuple[float, float]:
    """Return blocking-error energy evidence from a completed DeepQMC trace."""

    if first_tail_step < 0:
        raise DeepQMCScalingError("first training-tail step cannot be negative")
    tail = [float(energy) for energy in energies[first_tail_step:]]
    if len(tail) < 32:
        raise DeepQMCScalingError(
            f"DeepQMC training tail after step {first_tail_step} has only {len(tail)} values; "
            "need at least 32 for blocking"
        )
    stderr, blocks = blocking_stderr(tail)
    if blocks is None or stderr <= 0:
        raise DeepQMCScalingError("DeepQMC training-tail error is not a usable blocking estimate")
    return statistics.fmean(tail), stderr


def run(*, run_dir: str | Path, first_tail_step: int, hydra_overrides: Sequence[str]) -> tuple[float, float]:
    """Execute DeepQMC and print device, batch, step, and energy evidence.

    ``hydra_overrides`` must set DeepQMC's work directory, system, ansatz,
    steps, total ``task.electron_batch_size``, and seed.  The runner never
    calculates its evidence from the requested visibility: the device report
    comes from :func:`deepqmc.app.detect_devices`, while the per-device batch
    is recorded at DeepQMC's actual sampler-state construction call.
    """

    if not hydra_overrides:
        raise DeepQMCScalingError("at least one DeepQMC Hydra override is required")
    expected_run_dir = Path(run_dir).resolve()
    try:
        from deepqmc import app  # noqa: PLC0415
        import jax  # noqa: PLC0415
        from experiments.baselines.adapters.deepqmc import read_energies  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - requires the external runtime
        raise DeepQMCScalingError(f"DeepQMC runtime is unavailable: {exc}") from exc

    deepqmc_train = importlib.import_module("deepqmc.train")

    original_initializer = deepqmc_train.initialize_sampler_state
    original_h5_logger = deepqmc_train.H5Logger

    def recorded_initializer(
        rng: Any, sampler: Any, params: Any, electron_batch_size: int, nuc_coords: Any
    ) -> Any:
        # This function receives DeepQMC's validated total and does the actual
        # device split internally.  Reading JAX here proves the process count.
        batch = per_device_batch(int(electron_batch_size), jax.device_count())
        print(f"SCALING_PROBE device_batch_size={batch}", flush=True)
        return original_initializer(rng, sampler, params, electron_batch_size, nuc_coords)

    class ScalingH5Logger(original_h5_logger):
        """Emit a marker only after DeepQMC durably writes an optimizer step."""

        def update(self, single_device_data: Any) -> None:
            super().update(single_device_data)
            print(f"SCALING_PROBE Step {int(single_device_data['step'])}", flush=True)

    deepqmc_train.initialize_sampler_state = recorded_initializer
    deepqmc_train.H5Logger = ScalingH5Logger
    original_argv = sys.argv
    try:
        # Hydra's public CLI is the supported configuration entrypoint.  Use
        # the caller's complete argument vector rather than rebuilding config.
        sys.argv = [original_argv[0], *hydra_overrides]
        app.cli()
    finally:
        sys.argv = original_argv
        deepqmc_train.initialize_sampler_state = original_initializer
        deepqmc_train.H5Logger = original_h5_logger

    energies = read_energies(expected_run_dir / "training")
    energy, stderr = training_tail(energies, first_tail_step=first_tail_step)
    print(f"SCALING_PROBE energy={energy:.12g} stderr={stderr:.12g}", flush=True)
    return energy, stderr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--first-tail-step", type=int, required=True)
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entrypoint."""

    args = _parser().parse_args(argv)
    overrides = list(args.hydra_overrides)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    try:
        run(
            run_dir=args.run_dir,
            first_tail_step=args.first_tail_step,
            hydra_overrides=overrides,
        )
    except DeepQMCScalingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
