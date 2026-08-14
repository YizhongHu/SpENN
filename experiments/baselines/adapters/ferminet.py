"""Translate a FermiNet run directory into a :class:`BaselineRecord`.

A FermiNet run leaves ``train_stats.csv`` (columns ``step,energy,ewmean,ewvar,
pmove``) in its save path, and the job script leaves a stdout log carrying the
device identity and the run's start/end stamps. This module reads both and
emits the one ``baseline_record.json`` the collector expects.

Two properties of the energy estimate are deliberate and must survive any
refactor.

**The error bar comes from blocking, not from the naive standard error.**
Successive VMC steps are correlated, so ``sigma/sqrt(n)`` understates the
uncertainty. :func:`blocking_stderr` repeatedly pair-averages the series and
takes the largest standard error seen while enough blocks remain to estimate
one reliably.

**The estimator is a training-tail average, which is not what FermiNet's paper
reports.** Their table values come from a separate post-training evaluation
phase. The difference is recorded in the record's ``notes`` so a later
comparison cannot silently treat the two as equivalent.

Examples
--------
::

    uv run python -m experiments.baselines.adapters.ferminet \\
        --run-dir path/to/run --system-id li_atom --batch-size 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from experiments.baselines.records import BaselineRecord

TRAIN_STATS_FILENAME = "train_stats.csv"
RECORD_FILENAME = "baseline_record.json"

#: Smallest number of blocks that still supports a usable variance estimate.
#: Below this the standard error is itself so noisy that a larger value means
#: nothing, so blocking stops rather than reporting a spuriously wide bar.
MIN_BLOCKS = 32

#: ``NVIDIA A100-SXM4-40GB, GPU-39166c9d-..., 40960 MiB``
_NVIDIA_SMI = re.compile(r"^\s*(NVIDIA [^,]+),\s*(GPU-[0-9a-f-]+)", re.MULTILINE)
_START_STAMP = re.compile(r"start=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")
_END_STAMP = re.compile(r"end=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")


class AdapterError(RuntimeError):
    """A run directory could not be turned into a record.

    Raised rather than returning a partial record: a run with no usable energy
    must fail loudly, never appear as a record with a null energy or vanish
    from the collection silently.
    """


def read_energies(run_dir: Path) -> list[float]:
    """Return the per-step energy column of a FermiNet run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing ``train_stats.csv``.

    Returns
    -------
    list of float
        One energy per optimizer step, in file order.

    Raises
    ------
    AdapterError
        If the file is missing, has no ``energy`` column, or has no data rows.
        An empty run is an error, not an empty result.
    """

    stats_path = run_dir / TRAIN_STATS_FILENAME
    if not stats_path.is_file():
        raise AdapterError(f"no {TRAIN_STATS_FILENAME} in {run_dir}")

    with stats_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "energy" not in reader.fieldnames:
            raise AdapterError(f"{stats_path} has no 'energy' column")
        energies = [float(row["energy"]) for row in reader if row.get("energy")]

    if not energies:
        raise AdapterError(f"{stats_path} has no data rows")
    return energies


def blocking_stderr(values: Sequence[float], min_blocks: int = MIN_BLOCKS) -> tuple[float, int]:
    """Return a correlation-corrected standard error by pair-average blocking.

    Implements Flyvbjerg-Petersen blocking: repeatedly average adjacent pairs,
    recording the standard error at each level. Correlated data shows the
    standard error rising with block size and then plateauing; the plateau is
    the honest bar.

    Parameters
    ----------
    values : sequence of float
        The series to estimate the mean's uncertainty for.
    min_blocks : int, optional
        Stop once fewer than this many blocks remain.

    Returns
    -------
    tuple of (float, int)
        The standard error, and the number of blocks it was computed from.

    Raises
    ------
    AdapterError
        If fewer than two values are supplied.
    """

    data = [float(value) for value in values]
    if len(data) < 2:
        raise AdapterError("blocking needs at least two values")

    best_stderr = 0.0
    best_blocks = len(data)
    while len(data) >= max(min_blocks, 2):
        stderr = math.sqrt(statistics.variance(data) / len(data))
        if stderr > best_stderr:
            best_stderr, best_blocks = stderr, len(data)
        # Pair-average into the next blocking level, dropping a trailing odd
        # sample rather than pairing it with nothing.
        data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, len(data) - 1, 2)]

    return best_stderr, best_blocks


def parse_device(log_text: str) -> tuple[str | None, str | None]:
    """Return ``(device_type, gpu_model)`` parsed from a job log.

    The model is read from the ``nvidia-smi`` line the job scripts print inside
    the allocation, never inferred from the partition name: the delivered
    device has been observed to disagree with the partition's advertised GRES.
    """

    match = _NVIDIA_SMI.search(log_text)
    if match is None:
        return None, None
    return "cuda", match.group(1).strip()


def parse_wall_clock_seconds(log_text: str) -> float | None:
    """Return elapsed seconds from the ``start=``/``end=`` stamps, or None."""

    start, end = _START_STAMP.search(log_text), _END_STAMP.search(log_text)
    if start is None or end is None:
        return None
    delta = datetime.fromisoformat(end.group(1)) - datetime.fromisoformat(start.group(1))
    return delta.total_seconds()


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    batch_size: int,
    tail_fraction: float = 0.1,
    log_path: Path | None = None,
    code_commit: str | None = None,
    ansatz: str = "ferminet",
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    run_id: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a completed FermiNet run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory holding ``train_stats.csv``.
    system_id : str
        Key into ``experiments/baselines/systems.yaml``.
    batch_size : int
        Walkers per step, needed to convert steps into local-energy evaluations.
    tail_fraction : float, optional
        Fraction of the trailing run averaged for the energy estimate.
    log_path : pathlib.Path or None, optional
        Job stdout log, used for device identity and wall clock.
    dtype : str or None, optional
        Left ``None`` unless known. FermiNet does not state its dtype, so
        guessing one would fabricate provenance.

    Returns
    -------
    BaselineRecord
        Validated record, ready to serialise.

    Raises
    ------
    AdapterError
        If the run has no usable energy series, or ``tail_fraction`` selects
        fewer than two samples.
    """

    if not 0.0 < tail_fraction <= 1.0:
        raise AdapterError(f"tail_fraction must be in (0, 1], got {tail_fraction}")

    energies = read_energies(run_dir)
    tail_start = int(len(energies) * (1.0 - tail_fraction))
    tail = energies[tail_start:]
    if len(tail) < 2:
        raise AdapterError(
            f"tail_fraction {tail_fraction} selects {len(tail)} of {len(energies)} steps; need >= 2"
        )

    stderr, n_blocks = blocking_stderr(tail)
    device_type, gpu_model, wall_clock = None, None, None
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        device_type, gpu_model = parse_device(log_text)
        wall_clock = parse_wall_clock_seconds(log_text)

    return BaselineRecord(
        system_id=system_id,
        code="ferminet",
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=statistics.fmean(tail),
        energy_stderr_hartree=stderr,
        local_energy_variance_hartree2=None,
        steps=len(energies),
        samples=len(energies) * batch_size,
        wall_clock_seconds=wall_clock,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=1 if device_type == "cuda" else None,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id or run_dir.name,
        run_dir=run_dir.name,
        collected_at=None,
        notes=(
            f"Training-tail average over the last {tail_fraction:.0%} of steps "
            f"({len(tail)} samples), blocked standard error from {n_blocks} blocks. "
            "NOT the estimator FermiNet's published table uses: those values come "
            "from a separate post-training evaluation phase, so this number is "
            "expected to sit slightly high and the two must not be compared as "
            "though they were the same quantity."
        ),
    )


def write_record(record: BaselineRecord, run_dir: Path) -> Path:
    """Write ``baseline_record.json`` into ``run_dir`` and return its path."""

    path = run_dir / RECORD_FILENAME
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            batch_size=args.batch_size,
            tail_fraction=args.tail_fraction,
            log_path=args.log_path,
            code_commit=args.code_commit,
            seed=args.seed,
        )
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(record.to_json_dict(), indent=2))
        return 0

    print(write_record(record, args.run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
