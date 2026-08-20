"""Translate a DeepQMC run directory into a :class:`BaselineRecord`.

DeepQMC writes ``training/result.h5`` rather than the ``train_stats.csv`` the
FermiNet adapter reads, which is why a separate adapter exists at all. Without
it every DeepQMC measurement is stranded outside ``results.jsonl``.

Four properties of this module are deliberate and must survive any refactor.

**``code`` is ``"deepqmc"``, never ``"ferminet"``.** DeepQMC ships ansatz options
named ``ferminet``, ``psiformer``, ``lapnet`` and ``deeperwin``, but those are
DeepQMC's *reimplementations*. A record with ``code="ferminet",
ansatz="ferminet"`` claims a run of the `google-deepmind/ferminet` codebase;
``code="deepqmc", ansatz="ferminet"`` claims DeepQMC's version. Those are
different scientific objects, and the mistake is invisible to a reader because
the ``ansatz`` field looks right either way. ``default`` is the exception worth
knowing: DeepQMC *is* PauliNet's own codebase, so that one is native.

**Energies come from the HDF5 dataset, never from stdout.** DeepQMC logs
``Progress:`` at step 1 and then nothing, so stdout carries no trajectory at
all.

**The default tail is long.** A short tail has produced variationally
impossible below-exact energies four separate times in this program, including
in a table this lane published internally. :data:`DEFAULT_TAIL_FRACTION` is 0.25
rather than the FermiNet adapter's 0.1 for that reason, and the record's notes
carry the windowed convergence verdict so a reader cannot mistake a
still-descending run for a converged one.

**A killed run leaves the file flagged, and that is not corruption.** A DeepQMC
job stopped by a wall-clock timeout dies without closing its HDF5, leaving the
superblock marked open-for-write. The file is intact and readable; see
:func:`read_energies` for how, and note that the identical error message also
appears for a genuinely live writer.

Examples
--------
::

    uv run python -m experiments.baselines.adapters.deepqmc \\
        --run-dir path/to/run --system-id he_atom --batch-size 4096 \\
        --ansatz lapnet
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from experiments.baselines.errors import AdapterError
from experiments.baselines.records import BaselineRecord
from experiments.baselines.statistics import (
    MIN_TAIL_STEPS,
    SIGN_TEST_WINDOWS,
    blocking_inflation,
    blocking_stderr,
    select_tail,
    sign_test,
)

RESULT_FILENAME = "result.h5"
TRAINING_SUBDIR = "training"
RECORD_FILENAME = "baseline_record.json"

#: Dataset holding the per-step mean local energy. The sibling ``samples``
#: datasets are per-walker and run to gigabytes; they are never read here.
ENERGY_DATASET = "local_energy/mean"

#: Fraction of the trailing trace averaged by default. Deliberately larger than
#: the FermiNet adapter's 0.1 -- see the module docstring.
DEFAULT_TAIL_FRACTION = 0.25

#: DeepQMC's own repository is PauliNet's, so this ansatz name is native code
#: rather than a reimplementation of someone else's method.
NATIVE_ANSATZES = frozenset({"default"})

#: ``NVIDIA H200, GPU-1a4626a2-..., 143771 MiB``
_NVIDIA_SMI = re.compile(r"^\s*(NVIDIA [^,]+),\s*(GPU-[0-9a-f-]+)", re.MULTILINE)
_START_STAMP = re.compile(r"start=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")
_END_STAMP = re.compile(r"end=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")


def result_path(run_dir: Path) -> Path:
    """Return the HDF5 path for a run directory.

    DeepQMC nests its output under ``training/``; accept either the run root or
    that subdirectory so a caller need not remember which.
    """

    direct = run_dir / RESULT_FILENAME
    if direct.is_file():
        return direct
    return run_dir / TRAINING_SUBDIR / RESULT_FILENAME


def read_energies(run_dir: Path) -> list[float]:
    """Return the per-step mean local energy of a DeepQMC run.

    The dataset has shape ``(steps, 1, 1)`` in float32; it is flattened and
    widened to float64 so downstream arithmetic is not done in single
    precision.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run root, or its ``training`` subdirectory.

    Returns
    -------
    list of float
        One energy per optimizer step, in file order.

    Raises
    ------
    AdapterError
        If ``h5py`` is unavailable, the file or dataset is missing, or the
        dataset has no rows. An empty run is an error, not an empty result.

    Notes
    -----
    A run killed by a timeout leaves the superblock flagged open-for-write, and
    opening it then fails with *"file is already open for write"*. That is a
    recoverable state, not corruption: this function retries with
    ``swmr=True``, which reads the file without modifying it. ``h5clear -s``
    would also clear the flag but writes to the file, and run data is not to be
    modified. Passing ``locking=False`` does **not** help, because the flag is
    checked independently of file locking.

    The same message appears for a genuinely live writer, so a successful read
    here does not prove the producing job has finished. Check the scheduler for
    that.
    """

    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        raise AdapterError(
            "reading DeepQMC output needs h5py, which is not a TPEN dependency; "
            "run this adapter with the DeepQMC virtualenv that already provides it"
        ) from error

    path = result_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {RESULT_FILENAME} under {run_dir}")

    def _extract(handle: "h5py.File") -> list[float]:
        if ENERGY_DATASET not in handle:
            raise AdapterError(f"{path} has no '{ENERGY_DATASET}' dataset")
        return [float(value) for value in handle[ENERGY_DATASET][:].ravel()]

    try:
        with h5py.File(path, "r") as handle:
            energies = _extract(handle)
    except OSError as error:
        if "already open for write" not in str(error):
            raise AdapterError(f"cannot open {path}: {error}") from error
        # Killed-run recovery, or a live writer. Read-only, file unmodified.
        with h5py.File(path, "r", swmr=True) as handle:
            energies = _extract(handle)

    if not energies:
        raise AdapterError(f"{path} has an empty '{ENERGY_DATASET}' dataset")
    return energies


def parse_device(log_text: str) -> tuple[str | None, str | None]:
    """Return ``(device_type, gpu_model)`` parsed from a job log.

    The model is read from the ``nvidia-smi`` line the job scripts print inside
    the allocation, never inferred from the partition name. ``seas_gpu`` mixes
    H200 and A100 nodes, so the partition does not determine the hardware and
    two runs in this program silently drew different cards.
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


def record_from_series(
    energies: Sequence[float],
    *,
    system_id: str,
    batch_size: int,
    ansatz: str,
    estimator: str = "training_tail",
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    run_id: str,
    code_commit: str | None = None,
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    device_type: str | None = None,
    gpu_model: str | None = None,
    wall_clock_seconds: float | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build a record from an already-read energy series.

    Separated from :func:`build_record` so that every decision this adapter
    makes -- tail selection, blocking, the convergence verdict, the provenance
    text -- is exercisable without an HDF5 file, and therefore without ``h5py``
    installed.

    Parameters
    ----------
    energies : sequence of float
        One mean local energy per step, in file order.
    ansatz : str
        Which DeepQMC ansatz ran, e.g. ``"lapnet"`` or ``"default"``. Required,
        never inferred: directory and log names in this campaign do not track
        the ansatz, and a hardcoded value previously mislabelled two runs.
    estimator : str, optional
        ``"training_tail"`` or ``"inference"``.
    tail_fraction : float, optional
        Trailing fraction averaged for the estimate.
    note : str, optional
        Operator caveat appended verbatim to the generated provenance text, for
        facts the numbers cannot carry -- a run whose geometry deviates from the
        registered system, for instance. Appended, never substituted: the
        generated sentences are the record's own account of how it was made and
        no argument may edit them. Whitespace-only text is rejected, because a
        caveat that silently vanishes is worse than no argument at all.

    Returns
    -------
    BaselineRecord
        Validated record carrying ``code="deepqmc"``.

    Raises
    ------
    AdapterError
        If ``tail_fraction`` is out of range or selects fewer than two samples,
        or if ``note`` is given but carries no text.
    """

    if note is not None and not note.strip():
        raise AdapterError("note was given but is empty; omit it or write the caveat")

    series = [float(value) for value in energies]
    # The window is chosen by absolute step count, not by fraction alone: a
    # fraction is scale-relative and 0.25 of a short run is too few steps to
    # average out the slow mode. See statistics.MIN_TAIL_STEPS.
    window = select_tail(
        len(series),
        tail_fraction,
        min_steps=min_tail_steps,
        allow_below_floor=allow_short_tail,
    )
    tail = series[-window:]

    stderr, _ = blocking_stderr(tail)

    # Convergence and autocorrelation are reported, never used to alter the
    # number. A verdict that could change the estimate would be a selection
    # rule; this one only describes.
    try:
        signs, monotone = sign_test(tail, windows=SIGN_TEST_WINDOWS)
        verdict = (
            f"windowed sign test over {SIGN_TEST_WINDOWS} windows gives '{signs}': "
            + (
                "MONOTONE, so the series was still drifting at the end of this "
                "trace and the run may not have converged"
                if monotone
                else "not monotone, consistent with noise rather than drift"
            )
        )
    except AdapterError:
        verdict = (
            f"tail of {len(tail)} steps is too short for a {SIGN_TEST_WINDOWS}-window "
            "sign test, so convergence is UNASSESSED"
        )

    try:
        inflation = f"{blocking_inflation(tail):.2f}x"
    except AdapterError:
        inflation = "undefined"

    native = " DeepQMC is PauliNet's own codebase, so this ansatz is native rather than a reimplementation." if ansatz in NATIVE_ANSATZES else (
        f" '{ansatz}' here is DeepQMC's REIMPLEMENTATION, not the {ansatz} authors' own code; "
        "no claim about that method may rest on this row."
    )

    # State the window in STEPS, not only as a percentage: a reader cannot judge
    # "25%" without knowing the run length, and it was exactly that ambiguity
    # that let a 5000-step window pass for a long one.
    short = " Window is BELOW the standard minimum, so this estimate is provisional." if len(tail) < min_tail_steps else ""
    estimator_text = (
        f"Training-tail average over the last {len(tail)} of {len(series)} steps."
        if estimator == "training_tail"
        else f"Fixed-parameter inference pass over {len(tail)} of {len(series)} steps."
    ) + short

    return BaselineRecord(
        system_id=system_id,
        code="deepqmc",
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=statistics.fmean(tail),
        energy_stderr_hartree=stderr,
        local_energy_variance_hartree2=None,
        steps=len(series),
        samples=len(series) * batch_size,
        wall_clock_seconds=wall_clock_seconds,
        estimator=estimator,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=1 if device_type == "cuda" else None,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id,
        # The adapter cannot know the collector's scan root. Leave this blank so
        # collect() stamps the collision-free path relative to that root.
        run_dir=None,
        collected_at=None,
        notes=(
            f"{estimator_text} Blocked standard error, autocorrelation inflation "
            f"{inflation} over the naive estimate. Energies read from "
            f"'{ENERGY_DATASET}', never from stdout. {verdict}.{native}"
            # The operator caveat goes last so the generated account stays
            # intact and byte-identical to a no-note emission up to this point.
            + (f" {note.strip()}" if note is not None else "")
        ),
    )


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    batch_size: int,
    ansatz: str,
    estimator: str = "training_tail",
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    log_path: Path | None = None,
    code_commit: str | None = None,
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    run_id: str | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a completed DeepQMC run directory.

    Reads the energy series from HDF5 and the device identity and wall clock
    from the job log, then delegates every decision to
    :func:`record_from_series`.
    """

    energies = read_energies(run_dir)

    device_type, gpu_model, wall_clock = None, None, None
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        device_type, gpu_model = parse_device(log_text)
        wall_clock = parse_wall_clock_seconds(log_text)

    return record_from_series(
        energies,
        system_id=system_id,
        batch_size=batch_size,
        ansatz=ansatz,
        estimator=estimator,
        tail_fraction=tail_fraction,
        min_tail_steps=min_tail_steps,
        allow_short_tail=allow_short_tail,
        run_id=run_id or run_dir.name,
        code_commit=code_commit,
        optimizer=optimizer,
        dtype=dtype,
        seed=seed,
        parameter_count=parameter_count,
        device_type=device_type,
        gpu_model=gpu_model,
        wall_clock_seconds=wall_clock,
        note=note,
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
    # Required, not defaulted. The FermiNet adapter hardcoded its ansatz once
    # and mislabelled two runs; nothing about a DeepQMC run directory reveals
    # which ansatz produced it.
    parser.add_argument("--ansatz", required=True)
    parser.add_argument(
        "--estimator",
        choices=("training_tail", "inference"),
        default="training_tail",
        help="use 'inference' for a fixed-parameter evaluation pass",
    )
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument(
        "--min-tail-steps",
        type=int,
        default=MIN_TAIL_STEPS,
        help="absolute floor on the estimator window, in steps",
    )
    parser.add_argument(
        "--allow-short-tail",
        action="store_true",
        help="accept a window below the floor for a short run; the record says so",
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--note",
        default=None,
        help=(
            "operator caveat appended to the record's generated provenance text, "
            "for facts the numbers cannot carry (e.g. the run's geometry deviates "
            "from the registered system)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            batch_size=args.batch_size,
            ansatz=args.ansatz,
            estimator=args.estimator,
            tail_fraction=args.tail_fraction,
            min_tail_steps=args.min_tail_steps,
            allow_short_tail=args.allow_short_tail,
            log_path=args.log_path,
            code_commit=args.code_commit,
            seed=args.seed,
            note=args.note,
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
