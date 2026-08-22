"""Translate a Neural Pfaffian run directory into a :class:`BaselineRecord`.

A Neural Pfaffian run (Gao & Guennemann, NeurIPS 2024) leaves one CSV per stage
in its output directory: ``mcmc_log.csv`` (thermalization), ``pretrain_log.csv``
(Hartree-Fock pretraining) and ``train_log.csv`` (variational optimization).
Only the last of these carries a variational energy, so it is the only file this
adapter reads. There is no ``main_log.csv`` and no separate inference stage, so
every record this adapter emits is ``estimator="training_tail"``.

Four properties are deliberate and must survive any refactor.

**The column semantics are an unverifiable assumption, so they are asserted
rather than assumed.** The mapping below was read from the ``neural_pfaffian``
source on the cluster and recorded in a note; this module cannot re-read that
source, so it validates the CSV header against the columns it needs and raises a
message naming the note and the read date. An unverifiable assumption converted
into a loud assertion is safe; the same assumption left implicit in a parser is
how a wrong column becomes a published energy.

**``E_std`` is the population standard deviation of the local energies over
walkers, not the standard error of the step mean.** The two differ by
``sqrt(n_walkers)``, which at 4096 walkers is a factor of 64 -- large enough that
mistaking one for the other produces a plausible-looking number rather than an
obvious one. ``E_std**2`` therefore feeds
``local_energy_variance_hartree2``, and it is never used as an error bar.

**The error bar comes from blocking the ``E`` series, not from ``E_std``.**
Successive optimizer steps are correlated, so the uncertainty on the tail mean is
a property of that series and must be estimated from it.

**Wall clock is the sum of the logger's own per-step timer and therefore covers
the VMC stage only.** Pretraining, thermalization and JIT compilation are logged
to other files and are excluded. That is stated in the record's notes, because a
wall clock that silently omits a 10000-step pretrain would overstate this code's
efficiency against a baseline that reports end-to-end time.

Columns of ``train_log.csv``, per the note cited in :data:`SEMANTICS_NOTE`:

==============  ===============================================================
``E``           mean over walkers of the **unclipped** local energy
``E_std``       ``sqrt(mean((e_l - E)**2))`` over walkers -- a spread, not a bar
``grad``        gradient global norm (unused here)
``step``        optimizer step counter (unused here; row order is authoritative)
``time_step``   wall seconds for that step, from ``time.perf_counter()``
==============  ===============================================================

Examples
--------
::

    uv run python -m experiments.baselines.adapters.neural_pfaffian \\
        --run-dir path/to/run --system-id he_atom --walkers-per-step 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
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

TRAIN_LOG_FILENAME = "train_log.csv"
RECORD_FILENAME = "baseline_record.json"

#: Mean local energy per optimizer step.
ENERGY_COLUMN = "E"

#: Population standard deviation of the local energies over walkers. Named
#: "spread" throughout this module so that no reader can mistake it for a bar.
SPREAD_COLUMN = "E_std"

#: Per-step wall seconds. Optional: a record without it reports no wall clock
#: rather than a zero, per the records module's no-placeholder rule.
STEP_TIME_COLUMN = "time_step"

#: Columns without which no record can be built.
REQUIRED_COLUMNS = (ENERGY_COLUMN, SPREAD_COLUMN)

#: Provenance of the column semantics above. Named in the header-mismatch error
#: so that whoever sees it can find the assumption without hunting for it.
SEMANTICS_NOTE = "np-lane-logging-semantics-for-he-record-2026-08-20T1740"
SEMANTICS_READ_DATE = "2026-08-20"
SEMANTICS_SOURCE_COMMIT = "f711f08"

#: A quarter of the trace, matching the DeepQMC adapter. The absolute floor in
#: :data:`MIN_TAIL_STEPS` is what makes this safe on a short run.
DEFAULT_TAIL_FRACTION = 0.25

#: Optimizer stack, per the same note as the column semantics.
DEFAULT_OPTIMIZER = "spring (scale_by_hyperbolic_schedule + clip_by_global_norm)"

#: Mixed precision: the code sets ``JAX_DEFAULT_DTYPE_BITS=32`` while enabling
#: ``jax_enable_x64``, and the spring preconditioner runs in float64. Stated as
#: one string because "float32" or "float64" alone would both be false.
DEFAULT_DTYPE = "mixed: float32 arrays, float64 spring preconditioner (jax_enable_x64)"

#: The only ansatz this codebase implements.
DEFAULT_ANSATZ = "neural_pfaffian"

CODE_NAME = "neural-pfaffian"


def train_log_path(run_dir: Path) -> Path:
    """Return the path of the VMC log inside ``run_dir``."""

    return run_dir / TRAIN_LOG_FILENAME


def _header_mismatch_message(path: Path, fieldnames: Sequence[str] | None, missing: Sequence[str]) -> str:
    """Compose the header-mismatch error, carrying the assumption's provenance.

    The message names the note and the read date on purpose: the mapping cannot
    be re-derived from this repository, so a reader who hits this error needs to
    know which artifact to distrust.
    """

    observed = "no header at all" if fieldnames is None else repr(list(fieldnames))
    return (
        f"{path}: header is {observed}, missing required column(s) {list(missing)}. "
        f"This adapter's column semantics come from Task Orchestrator note "
        f"'{SEMANTICS_NOTE}', read from the neural_pfaffian source at commit "
        f"{SEMANTICS_SOURCE_COMMIT} on {SEMANTICS_READ_DATE}. If the logger's columns "
        "have changed since that read, the note is stale: re-derive the mapping from "
        "source before emitting any record, and do not rename a column to fit this "
        "adapter."
    )


def read_train_log(run_dir: Path) -> tuple[list[float], list[float], list[float] | None]:
    """Read the VMC log and return ``(energies, spreads, step_times)``.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing ``train_log.csv``.

    Returns
    -------
    tuple
        Per-step mean local energies, per-step local-energy spreads, and
        per-step wall seconds -- the last being ``None`` when the log carries no
        ``time_step`` column, so that a missing measurement stays missing.

    Raises
    ------
    AdapterError
        If the file is absent; if the header lacks ``E`` or ``E_std``; if the
        file has no data rows; or if a row carries one of the two required
        values without the other. The last case is rejected rather than dropped
        because dropping it would silently misalign the two series.

    Notes
    -----
    The logger freezes its header from the first row's keys and thereafter emits
    ``data.get(header, "")``, so a later row may legitimately hold empty cells.
    A fully blank row is skipped; a half-filled one is an error.
    """

    path = train_log_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {TRAIN_LOG_FILENAME} in {run_dir}")

    energies: list[float] = []
    spreads: list[float] = []
    step_times: list[float] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if fieldnames is None or column not in fieldnames
        ]
        if missing:
            raise AdapterError(_header_mismatch_message(path, fieldnames, missing))

        has_step_time = STEP_TIME_COLUMN in fieldnames

        for line_number, row in enumerate(reader, start=2):
            energy_cell = (row.get(ENERGY_COLUMN) or "").strip()
            spread_cell = (row.get(SPREAD_COLUMN) or "").strip()
            if not energy_cell and not spread_cell:
                # Wholly blank row: a trailing newline, not a measurement.
                continue
            if not energy_cell or not spread_cell:
                raise AdapterError(
                    f"{path} line {line_number}: {ENERGY_COLUMN}={energy_cell!r} and "
                    f"{SPREAD_COLUMN}={spread_cell!r} -- one is present without the "
                    "other, so the two series would misalign if this row were dropped"
                )
            try:
                energies.append(float(energy_cell))
                spreads.append(float(spread_cell))
            except ValueError as error:
                raise AdapterError(f"{path} line {line_number}: unparseable number: {error}")

            if has_step_time:
                time_cell = (row.get(STEP_TIME_COLUMN) or "").strip()
                # A blank per-step time makes the sum an undercount rather than a
                # measurement, so the whole wall clock is withheld.
                step_times.append(float("nan") if not time_cell else float(time_cell))

    if not energies:
        raise AdapterError(f"{path} has no data rows")

    if not has_step_time or any(value != value for value in step_times):
        return energies, spreads, None
    return energies, spreads, step_times


def record_from_series(
    energies: Sequence[float],
    spreads: Sequence[float],
    *,
    system_id: str,
    walkers_per_step: int,
    step_times: Sequence[float] | None = None,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    ansatz: str = DEFAULT_ANSATZ,
    code_commit: str | None = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    dtype: str | None = DEFAULT_DTYPE,
    device_type: str | None = None,
    gpu_model: str | None = None,
    n_gpus: int | None = None,
    parameter_count: int | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> BaselineRecord:
    """Build a record from in-memory series, with no run directory involved.

    Split out from :func:`build_record` so that every decision this adapter makes
    about an energy is exercisable without a CSV on disk.

    Parameters
    ----------
    energies : sequence of float
        Per-step mean local energy, the ``E`` column.
    spreads : sequence of float
        Per-step local-energy spread, the ``E_std`` column. Must be the same
        length as ``energies``.
    walkers_per_step : int
        Walkers per molecule per step (``num_walker_per_mol``), needed to convert
        steps into local-energy evaluations.
    step_times : sequence of float or None, optional
        Per-step wall seconds. ``None`` leaves ``wall_clock_seconds`` unset.
    allow_short_tail : bool, optional
        Forwarded to :func:`select_tail` as ``allow_below_floor``. Deliberately
        per-call-site: a short-run exemption granted once must not leak into
        other adapters or other runs.

    Returns
    -------
    BaselineRecord
        Validated record, ready to serialise.

    Raises
    ------
    AdapterError
        If the two series differ in length; if the tail window cannot be chosen;
        or if a supplied per-step time is negative, which would mean the column
        is not the monotonic timer this adapter assumes.
    """

    if len(energies) != len(spreads):
        raise AdapterError(
            f"{ENERGY_COLUMN} has {len(energies)} values but {SPREAD_COLUMN} has "
            f"{len(spreads)}; the two columns must be read from the same rows"
        )

    # Single call site, so the exemption is granted here and nowhere else.
    window = select_tail(
        len(energies),
        tail_fraction,
        min_steps=min_tail_steps,
        allow_below_floor=allow_short_tail,
    )
    energy_tail = list(energies[-window:])
    spread_tail = list(spreads[-window:])

    stderr, n_blocks = blocking_stderr(energy_tail)
    # `n_blocks` is provenance only, and a future contract change may return it
    # as None. Branch explicitly: interpolating None renders "from None blocks",
    # which reads as a measurement and raises nothing.
    blocks_text = "an unreported number of" if n_blocks is None else str(n_blocks)

    try:
        signs, monotone = sign_test(energy_tail, windows=SIGN_TEST_WINDOWS)
        drift_text = (
            f"sign pattern {signs} over {SIGN_TEST_WINDOWS} windows is monotone, so the "
            "run was still descending and this tail is an upper bound, not a plateau"
            if monotone
            else f"sign pattern {signs} over {SIGN_TEST_WINDOWS} windows is mixed, "
            "consistent with a plateau"
        )
    except AdapterError:
        drift_text = (
            f"tail of {len(energy_tail)} steps is too short to split into "
            f"{SIGN_TEST_WINDOWS} windows, so convergence was not tested"
        )

    try:
        inflation_text = f"{blocking_inflation(energy_tail):.2f}x"
    except AdapterError:
        inflation_text = "undefined"

    wall_clock = None
    if step_times is not None:
        negative = [value for value in step_times if value < 0.0]
        if negative:
            raise AdapterError(
                f"{STEP_TIME_COLUMN} carries {len(negative)} negative value(s), first "
                f"{negative[0]!r}; a per-step duration cannot be negative, so this "
                f"column is not the timer described in note '{SEMANTICS_NOTE}'"
            )
        wall_clock = float(sum(step_times))

    return BaselineRecord(
        system_id=system_id,
        code=CODE_NAME,
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=statistics.fmean(energy_tail),
        energy_stderr_hartree=stderr,
        # mean of E_std**2, i.e. the local-energy variance over walkers averaged
        # across the tail. NOT stderr**2 and NOT (E_std/sqrt(walkers))**2.
        local_energy_variance_hartree2=statistics.fmean(
            [value * value for value in spread_tail]
        ),
        steps=len(energies),
        samples=len(energies) * walkers_per_step,
        wall_clock_seconds=wall_clock,
        # No inference stage exists in this codebase, so this is never anything
        # else and is not a parameter.
        estimator="training_tail",
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=1 if n_gpus is None and device_type == "cuda" else n_gpus,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id,
        # The adapter cannot know the collector's scan root; collect() stamps it.
        run_dir=None,
        collected_at=None,
        notes=(
            f"Training-tail average of the unclipped {ENERGY_COLUMN} column over the last "
            f"{len(energy_tail)} of {len(energies)} steps; blocked standard error from "
            f"{blocks_text} blocks, blocking inflation {inflation_text}. Convergence: "
            f"{drift_text}. local_energy_variance_hartree2 is the mean of "
            f"{SPREAD_COLUMN}**2, which this code logs as the population spread of the "
            "local energies over walkers -- not a standard error, and larger than one by "
            "sqrt(walkers). "
            + (
                "wall_clock_seconds is the sum of the logger's per-step timer and covers "
                "the VMC stage ONLY: pretraining, thermalization and JIT compilation are "
                "logged to separate files and are excluded, so this understates "
                "end-to-end cost. "
                if wall_clock is not None
                else "No per-step timing was available, so wall_clock_seconds is unset "
                "rather than zero. "
            )
            + f"Column semantics, optimizer and dtype come from note '{SEMANTICS_NOTE}' "
            f"(source commit {SEMANTICS_SOURCE_COMMIT}, read {SEMANTICS_READ_DATE}), not "
            "from a runtime probe."
        ),
    )


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    walkers_per_step: int,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    ansatz: str = DEFAULT_ANSATZ,
    code_commit: str | None = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    dtype: str | None = DEFAULT_DTYPE,
    device_type: str | None = None,
    gpu_model: str | None = None,
    n_gpus: int | None = None,
    parameter_count: int | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a completed Neural Pfaffian run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory holding ``train_log.csv``.
    device_type, gpu_model : str or None, optional
        Passed in by the caller rather than parsed from a job log. This adapter
        deliberately ships no log regex: it cannot verify what the job scripts
        print, and a regex that never matches returns ``None`` indistinguishably
        from a CPU run.

    Returns
    -------
    BaselineRecord
        Validated record.

    Raises
    ------
    AdapterError
        Propagated from :func:`read_train_log` or :func:`record_from_series`.
    """

    energies, spreads, step_times = read_train_log(run_dir)
    return record_from_series(
        energies,
        spreads,
        system_id=system_id,
        walkers_per_step=walkers_per_step,
        step_times=step_times,
        tail_fraction=tail_fraction,
        min_tail_steps=min_tail_steps,
        allow_short_tail=allow_short_tail,
        ansatz=ansatz,
        code_commit=code_commit,
        optimizer=optimizer,
        dtype=dtype,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=n_gpus,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id or run_dir.name,
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
    parser.add_argument(
        "--walkers-per-step",
        type=int,
        required=True,
        help="num_walker_per_mol; required because samples cannot be derived without it",
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
    parser.add_argument("--ansatz", default=DEFAULT_ANSATZ)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--device-type", default=None, help="e.g. cuda; not inferred")
    parser.add_argument("--gpu-model", default=None, help="e.g. 'NVIDIA A100-SXM4-40GB'")
    parser.add_argument("--n-gpus", type=int, default=None)
    parser.add_argument("--parameter-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            walkers_per_step=args.walkers_per_step,
            tail_fraction=args.tail_fraction,
            min_tail_steps=args.min_tail_steps,
            allow_short_tail=args.allow_short_tail,
            ansatz=args.ansatz,
            code_commit=args.code_commit,
            device_type=args.device_type,
            gpu_model=args.gpu_model,
            n_gpus=args.n_gpus,
            parameter_count=args.parameter_count,
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
