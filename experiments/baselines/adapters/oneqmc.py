"""Translate a OneQMC (Orbformer) run directory into a :class:`BaselineRecord`.

OneQMC writes ``<workdir>/training/result.h5`` through
``oneqmc.log.H5MetricLogStream``. Neither of the existing adapters can read it,
so without this module every Orbformer measurement is stranded outside
``results.jsonl``.

Five properties of this module are deliberate and must survive any refactor.

**The column axis is a mol-batch slot, not a molecule.**
``metrics/E_loc/mean_elec`` has shape ``(logged_steps, mol_batch_size)`` and
``metrics/mol_idx`` -- same shape -- says which molecule occupied each slot at
each logged step. Taking column ``j`` as "molecule ``j``" is correct only when
``mol_batch_size == 1``, and silently mixes molecules otherwise. Every read here
gathers through ``mol_idx``; see :func:`gather_molecule`.

**Gaps are skipped, never forward-filled.** The vendor's own reader
(``oneqmc.analysis.h5_io.read_result``) scatters by ``mol_idx`` into a dense
``(steps, n_mol)`` array and then forward-fills the holes. That is right for
plotting a curve and wrong for an estimator: a repeated value is not a new
sample, so filling deflates the variance and invents steps that were never
sampled. This adapter drops the steps a molecule did not appear in and reports
the coverage in the record's notes.

**The energy estimator is a Huber M-estimate, not a plain mean.** The vendor
protocol (README, "Evaluation of the energy") is a fresh fixed-parameter
``--test`` run followed by ``oneqmc.analysis.energy.robust_mean``, which
minimizes a Huber loss with ``delta = 1`` hartree. :func:`huber_mean` computes
that same estimand without importing ``oneqmc`` or ``scipy``, by solving the
first-order condition exactly rather than by numerical minimization. On a
well-behaved series no residual reaches one hartree and the estimate *is* the
arithmetic mean; the notes say how many steps were clipped so a reader can tell
the two cases apart.

**The error bar is not the paper's.** Orbformer's published bars come from the
spread across independent chains (arXiv:2506.19960, Appendix H). That is not
reconstructible from this file: per-walker local energies are never written,
only the electron-batch mean ``E_loc/mean_elec`` and its across-walker standard
deviation ``E_loc/std_elec``. This adapter reports a blocked standard error over
the step series instead, and says so in the notes rather than letting the two
quantities share a column silently.

**Device and timing metadata are file attrs, not a job log.** The logger writes
``start_time``, ``num_gpus`` and ``gpu_type`` at construction and rewrites
``stop_time`` on *every* scalar log. So ``stop_time`` exists even for a run the
scheduler killed, and it means "time of the last logged scalar", not process
exit. :func:`metadata_from_attrs` keeps that semantics; the notes carry it.

Examples
--------
::

    uv run python -m experiments.baselines.adapters.oneqmc \\
        --run-dir path/to/orbformer-he-eval --system-id he_atom \\
        --ansatz orbformer-se --electron-batch-size 1024 \\
        --estimator inference --training-provenance finetune-from-release \\
        --checkpoint-provenance "lac.chkpt sha256:7c140f15..."
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.baselines.errors import AdapterError
from experiments.baselines.records import BaselineRecord
from experiments.baselines.statistics import (
    MIN_BLOCKS,
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

#: Electron-batch mean local energy, one row per logged step and one column per
#: mol-batch slot. The per-walker energies behind it are never written.
ENERGY_DATASET = "metrics/E_loc/mean_elec"

#: Across-walker standard deviation of the local energy, same shape. Squared and
#: averaged it gives the local-energy variance, the ansatz-quality signal.
SPREAD_DATASET = "metrics/E_loc/std_elec"

#: Slot-to-molecule map, same shape. Without it the columns are meaningless.
MOL_INDEX_DATASET = "metrics/mol_idx"

#: Huber loss width used by ``oneqmc.analysis.energy.robust_mean``, in hartree.
#: One hartree is enormous next to the step-to-step spread of a converged He
#: series, so on such a series the estimate coincides with the plain mean; the
#: value is kept because matching the vendor's estimand is the point.
HUBER_DELTA_HARTREE = 1.0

#: Trailing fraction averaged by default. Matches the DeepQMC adapter, and is
#: wanted here for an extra reason: the recommended evaluation run starts from
#: ``--discard-sampler-state``, so its early steps are still relaxing towards
#: the stationary distribution and are not part of the measurement.
DEFAULT_TAIL_FRACTION = 0.25

#: Ansatzes that are OneQMC's own work. OneQMC is Orbformer's codebase, so an
#: ``orbformer-*`` row is native. Everything else it ships -- ``psiformer``,
#: ``psiformer-new``, ``envnet`` -- is OneQMC's reimplementation of someone
#: else's method, and no claim about that method may rest on such a row. The set
#: is deliberately narrow: mislabelling a reimplementation as native overstates,
#: while the reverse only understates.
NATIVE_ANSATZES = frozenset({"orbformer-se", "orbformer-se-small"})

#: How the parameters being measured came to exist. Required on every record,
#: never inferred from a path or a step count: fine-tuning a released checkpoint
#: for a few thousand steps and reproducing a 400000-step pretraining run are
#: different scientific claims, and nothing in ``result.h5`` distinguishes them.
PROVENANCE_TIERS = ("from-scratch", "finetune-from-release")


@dataclasses.dataclass(frozen=True)
class MoleculeSeries:
    """One molecule's step series, gathered out of the slot-shaped datasets.

    Attributes
    ----------
    energies : tuple of float
        Electron-batch mean local energy, one entry per logged step in which the
        molecule was actually sampled.
    spreads : tuple of float
        Matching across-walker standard deviations.
    mol_idx : int
        Molecule index these entries belong to.
    logged_steps : int
        Rows in the file, i.e. how many logged steps the run wrote at all.
    nonfinite_dropped : int
        Steps discarded because the energy was NaN or infinite. Reported rather
        than hidden: the vendor's ``robust_mean`` filters these too, and a run
        that produced many of them is not a healthy run.
    """

    energies: tuple[float, ...]
    spreads: tuple[float, ...]
    mol_idx: int
    logged_steps: int
    nonfinite_dropped: int


def result_path(run_dir: Path) -> Path:
    """Return the HDF5 path for a run directory.

    OneQMC nests its output under ``training/``; accept either the run root or
    that subdirectory so a caller need not remember which.
    """

    direct = run_dir / RESULT_FILENAME
    if direct.is_file():
        return direct
    return run_dir / TRAINING_SUBDIR / RESULT_FILENAME


def gather_molecule(
    mol_indices: Sequence[Sequence[float]],
    energies: Sequence[Sequence[float]],
    spreads: Sequence[Sequence[float]],
    mol_idx: int,
) -> MoleculeSeries:
    """Gather one molecule's series out of slot-shaped rows.

    Pure function over nested sequences so that the slot-versus-molecule logic
    -- the part that is easy to get silently wrong -- is testable without
    ``h5py`` and without a run directory.

    Parameters
    ----------
    mol_indices : sequence of sequence of float
        ``metrics/mol_idx`` rows: which molecule sat in each slot at each step.
    energies, spreads : sequence of sequence of float
        ``metrics/E_loc/mean_elec`` and ``metrics/E_loc/std_elec`` rows.
    mol_idx : int
        Molecule to extract.

    Returns
    -------
    MoleculeSeries
        The gathered series, with coverage counts.

    Raises
    ------
    AdapterError
        If the three inputs disagree in shape; if a ``mol_idx`` entry is not a
        finite whole number; if ``mol_idx`` never appears (an empty series would
        otherwise look like a short run); if a single step carries the molecule
        in more than one slot; if a gathered spread is negative; or if every
        step was dropped as non-finite.

    Notes
    -----
    A step in which the molecule is absent is skipped. It is *not* forward
    filled, unlike ``oneqmc.analysis.h5_io.read_result``: repeating the previous
    value adds no information but does add a sample, which narrows every
    variance estimate downstream.

    A step carrying the molecule in several slots raises rather than averaging.
    Those slots are separate batch means whose relative weight depends on how
    the sampler assigned walkers, and guessing a weighting is exactly the kind
    of silent choice this adapter exists to prevent.
    """

    if len(energies) != len(mol_indices) or len(spreads) != len(mol_indices):
        raise AdapterError(
            f"shape mismatch: {len(mol_indices)} mol_idx rows, {len(energies)} energy "
            f"rows, {len(spreads)} spread rows"
        )

    kept_energies: list[float] = []
    kept_spreads: list[float] = []
    seen = False
    nonfinite = 0

    for step, slots in enumerate(mol_indices):
        if len(energies[step]) != len(slots) or len(spreads[step]) != len(slots):
            raise AdapterError(
                f"step {step} has {len(slots)} mol_idx slots but "
                f"{len(energies[step])} energy and {len(spreads[step])} spread slots"
            )
        hits = [
            column
            for column, value in enumerate(slots)
            if _exact_slot_index(value, step, column) == mol_idx
        ]
        if not hits:
            continue
        seen = True
        if len(hits) > 1:
            raise AdapterError(
                f"step {step} carries molecule {mol_idx} in {len(hits)} slots; this "
                "adapter refuses to choose a weighting between them"
            )
        energy = float(energies[step][hits[0]])
        spread = float(spreads[step][hits[0]])
        if not _is_finite(energy) or not _is_finite(spread):
            nonfinite += 1
            continue
        # Checked here and not only at the variance sum: `std_elec` is an
        # across-walker standard deviation, so a negative value is a corrupt
        # dataset rather than an unusually wide one. Squaring it downstream
        # would turn the corruption into an entirely plausible positive
        # variance, which is the one outcome no reader could detect.
        _reject_negative_spread(spread, f"step {step}")
        kept_energies.append(energy)
        kept_spreads.append(spread)

    if not seen:
        raise AdapterError(
            f"molecule index {mol_idx} never appears in {MOL_INDEX_DATASET}; "
            "check --mol-idx against the dataset the run was launched on"
        )
    if not kept_energies:
        raise AdapterError(
            f"every one of the {nonfinite} steps carrying molecule {mol_idx} had a "
            "non-finite energy or spread"
        )

    return MoleculeSeries(
        energies=tuple(kept_energies),
        spreads=tuple(kept_spreads),
        mol_idx=mol_idx,
        logged_steps=len(mol_indices),
        nonfinite_dropped=nonfinite,
    )


def read_series(run_dir: Path, mol_idx: int = 0) -> MoleculeSeries:
    """Read one molecule's step series from a OneQMC run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run root, or its ``training`` subdirectory.
    mol_idx : int, optional
        Molecule index to extract, matching ``metrics/mol_idx``.

    Returns
    -------
    MoleculeSeries
        Gathered series; see :func:`gather_molecule` for the gather rules.

    Raises
    ------
    AdapterError
        If ``h5py`` is unavailable, the file or any required dataset is missing,
        or the file holds no rows.

    Notes
    -----
    The file is opened with ``swmr=True, libver="v110"``, the same way the
    vendor's reader opens it and the same way the writer created it. That
    matters for a run the scheduler killed: the writer dies without closing, so
    the superblock stays flagged open-for-write and a plain open fails with
    *"file is already open for write"*. SWMR reads it as-is. ``h5clear -s``
    would also clear the flag, but it writes to the file, and run data is not to
    be modified.

    A successful read does not prove the producing job finished -- a live writer
    looks identical. Ask the scheduler for that.
    """

    # Existence is checked BEFORE h5py is imported, and the order is the whole
    # point. When a caller points at the wrong directory, "no result.h5 under
    # <dir>" is the actionable diagnosis and "you need h5py" is a distraction
    # that sends them to fix an environment that was never the problem. The
    # reverse order also made this function's error depend on whether the
    # interpreter happened to carry h5py, which is how it reached a cluster
    # verification green locally and red in the project venv.
    path = result_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {RESULT_FILENAME} under {run_dir}")

    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        raise AdapterError(
            "reading OneQMC output needs h5py, which is not a TPEN dependency; run "
            "this adapter with the OneQMC virtualenv that already provides it"
        ) from error

    try:
        with h5py.File(path, "r", swmr=True, libver="v110") as handle:
            rows = {}
            for dataset in (MOL_INDEX_DATASET, ENERGY_DATASET, SPREAD_DATASET):
                if dataset not in handle:
                    raise AdapterError(f"{path} has no '{dataset}' dataset")
                rows[dataset] = [
                    [float(value) for value in _as_row(row)] for row in handle[dataset][:]
                ]
    except OSError as error:
        raise AdapterError(f"cannot open {path}: {error}") from error

    if not rows[MOL_INDEX_DATASET]:
        raise AdapterError(f"{path} has no logged steps")

    # Attrs are provenance, not measurement, and reach the caller through
    # read_attrs instead of riding along here; only the series feeds the
    # estimator.
    return gather_molecule(
        rows[MOL_INDEX_DATASET],
        rows[ENERGY_DATASET],
        rows[SPREAD_DATASET],
        mol_idx,
    )


def read_attrs(run_dir: Path) -> dict[str, Any]:
    """Return the HDF5 file attributes of a OneQMC run.

    Separate from :func:`read_series` so a caller can inspect provenance without
    reading the trace.
    """

    # Same ordering as read_series, for the same reason: a missing file is the
    # more actionable diagnosis, and it must not depend on whether h5py is
    # installed.
    path = result_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {RESULT_FILENAME} under {run_dir}")

    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        raise AdapterError("reading OneQMC output needs h5py") from error
    try:
        with h5py.File(path, "r", swmr=True, libver="v110") as handle:
            return dict(handle.attrs)
    except OSError as error:
        raise AdapterError(f"cannot open {path}: {error}") from error


def metadata_from_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return device and wall-clock metadata from HDF5 file attributes.

    Parameters
    ----------
    attrs : mapping
        The file's attribute mapping. Values may be ``bytes`` or ``str``
        depending on how ``h5py`` stored them.

    Returns
    -------
    dict
        ``device_type``, ``gpu_model``, ``n_gpus`` and ``wall_clock_seconds``,
        each ``None`` when the corresponding attribute is ABSENT. Nothing is
        guessed: a missing ``stop_time`` yields no duration rather than a
        fabricated one, and a device kind this function does not recognise
        yields no ``device_type`` rather than ``"cuda"``.

    Raises
    ------
    AdapterError
        If an attribute is PRESENT but unusable: a non-integer ``num_gpus``, an
        unparseable or mixed-offset timestamp pair, a ``stop_time`` before
        ``start_time``, or a CPU ``gpu_type`` contradicted by a positive
        ``num_gpus``. Absent and corrupt are different conditions and a record
        that renders them identically hides schema damage as missing metadata.

    Notes
    -----
    ``gpu_type`` is JAX's ``device_kind``, so ``"cpu"`` there is a genuine CPU
    run rather than missing data, and it is reported as such.

    ``start_time``/``stop_time`` are ``str(datetime.now())`` -- naive local time,
    no zone. The duration between them measures the metric logger's lifetime:
    it starts when the logger is constructed, which is after job startup, and
    ends at the last logged scalar, which for a killed run is before the job
    died. It is therefore a lower bound on job wall clock, and the record's
    notes say so.
    """

    gpu_type = _decode(attrs.get("gpu_type"))
    device_type: str | None
    if gpu_type is None:
        device_type = None
    elif gpu_type.strip().lower() == "cpu":
        device_type, gpu_type = "cpu", None
    elif _is_cuda_device_kind(gpu_type):
        device_type = "cuda"
    else:
        # Not "cuda". `gpu_type` is JAX's `device_kind`, which also names TPU,
        # ROCm and Metal devices, and an unrecognised or blank string is not
        # evidence of NVIDIA hardware. Mapping every non-CPU string to "cuda"
        # made `device_type` a field the file did not support. Left as None for
        # the caller to state through --device-type, the same way the neural
        # Pfaffian adapter takes it; `gpu_model` still carries the raw kind, so
        # nothing is lost.
        device_type = None

    raw_n_gpus = attrs.get("num_gpus")
    n_gpus: int | None
    if raw_n_gpus is None:
        n_gpus = None
    else:
        # Present but unparseable is NOT the same as absent. Coercing it to None
        # published a record that read as "the run did not say", when in fact the
        # run said something unusable -- schema damage, silently laundered into a
        # missing field.
        try:
            n_gpus = int(raw_n_gpus)
        except (TypeError, ValueError) as error:
            raise AdapterError(
                f"attrs num_gpus is {raw_n_gpus!r}, which is not an integer count; "
                "the attribute is present but unusable, which is a corrupt file rather "
                "than a run that omitted it"
            ) from error
    if device_type == "cpu" and n_gpus is not None and n_gpus > 0:
        # The two attrs contradict each other, and the record would have carried
        # both: device_type="cpu" beside n_gpus=1. Whichever is right, emitting
        # the pair asserts something no run can be.
        raise AdapterError(
            f"attrs say gpu_type is a CPU but num_gpus is {n_gpus}; these contradict, "
            "so the device metadata cannot be reported as measured"
        )

    start, stop = _decode(attrs.get("start_time")), _decode(attrs.get("stop_time"))
    wall_clock: float | None = None
    if start is not None and stop is not None:
        try:
            # TypeError, not only ValueError. `datetime.fromisoformat` parses an
            # offset-aware and an offset-naive stamp equally happily, and it is
            # the SUBTRACTION that then raises TypeError -- which no
            # `except ValueError` catches, so a mixed pair escaped this function
            # as an uncaught TypeError from inside an adapter whose contract is
            # AdapterError.
            wall_clock = (
                datetime.fromisoformat(stop) - datetime.fromisoformat(start)
            ).total_seconds()
        except (TypeError, ValueError) as error:
            raise AdapterError(
                f"attrs carry start_time {start!r} and stop_time {stop!r}, which cannot "
                f"be differenced ({error}); both are present, so this is a malformed "
                "timestamp pair rather than missing timing"
            ) from error
        if wall_clock < 0.0:
            # Cannot happen from a single writer. Previously dropped to None,
            # which reported a corrupt pair as an absent one.
            raise AdapterError(
                f"attrs stop_time {stop!r} precedes start_time {start!r}, giving a "
                f"duration of {wall_clock} seconds; the pair is corrupt"
            )

    return {
        "device_type": device_type,
        "gpu_model": gpu_type,
        "n_gpus": n_gpus,
        "wall_clock_seconds": wall_clock,
    }


def huber_mean(
    values: Sequence[float], delta: float = HUBER_DELTA_HARTREE
) -> tuple[float, int]:
    """Return the Huber M-estimate of a series, and how many points it clipped.

    This is the estimand of ``oneqmc.analysis.energy.robust_mean``: the
    minimizer of ``sum(huber(v - mu, delta))``. The vendor reaches it with
    ``scipy.optimize.minimize`` from the arithmetic mean; this implementation
    solves the first-order condition ``sum(clip(v - mu, -delta, +delta)) = 0``
    by bisection instead, which needs neither ``scipy`` nor a convergence
    tolerance argument and cannot stop early on a flat gradient.

    Parameters
    ----------
    values : sequence of float
        The series, normally an already-selected estimator window.
    delta : float, optional
        Huber width in hartree. Residuals beyond it contribute linearly, which
        is what bounds an outlier's pull on the estimate.

    Returns
    -------
    tuple of (float, int)
        The estimate, and the number of points whose residual exceeded
        ``delta``. **A count of zero means the estimate is exactly the
        arithmetic mean**, because inside ``delta`` the objective is quadratic;
        reporting it is how a reader tells a robust estimate from a plain
        average.

    Raises
    ------
    AdapterError
        If ``values`` is empty or ``delta`` is not positive.

    Notes
    -----
    The objective's derivative is non-increasing in ``mu`` and changes sign
    inside ``[min(values), max(values)]``, so bisection is exact to machine
    precision in a fixed 100 iterations -- no tolerance to tune and no
    dependence on a starting point.
    """

    data = [float(value) for value in values]
    if not data:
        raise AdapterError("huber mean needs at least one value")
    if not delta > 0.0:
        raise AdapterError(f"huber delta must be positive, got {delta}")

    low, high = min(data), max(data)
    if low == high:
        return low, 0

    def gradient(mu: float) -> float:
        # d/d mu of the loss, negated: non-increasing in mu, positive at `low`
        # and negative at `high`, so the root is bracketed.
        return sum(min(max(value - mu, -delta), delta) for value in data)

    for _ in range(100):
        middle = 0.5 * (low + high)
        if gradient(middle) > 0.0:
            low = middle
        else:
            high = middle

    estimate = 0.5 * (low + high)
    clipped = sum(1 for value in data if abs(value - estimate) > delta)
    return estimate, clipped


def record_from_series(
    energies: Sequence[float],
    spreads: Sequence[float],
    *,
    system_id: str,
    electron_batch_size: int,
    ansatz: str,
    estimator: str,
    training_provenance: str,
    checkpoint_provenance: str | None = None,
    logged_steps: int,
    nonfinite_dropped: int,
    mol_idx: int,
    metric_logger_period: int,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    run_id: str,
    code_commit: str | None = None,
    optimizer: str,
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    device_type: str | None = None,
    gpu_model: str | None = None,
    n_gpus: int | None = None,
    wall_clock_seconds: float | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build a record from an already-gathered series.

    Separated from :func:`build_record` so that every decision this adapter
    makes -- window selection, the Huber estimate, the error bar, the
    provenance text -- is exercisable without an HDF5 file, and therefore
    without ``h5py`` installed.

    Parameters
    ----------
    energies, spreads : sequence of float
        One electron-batch mean local energy and one across-walker standard
        deviation per logged step that carried this molecule.
    electron_batch_size : int
        Walkers per step, needed for the ``samples`` denominator. Required, not
        inferred: the file records the batch mean, not the batch size.
    ansatz : str
        OneQMC ansatz name, e.g. ``"orbformer-se"``. Required and never
        inferred; a run directory does not reveal it, and it decides whether the
        row is native code or a reimplementation.
    estimator : str
        ``"inference"`` for a fixed-parameter ``--test`` run, which is the
        protocol the vendor recommends, or ``"training_tail"`` for an average
        over the end of a fine-tuning run. Required: the two are different
        quantities and the file looks the same either way.
    training_provenance : str
        One of :data:`PROVENANCE_TIERS`. ``"finetune-from-release"`` additionally
        requires ``checkpoint_provenance``.
    checkpoint_provenance : str or None, optional
        Identity of the released checkpoint -- URL, hash, or both. Mandatory for
        a fine-tuned row: without it the row cannot be reproduced and cannot be
        audited against the release it claims to start from.
    logged_steps : int
        Rows the run wrote, which exceeds ``len(energies)`` when other molecules
        shared the mol batch. Required, not defaulted to ``len(energies)``: that
        default silently asserted full coverage, and ``steps`` is computed from
        this count.
    nonfinite_dropped : int
        Rows that carried this molecule but whose energy or spread was
        non-finite. Required for the same reason: a defaulted ``0`` asserted that
        nothing was dropped. Counted into ``samples``, because a row that
        produced a NaN was still evaluated.
    mol_idx : int
        Molecule index these series were gathered for. Appears in the notes
        beside ``system_id``, whose correspondence to it is an operator
        assertion that no file content can confirm.
    metric_logger_period : int
        ``--metric-logger-period`` of the run. Logged rows are multiplied by it
        to recover optimizer/evaluation steps, since walkers are propagated on
        unlogged steps too. Required: ``result.h5`` holds no step or iteration
        index anywhere, so this is not recoverable from the file, and it scales
        both ``steps`` and ``samples``.
    optimizer : str
        Optimizer the run used, e.g. ``"kfac"``. Required for the same reason:
        the file does not record it, so a default would be an unsourced
        provenance claim.
    note : str or None, optional
        Operator caveat, appended to the generated notes. Never replaces any
        generated sentence.

    Returns
    -------
    BaselineRecord
        Validated record carrying ``code="oneqmc"``.

    Raises
    ------
    AdapterError
        On a shape mismatch, an unknown provenance tier, a fine-tuned row with
        no checkpoint provenance, a blank ``note`` or ``optimizer``, a
        non-positive ``electron_batch_size`` or ``metric_logger_period``, a
        negative spread or drop count, a ``logged_steps`` smaller than the rows
        gathered from it, a window that cannot be selected, or a window below the
        blocking floor -- which is refused rather than answered with an
        uncorrected naive error bar.
    """

    energy_series = [float(value) for value in energies]
    spread_series = [float(value) for value in spreads]
    if len(energy_series) != len(spread_series):
        raise AdapterError(
            f"{len(energy_series)} energies but {len(spread_series)} spreads; the two "
            "datasets must be gathered together"
        )
    if training_provenance not in PROVENANCE_TIERS:
        raise AdapterError(
            f"training_provenance must be one of {PROVENANCE_TIERS}, got "
            f"{training_provenance!r}"
        )
    if training_provenance == "finetune-from-release" and not (
        checkpoint_provenance or ""
    ).strip():
        raise AdapterError(
            "a fine-tuned row requires checkpoint_provenance: a record that cannot "
            "name the released checkpoint it started from is not reproducible"
        )
    if note is not None and not note.strip():
        raise AdapterError("note must be non-empty when given; a caveat that vanishes is worse than none")
    if electron_batch_size <= 0:
        raise AdapterError(f"electron_batch_size must be positive, got {electron_batch_size}")
    if metric_logger_period <= 0:
        raise AdapterError(f"metric_logger_period must be positive, got {metric_logger_period}")
    if not optimizer.strip():
        raise AdapterError(
            "optimizer must be named: it is not recorded anywhere in result.h5, so a "
            "blank value is an unsourced claim rather than a missing field"
        )
    # Checked on the direct API too, not only in `gather_molecule`: this function
    # is the supported entry point for a caller that gathered its own series,
    # and the variance below squares every one of these values.
    for position, spread in enumerate(spread_series):
        _reject_negative_spread(spread, f"series position {position}")
    if nonfinite_dropped < 0:
        raise AdapterError(
            f"nonfinite_dropped must not be negative, got {nonfinite_dropped}; a "
            "negative drop count would overstate this molecule's coverage"
        )
    if logged_steps < len(energy_series) + nonfinite_dropped:
        raise AdapterError(
            f"logged_steps={logged_steps} is smaller than the {len(energy_series)} "
            f"retained plus {nonfinite_dropped} dropped rows that carried this "
            "molecule; the file cannot hold fewer rows than were gathered from it, so "
            "one of the two counts is wrong"
        )

    window = select_tail(
        len(energy_series),
        tail_fraction,
        min_steps=min_tail_steps,
        allow_below_floor=allow_short_tail,
    )
    tail = energy_series[-window:]
    tail_spreads = spread_series[-window:]

    energy, clipped = huber_mean(tail)

    # The window floor and the BLOCKING floor are different floors, and only the
    # first is opt-outable. `allow_short_tail` is forwarded rather than withheld,
    # so one flag does not mean two things at two call sites -- but the reply is
    # then checked, because the opt-in buys the uncorrected naive bar, not a
    # licence to publish it. This is the stance deepqmc and ferminet already
    # take; see `deepqmc.py` and `ferminet.py` around their `blocking_stderr`
    # calls.
    #
    # What this replaces: `block_floor = min(MIN_BLOCKS, len(tail))`, which
    # lowered the floor to the raw sample count so that blocking's first level
    # always ran. On a short window that made the blocked-over-naive ratio
    # exactly 1.00 by construction -- one level, so numerator and denominator
    # were the same quantity -- and the record then printed "1.00x" as an
    # autocorrelation diagnostic although no blocking level had assessed
    # autocorrelation at all. A reader cannot tell that number from a genuine
    # measurement of no autocorrelation, so it was false reassurance rather than
    # a wide interval. Refusing is the only honest option available here.
    try:
        stderr, blocks = blocking_stderr(tail, allow_below_floor=allow_short_tail)
    except AdapterError as error:
        # A zero-variance window is the other route to a zero bar, and it is not
        # a measurement: every step in the window carries the identical value,
        # which means the sampler stopped moving or the series was
        # forward-filled upstream. Refusing is the same stance this module takes
        # on forward-filling in `gather_molecule`. Caught rather than allowed to
        # surface raw so the message names the window, and so this adapter
        # behaves the same whether `blocking_stderr` signals the degenerate case
        # by raising or by returning zero.
        raise AdapterError(
            f"zero-variance window of {len(tail)} steps: every step carries the "
            f"identical energy, so no error bar can be estimated from it ({error})"
        ) from error
    if stderr <= 0.0:
        raise AdapterError(
            f"zero-variance window of {len(tail)} steps: blocking returned a "
            f"non-positive error bar ({stderr!r}), which the record schema would "
            "accept as non-negative and publish as infinite precision"
        )
    # A block count of `None` means no blocking level ran, so autocorrelation is
    # unassessed and the bar is the uncorrected naive one. Reached under
    # `allow_short_tail`, which is exactly the case the lowered floor used to
    # paper over: `--allow-short-tail` widens the estimator WINDOW, and it is not
    # also a licence to publish an unblocked bar as though it had been blocked.
    if blocks is None:
        raise AdapterError(
            f"the {len(tail)}-step window is below the {MIN_BLOCKS}-block minimum for "
            "blocking, so no blocking level ran and this run's autocorrelation is "
            "UNASSESSED; refusing to emit a record whose error bar is an uncorrected "
            "naive estimate that understates the uncertainty. Raise --tail-fraction, "
            f"lower --metric-logger-period, or log at least {MIN_BLOCKS} steps in the "
            "window"
        )
    # `std_elec` is the across-walker spread of the local energy, so its square
    # is the local-energy variance itself -- not the variance of the mean.
    local_variance = statistics.fmean(spread**2 for spread in tail_spreads)

    # Convergence and autocorrelation are reported, never used to alter the
    # number. A verdict that could change the estimate would be a selection
    # rule; this one only describes.
    try:
        signs, monotone = sign_test(tail, windows=SIGN_TEST_WINDOWS)
        verdict = (
            f"windowed sign test over {SIGN_TEST_WINDOWS} windows gives '{signs}': "
            + (
                "MONOTONE, so the series was still drifting at the end of this trace "
                "and the run may not have converged"
                if monotone
                else "not monotone, consistent with noise rather than drift"
            )
        )
    except AdapterError:
        verdict = (
            f"tail of {len(tail)} steps is too short for a {SIGN_TEST_WINDOWS}-window "
            "sign test, so convergence is UNASSESSED"
        )

    # No try/except here on purpose: `blocking_inflation` raises only for a
    # window below two values or with zero variance, and both are already
    # refused above. Swallowing an exception that cannot fire would hide a real
    # regression behind the word "undefined".
    #
    # The VALUE is branched on before formatting, though. Under
    # `allow_below_floor` this returns None rather than raising, and
    # `f"{None:.2f}x"` is a TypeError that `except AdapterError` would not catch,
    # while `f"{None}x"` would render the word "None" into an emitted record.
    # Unreachable while the block-count refusal above precedes it -- both read
    # the same tail against the same floor -- and kept so that reordering the two
    # cannot resurrect a formatted None.
    inflation_factor = blocking_inflation(tail, allow_below_floor=allow_short_tail)
    if inflation_factor is None:
        raise AdapterError(
            f"autocorrelation inflation is unmeasurable for a window of {len(tail)} "
            "steps, so no diagnostic can be reported for it"
        )
    inflation = f"{inflation_factor:.2f}x"

    # `steps` is what records.py defines it to be -- optimizer steps TAKEN -- so it
    # is the file's own row count times the logger period, not the rows retained
    # for this molecule. Those differ whenever coverage is sparse (a mol-batch run
    # need not carry this molecule at every step) or a row was dropped as
    # non-finite, and the retained-row form understated the work the run actually
    # did. The two coincide only at full coverage with no drops, which is why the
    # difference is invisible on a single-molecule fixture.
    steps = logged_steps * metric_logger_period
    # `samples` is cumulative local-energy evaluations, and only the rows that
    # carried THIS molecule produced any for this row. Dropped rows are counted:
    # a non-finite energy was still evaluated, and hiding the cost of a failed
    # evaluation flatters the run.
    covered_rows = len(energy_series) + nonfinite_dropped
    samples = covered_rows * metric_logger_period * electron_batch_size

    short = (
        " Window is BELOW the standard minimum, so this estimate is provisional."
        if len(tail) < min_tail_steps
        else ""
    )
    estimator_text = (
        f"Fixed-parameter inference pass over the last {len(tail)} of "
        f"{len(energy_series)} logged steps."
        if estimator == "inference"
        else f"Training-tail average over the last {len(tail)} of "
        f"{len(energy_series)} logged steps."
    ) + short

    huber_text = (
        f"Energy is the Huber M-estimate matching oneqmc robust_mean at delta="
        f"{HUBER_DELTA_HARTREE} hartree; "
        + (
            "no step in the window exceeded delta, so it equals the arithmetic mean"
            if clipped == 0
            else f"{clipped} of {len(tail)} window steps exceeded delta and were clipped"
        )
    )

    coverage = (
        f"Gathered through {MOL_INDEX_DATASET} for molecule index {mol_idx}: "
        f"{len(energy_series)} of {logged_steps} logged steps carried it, gaps skipped "
        "rather than forward-filled"
        + (f", {nonfinite_dropped} step(s) dropped as non-finite" if nonfinite_dropped else "")
        + "."
    )
    coverage += (
        f" Efficiency denominators: `steps` is the run's optimizer steps, {logged_steps} "
        f"logged rows times logger period {metric_logger_period}, so it counts the whole "
        "run's work and is NOT reduced to the rows that carried this molecule; "
        f"`samples` is this molecule's own local-energy evaluations, {covered_rows} of "
        f"those rows times the period times electron batch {electron_batch_size}, "
        "because a mol-batch step divides its walkers among several molecules."
    )
    if metric_logger_period != 1:
        coverage += (
            f" Logger period is {metric_logger_period}, so both denominators scale "
            "logged rows up to unlogged steps, which assumes this molecule also "
            "occupied a slot in the steps between rows; result.h5 does not record that."
        )
    # The field changes character under this adapter and the record must say so.
    # `energy_hartree` and the error bar are measured from datasets in the file;
    # `steps` is not, because result.h5 carries no step or iteration index of any
    # kind, so the logger period that converts logged rows into optimizer steps
    # can only come from the operator. A reader comparing this row's efficiency
    # denominator against another code's has to know which of the two they are
    # looking at.
    coverage += (
        f" NOTE ON PROVENANCE OF `steps`: it is DECLARED, not measured. result.h5 "
        "records no step or iteration index, so the logger period above is an "
        "operator-supplied value rather than something read from the file, and both "
        "efficiency denominators rest on it. The energy and its error bar, by "
        "contrast, are computed from the file's own datasets."
    )
    # F1: nothing in result.h5 names the system, so this pairing is an operator
    # assertion and the record says so out loud rather than implying the adapter
    # checked it.
    pairing = (
        f" The pairing of system_id '{system_id}' with molecule index {mol_idx} is "
        "ASSERTED BY THE OPERATOR and is NOT verifiable from result.h5, which records "
        "which molecule occupied each mol-batch slot but never names a system. Both "
        "values are explicit arguments with no default for that reason."
    )

    bar_text = (
        "Error bar is a blocked (Flyvbjerg-Petersen) standard error over the step "
        f"series, autocorrelation inflation {inflation} over the naive estimate. This "
        "is NOT the across-chain variance the Orbformer paper reports (arXiv:2506.19960 "
        "Appendix H): per-walker local energies are not logged, so that quantity is not "
        "reconstructible from result.h5."
    )

    native = (
        " OneQMC is Orbformer's own codebase, so this ansatz is native rather than a "
        "reimplementation."
        if ansatz in NATIVE_ANSATZES
        else f" '{ansatz}' here is OneQMC's REIMPLEMENTATION, not the {ansatz} authors' "
        "own code; no claim about that method may rest on this row."
    )

    if training_provenance == "finetune-from-release":
        provenance_text = (
            f" PROVENANCE: fine-tuned from a released checkpoint ({checkpoint_provenance}). "
            "This is NOT a from-scratch reproduction of the paper's training run, whose "
            "pretraining alone is ~11200 A100-hours over 800000 steps."
        )
    else:
        provenance_text = " PROVENANCE: trained from scratch in this program."

    clock_text = (
        " Wall clock is the metric logger's start_time-to-stop_time span, a lower bound "
        "on job wall clock: it excludes job startup before the logger exists and ends at "
        "the last logged scalar, which for a killed run precedes the job's death."
        if wall_clock_seconds is not None
        else ""
    )

    notes = (
        f"{estimator_text} {huber_text}. {bar_text} {coverage}{pairing} {verdict}."
        f"{native}{provenance_text}{clock_text}"
    )
    if note is not None:
        notes = f"{notes} {note.strip()}"

    return BaselineRecord(
        system_id=system_id,
        code="oneqmc",
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=energy,
        energy_stderr_hartree=stderr,
        local_energy_variance_hartree2=local_variance,
        steps=steps,
        samples=samples,
        wall_clock_seconds=wall_clock_seconds,
        estimator=estimator,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=n_gpus,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id,
        # The adapter cannot know the collector's scan root. Leave this blank so
        # collect() stamps the collision-free path relative to that root.
        run_dir=None,
        collected_at=None,
        notes=notes,
    )


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    electron_batch_size: int,
    ansatz: str,
    estimator: str,
    training_provenance: str,
    checkpoint_provenance: str | None = None,
    mol_idx: int,
    metric_logger_period: int,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    code_commit: str | None = None,
    optimizer: str,
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    device_type: str | None = None,
    run_id: str | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a OneQMC run directory.

    Reads the gathered series and the device/timing attrs from ``result.h5``,
    then delegates every decision to :func:`record_from_series`.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run root, or its ``training`` subdirectory.
    system_id, electron_batch_size, ansatz, estimator, training_provenance, checkpoint_provenance, mol_idx, metric_logger_period, tail_fraction, min_tail_steps, allow_short_tail, code_commit, optimizer, dtype, seed, parameter_count, note
        Forwarded unchanged to :func:`record_from_series`; see its parameter
        list. ``mol_idx``, ``metric_logger_period`` and ``optimizer`` are
        required here too, because nothing in the run directory supplies them.
    device_type : str or None, optional
        Device kind to record when ``gpu_type`` in the file is one this adapter
        does not recognise. Refused when it contradicts a kind the file did
        state.
    run_id : str or None, optional
        Record identifier; defaults to ``run_dir.name``.

    Returns
    -------
    BaselineRecord
        Validated record carrying ``code="oneqmc"``.

    Raises
    ------
    AdapterError
        From the read helpers, from :func:`record_from_series`, or when
        ``device_type`` contradicts the file.
    """

    series = read_series(run_dir, mol_idx=mol_idx)
    metadata = metadata_from_attrs(read_attrs(run_dir))
    if device_type is not None:
        # `metadata_from_attrs` reports a device type only for a device kind it
        # recognises, so this fills the gap for TPU/ROCm/Metal or an unknown
        # string rather than overriding a reading. A caller assertion that
        # CONTRADICTS the file is refused: the point of taking the value from the
        # operator is to cover what the file cannot say, not to overwrite what it
        # did say.
        recorded = metadata.get("device_type")
        if recorded is not None and recorded != device_type:
            raise AdapterError(
                f"device_type={device_type!r} contradicts the file's own gpu_type, "
                f"which reads as {recorded!r}; refusing to replace measured metadata "
                "with an assertion"
            )
        metadata["device_type"] = device_type

    return record_from_series(
        series.energies,
        series.spreads,
        system_id=system_id,
        electron_batch_size=electron_batch_size,
        ansatz=ansatz,
        estimator=estimator,
        training_provenance=training_provenance,
        checkpoint_provenance=checkpoint_provenance,
        logged_steps=series.logged_steps,
        nonfinite_dropped=series.nonfinite_dropped,
        mol_idx=series.mol_idx,
        metric_logger_period=metric_logger_period,
        tail_fraction=tail_fraction,
        min_tail_steps=min_tail_steps,
        allow_short_tail=allow_short_tail,
        run_id=run_id or run_dir.name,
        code_commit=code_commit,
        optimizer=optimizer,
        dtype=dtype,
        seed=seed,
        parameter_count=parameter_count,
        note=note,
        **metadata,
    )


def write_record(record: BaselineRecord, run_dir: Path) -> Path:
    """Write ``baseline_record.json`` into ``run_dir`` and return its path."""

    path = run_dir / RECORD_FILENAME
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _as_row(row: Any) -> Sequence[Any]:
    """Return a 1-D row as a sequence, wrapping a scalar into a one-slot row."""

    if hasattr(row, "__len__") and not isinstance(row, (str, bytes)):
        return row
    return [row]


def _decode(value: Any) -> str | None:
    """Return an HDF5 string attribute as ``str``, or None if absent."""

    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


#: Lowercased prefixes of JAX ``device_kind`` strings that denote CUDA hardware.
#: JAX reports the marketing name, e.g. ``"NVIDIA A100-SXM4-40GB"`` or
#: ``"Tesla V100-SXM2-16GB"``, never the literal word ``"cuda"`` -- which is
#: accepted anyway, for a caller that normalised the string before storing it.
CUDA_DEVICE_KINDS = ("nvidia", "tesla", "quadro", "geforce", "titan", "cuda")


def _is_cuda_device_kind(gpu_type: str) -> bool:
    """Return True when a JAX ``device_kind`` string names CUDA hardware.

    Parameters
    ----------
    gpu_type : str
        The ``gpu_type`` file attribute, as decoded.

    Returns
    -------
    bool
        True for a recognised NVIDIA-family device kind.

    Notes
    -----
    Recognition, not a fallback. The alternative -- treat every non-CPU string
    as CUDA -- labelled TPU, ROCm, Metal and blank device kinds as CUDA, which
    is a claim about hardware made from a string that contradicted it.
    """

    kind = gpu_type.strip().lower()
    return bool(kind) and kind.startswith(CUDA_DEVICE_KINDS)


def _exact_slot_index(value: Any, step: int, column: int) -> int:
    """Return a ``metrics/mol_idx`` entry as an exact molecule index.

    Parameters
    ----------
    value : Any
        One entry of the ``mol_idx`` dataset.
    step, column : int
        Position of the entry, for the error message only.

    Returns
    -------
    int
        The entry, when it is a finite whole number.

    Raises
    ------
    AdapterError
        If the entry is not numeric, not finite, or not integral.

    Notes
    -----
    ``int(value)`` truncates. A corrupt ``2.7`` would silently become ``2`` and
    select molecule 2's slot, publishing one molecule's energy under another
    molecule's label -- the exact failure this adapter exists to prevent, but
    arriving through the dataset instead of through column position. A
    non-finite entry would raise ``ValueError``/``OverflowError`` out of ``int``
    rather than an :class:`AdapterError`, so it is converted here too.
    """

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(
            f"{MOL_INDEX_DATASET} step {step} slot {column} holds {value!r}, which is "
            "not a number; the molecule index cannot be read from it"
        ) from error
    if not math.isfinite(number):
        raise AdapterError(
            f"{MOL_INDEX_DATASET} step {step} slot {column} is {number!r}; a "
            "non-finite molecule index means the dataset is corrupt"
        )
    if number != int(number):
        raise AdapterError(
            f"{MOL_INDEX_DATASET} step {step} slot {column} is {number!r}, which is not "
            "a whole number; truncating it would select a molecule the run never put "
            "in that slot"
        )
    return int(number)


def _reject_negative_spread(spread: float, where: str) -> None:
    """Raise if an ``E_loc/std_elec`` value is negative.

    Parameters
    ----------
    spread : float
        A single across-walker standard deviation.
    where : str
        Location text for the message, e.g. ``"step 7"``.

    Raises
    ------
    AdapterError
        If ``spread`` is negative.

    Notes
    -----
    Finiteness alone is not enough. The variance this adapter publishes is the
    mean of ``spread ** 2``, and squaring destroys the sign, so a negative
    standard deviation is laundered into a perfectly plausible positive
    variance that no downstream reader can question.
    """

    if spread < 0.0:
        raise AdapterError(
            f"{SPREAD_DATASET} at {where} is {spread!r}; an across-walker standard "
            "deviation cannot be negative, and squaring it would publish the "
            "corruption as a plausible variance"
        )


def _is_finite(value: float) -> bool:
    """Return True for a real, finite float."""

    return math.isfinite(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument(
        "--electron-batch-size",
        type=int,
        required=True,
        help="walkers per step; result.h5 records the batch mean, not the batch size",
    )
    # Required, not defaulted. Nothing in a OneQMC run directory reveals the
    # ansatz, and the value decides whether the row is native code or a
    # reimplementation of someone else's method.
    parser.add_argument("--ansatz", required=True)
    parser.add_argument(
        "--estimator",
        choices=("training_tail", "inference"),
        required=True,
        help="'inference' for a --test run, the protocol the vendor recommends",
    )
    parser.add_argument(
        "--training-provenance",
        choices=PROVENANCE_TIERS,
        required=True,
        help="how the measured parameters came to exist; never inferred",
    )
    parser.add_argument(
        "--checkpoint-provenance",
        default=None,
        help="URL and/or hash of the released checkpoint; required when fine-tuned",
    )
    # Required, not defaulted to 0. `--system-id` and `--mol-idx` are
    # independent, and nothing in result.h5 names a system, so a defaulted index
    # let an operator ask for one system, omit the index, and publish molecule
    # 0's energy under that system's label. Making the index explicit does not
    # let the adapter VERIFY the pairing -- no file content can -- but it removes
    # the case where the operator never stated it, and the record's notes say
    # plainly that the pairing is an operator assertion.
    parser.add_argument(
        "--mol-idx",
        type=int,
        required=True,
        help="molecule index within metrics/mol_idx; must correspond to --system-id, "
        "which result.h5 cannot confirm",
    )
    # Required: result.h5 carries no step or iteration index anywhere, so the
    # logger period is not recoverable from the file, and it multiplies straight
    # into `steps` and `samples`.
    parser.add_argument(
        "--metric-logger-period",
        type=int,
        required=True,
        help="steps between logged rows; not recoverable from result.h5 and it scales "
        "both efficiency denominators",
    )
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument(
        "--min-tail-steps",
        type=int,
        default=MIN_TAIL_STEPS,
        help="absolute floor on the estimator window, in logged steps",
    )
    parser.add_argument(
        "--allow-short-tail",
        action="store_true",
        help=(
            "accept an estimator window below --min-tail-steps when the run is "
            "too short to fill it, instead of refusing on that ground; the notes "
            "then say the estimate is provisional. How wide the accepted window "
            "ends up is select_tail's rule in experiments/baselines/statistics.py, "
            "not this flag's, and that rule has changed there before -- read it "
            "there rather than assuming a value. This flag does NOT relax the "
            "separate MIN_BLOCKS floor: a window with too few steps to run one "
            "blocking level is still refused, because its error bar would be an "
            "uncorrected naive estimate. Two different floors, one flag, and it "
            "only governs the first"
        ),
    )
    parser.add_argument("--code-commit", default=None)
    # Required: the file does not record the optimizer either, and "kfac"
    # arriving by default is a provenance claim nothing sourced.
    parser.add_argument(
        "--optimizer", required=True, help="not recorded in result.h5; never inferred"
    )
    parser.add_argument(
        "--device-type",
        default=None,
        help="e.g. cuda; needed only when gpu_type is a kind this adapter does not "
        "recognise, and refused if it contradicts the file",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parameter-count", type=int, default=None)
    parser.add_argument(
        "--note",
        default=None,
        help="operator caveat appended to the generated notes, never replacing them",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            electron_batch_size=args.electron_batch_size,
            ansatz=args.ansatz,
            estimator=args.estimator,
            training_provenance=args.training_provenance,
            checkpoint_provenance=args.checkpoint_provenance,
            mol_idx=args.mol_idx,
            metric_logger_period=args.metric_logger_period,
            tail_fraction=args.tail_fraction,
            min_tail_steps=args.min_tail_steps,
            allow_short_tail=args.allow_short_tail,
            code_commit=args.code_commit,
            optimizer=args.optimizer,
            dtype=args.dtype,
            seed=args.seed,
            parameter_count=args.parameter_count,
            device_type=args.device_type,
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
