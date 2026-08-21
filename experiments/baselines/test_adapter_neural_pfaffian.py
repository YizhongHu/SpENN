"""Tests for the Neural Pfaffian run-directory adapter.

Every test here runs from a synthetic ``train_log.csv`` written into ``tmp_path``
or from in-memory series. Nothing needs JAX, a GPU, or a real run: the adapter is
a parser plus an estimator, and both are fully exercisable without either.

The tests that matter most are the ones asserting what the adapter *refuses* to
do -- emit a record from a log whose header it does not recognise, and report the
local-energy spread as if it were an error bar.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.baselines.errors import AdapterError
from experiments.baselines.records import BaselineRecord
from experiments.baselines.adapters.neural_pfaffian import (
    CODE_NAME,
    DEFAULT_ANSATZ,
    DEFAULT_TAIL_FRACTION,
    ENERGY_COLUMN,
    RECORD_FILENAME,
    SEMANTICS_NOTE,
    SEMANTICS_READ_DATE,
    SPREAD_COLUMN,
    STEP_TIME_COLUMN,
    TRAIN_LOG_FILENAME,
    build_record,
    main,
    read_train_log,
    record_from_series,
    write_record,
)
from experiments.baselines.adapters import neural_pfaffian

WALKERS = 4096


def _plateau(count: int = 40000, seed: int = 11) -> list[float]:
    """A converged series: noise about a fixed mean, no drift."""

    rng = random.Random(seed)
    return [-2.90372 + rng.gauss(0.0, 1e-5) for _ in range(count)]


def _descending(count: int = 40000) -> list[float]:
    """A series still falling at the end of the trace."""

    return [-2.90 - 1e-6 * index for index in range(count)]


def _write_log(
    run_dir: Path,
    energies: list[float],
    spreads: list[float] | None = None,
    step_times: list[float] | None = None,
    *,
    header: list[str] | None = None,
) -> Path:
    """Write a ``train_log.csv`` shaped like the code's own logger output."""

    run_dir.mkdir(parents=True, exist_ok=True)
    spreads = [0.05] * len(energies) if spreads is None else spreads
    columns = header if header is not None else (
        [ENERGY_COLUMN, SPREAD_COLUMN, "grad", "step"]
        + ([STEP_TIME_COLUMN] if step_times is not None else [])
    )
    lines = [",".join(columns)]
    for index, energy in enumerate(energies):
        cells = {
            ENERGY_COLUMN: repr(energy),
            SPREAD_COLUMN: repr(spreads[index]),
            "grad": "0.5",
            "step": str(index),
        }
        if step_times is not None:
            cells[STEP_TIME_COLUMN] = repr(step_times[index])
        lines.append(",".join(cells.get(column, "") for column in columns))

    path = run_dir / TRAIN_LOG_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the header assumption, asserted rather than assumed
# --------------------------------------------------------------------------


def test_missing_energy_column_is_rejected_and_cites_the_note(tmp_path: Path) -> None:
    """A header without ``E`` must fail loudly, naming the stale assumption.

    The mapping came from reading the code on the cluster, and this repository
    cannot re-read it. So the failure has to tell the reader which artifact to
    distrust, not merely that a column is absent.
    """

    _write_log(
        tmp_path / "run",
        _plateau(100),
        header=["energy", SPREAD_COLUMN, "step"],
    )
    with pytest.raises(AdapterError) as raised:
        read_train_log(tmp_path / "run")

    message = str(raised.value)
    assert ENERGY_COLUMN in message
    assert SEMANTICS_NOTE in message
    assert SEMANTICS_READ_DATE in message
    # The observed header must appear, so the reader sees what the file has and
    # not only what the adapter wanted.
    assert "energy" in message


def test_missing_spread_column_is_rejected_and_cites_the_note(tmp_path: Path) -> None:
    """``E`` alone is not enough: the variance column is required, not optional."""

    _write_log(tmp_path / "run", _plateau(100), header=[ENERGY_COLUMN, "grad", "step"])
    with pytest.raises(AdapterError) as raised:
        read_train_log(tmp_path / "run")

    message = str(raised.value)
    assert SPREAD_COLUMN in message
    assert SEMANTICS_NOTE in message
    assert SEMANTICS_READ_DATE in message


def test_zero_data_rows_is_an_error_not_an_empty_record(tmp_path: Path) -> None:
    """A header-only log means the run produced nothing, which must not pass."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TRAIN_LOG_FILENAME).write_text(
        f"{ENERGY_COLUMN},{SPREAD_COLUMN},step\n", encoding="utf-8"
    )
    with pytest.raises(AdapterError, match="no data rows"):
        read_train_log(run_dir)


def test_missing_train_log_is_an_error(tmp_path: Path) -> None:
    """An empty directory is a failed run, not a run with no energy."""

    (tmp_path / "run").mkdir()
    with pytest.raises(AdapterError, match=TRAIN_LOG_FILENAME):
        read_train_log(tmp_path / "run")


def test_half_filled_row_is_rejected_rather_than_dropped(tmp_path: Path) -> None:
    """Dropping a row with ``E`` but no ``E_std`` would misalign the series.

    The logger emits ``data.get(header, "")``, so a partially filled row is
    physically possible. Skipping it would shift every later spread by one step
    relative to its energy, which no downstream check could detect.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TRAIN_LOG_FILENAME).write_text(
        f"{ENERGY_COLUMN},{SPREAD_COLUMN},step\n-2.9,0.05,0\n-2.9,,1\n",
        encoding="utf-8",
    )
    with pytest.raises(AdapterError, match="misalign"):
        read_train_log(run_dir)


def test_fully_blank_trailing_row_is_skipped(tmp_path: Path) -> None:
    """A blank line is formatting, not a measurement."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TRAIN_LOG_FILENAME).write_text(
        f"{ENERGY_COLUMN},{SPREAD_COLUMN},step\n-2.9,0.05,0\n-2.8,0.04,1\n,,\n",
        encoding="utf-8",
    )
    energies, spreads, step_times = read_train_log(run_dir)
    assert energies == [-2.9, -2.8]
    assert spreads == [0.05, 0.04]
    assert step_times is None


def test_unparseable_number_names_its_line(tmp_path: Path) -> None:
    """A corrupt cell must not become a silent zero."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TRAIN_LOG_FILENAME).write_text(
        f"{ENERGY_COLUMN},{SPREAD_COLUMN}\n-2.9,0.05\nnan-ish,0.05\n", encoding="utf-8"
    )
    with pytest.raises(AdapterError, match="line 3"):
        read_train_log(run_dir)


# --------------------------------------------------------------------------
# E_std is a spread, not an error bar
# --------------------------------------------------------------------------


def test_variance_is_mean_of_squared_spread_not_the_stderr() -> None:
    """``local_energy_variance_hartree2`` is ``mean(E_std**2)``.

    This is the defect the field exists to avoid. ``E_std`` is the population
    spread of the local energies over walkers; the standard error of the step
    mean is smaller by ``sqrt(walkers)``. At 4096 walkers the two differ by a
    factor of 4096 in variance, so a test that only checked "roughly the right
    magnitude" would pass either way. The assertion therefore pins the exact
    value and separately rejects the walker-divided alternative.
    """

    spread = 0.05
    record = record_from_series(
        _plateau(),
        [spread] * 40000,
        system_id="he_atom",
        walkers_per_step=WALKERS,
    )
    assert record.local_energy_variance_hartree2 == pytest.approx(spread**2, rel=1e-12)
    # The two candidate wrong answers, both plausible-looking:
    assert record.local_energy_variance_hartree2 != pytest.approx(spread**2 / WALKERS)
    assert record.local_energy_variance_hartree2 != pytest.approx(
        record.energy_stderr_hartree**2
    )


def test_variance_averages_squares_not_squares_the_average() -> None:
    """``mean(E_std**2)`` differs from ``mean(E_std)**2`` when the spread varies.

    Jensen's inequality makes the second strictly smaller, so a run whose spread
    changed over the tail would report too small a variance under the wrong
    order of operations.
    """

    spreads = [0.02] * 20000 + [0.08] * 20000
    record = record_from_series(
        _plateau(),
        spreads,
        system_id="he_atom",
        walkers_per_step=WALKERS,
        tail_fraction=1.0,
    )
    mean_of_squares = (0.02**2 + 0.08**2) / 2.0
    square_of_mean = 0.05**2
    assert record.local_energy_variance_hartree2 == pytest.approx(mean_of_squares, rel=1e-12)
    assert record.local_energy_variance_hartree2 != pytest.approx(square_of_mean, rel=1e-6)


def test_stderr_comes_from_blocking_the_energy_series() -> None:
    """The bar must respond to the ``E`` series, not to ``E_std``.

    Holding the energies fixed and scaling every spread by ten must leave the
    error bar untouched; if the adapter ever sourced the bar from ``E_std`` this
    test fails.

    The spreads must VARY for that to be true. A first version of this test used
    a constant spread, and blocking a constant series returns 0.0 for any
    constant -- so an adapter that blocked ``E_std`` instead of ``E`` produced
    0.0 == 0.0 and the test passed. The mutant survived. Varying spreads make the
    blocked value scale with them, which is what makes the assertion bite.
    """

    rng = random.Random(3)
    energies = _plateau()
    base = [0.01 + rng.gauss(0.0, 2e-3) for _ in energies]
    small = record_from_series(
        energies, base, system_id="he_atom", walkers_per_step=WALKERS
    )
    large = record_from_series(
        energies,
        [10.0 * value for value in base],
        system_id="he_atom",
        walkers_per_step=WALKERS,
    )
    assert small.energy_stderr_hartree == large.energy_stderr_hartree
    assert small.local_energy_variance_hartree2 != large.local_energy_variance_hartree2


# --------------------------------------------------------------------------
# energy, counts, and the estimator
# --------------------------------------------------------------------------


def test_energy_is_the_tail_mean_and_window_follows_the_fraction() -> None:
    """The estimate averages the trailing fraction, not the whole trace."""

    # First half far above, second half at the true plateau: a whole-trace mean
    # would land near -2.4, a tail mean near -2.9.
    energies = [-1.9] * 20000 + [-2.9] * 20000
    record = record_from_series(
        energies, [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert record.energy_hartree == pytest.approx(-2.9, rel=1e-12)
    assert f"last {int(DEFAULT_TAIL_FRACTION * 40000)} of 40000 steps" in record.notes


def test_steps_and_samples_count_the_whole_run_not_the_tail() -> None:
    """Efficiency denominators must reflect the work done, not the window used."""

    record = record_from_series(
        _plateau(40000), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert record.steps == 40000
    assert record.samples == 40000 * WALKERS


def test_code_and_estimator_are_fixed_by_construction() -> None:
    """No inference stage exists in this codebase, so no record may claim one."""

    record = record_from_series(
        _plateau(), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert record.code == CODE_NAME == "neural-pfaffian"
    assert record.ansatz == DEFAULT_ANSATZ
    assert record.estimator == "training_tail"


def test_mismatched_series_lengths_are_rejected() -> None:
    """Two columns of different length cannot have come from the same rows."""

    with pytest.raises(AdapterError, match="same rows"):
        record_from_series(
            _plateau(40000), [0.05] * 39999, system_id="he_atom", walkers_per_step=WALKERS
        )


# --------------------------------------------------------------------------
# the short-tail exemption, forwarded per call site
# --------------------------------------------------------------------------


def test_short_run_refuses_the_floor_unless_the_caller_opts_in() -> None:
    """A 2000-step run cannot fill the 10000-step floor and must say so."""

    energies, spreads = _plateau(2000), [0.05] * 2000
    with pytest.raises(AdapterError, match="minimum"):
        record_from_series(
            energies, spreads, system_id="he_atom", walkers_per_step=WALKERS
        )

    record = record_from_series(
        energies,
        spreads,
        system_id="he_atom",
        walkers_per_step=WALKERS,
        allow_short_tail=True,
    )
    assert record.steps == 2000
    assert "of 2000 steps" in record.notes


# --------------------------------------------------------------------------
# convergence and provenance in the notes
# --------------------------------------------------------------------------


def test_descending_run_is_flagged_as_not_a_plateau() -> None:
    """A monotone tail means the energy was still falling when the run stopped."""

    energies = _descending()
    record = record_from_series(
        energies, [0.05] * len(energies), system_id="he_atom", walkers_per_step=WALKERS
    )
    assert "monotone" in record.notes
    assert "upper bound" in record.notes


def test_plateaued_run_is_not_flagged_as_descending() -> None:
    """The convergence verdict has to discriminate, or it is not a check."""

    record = record_from_series(
        _plateau(), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert "is mixed" in record.notes
    assert "monotone" not in record.notes


def test_block_count_of_none_never_renders_as_a_number(monkeypatch) -> None:
    """A ``None`` block count must not be interpolated into the notes.

    Two shipped adapters render "from None blocks", which reads as a
    measurement and raises nothing. A deferred contract change to
    ``blocking_stderr`` makes that return value reachable, so this adapter
    branches on ``is None`` and the branch is pinned here.
    """

    monkeypatch.setattr(
        neural_pfaffian, "blocking_stderr", lambda values, *args, **kwargs: (1e-5, None)
    )
    record = record_from_series(
        _plateau(), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert "None blocks" not in record.notes
    assert "an unreported number of blocks" in record.notes


def test_notes_cite_the_semantics_note_for_dtype_and_optimizer() -> None:
    """dtype and optimizer are read from a note, not probed; the record says so."""

    record = record_from_series(
        _plateau(), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert SEMANTICS_NOTE in record.notes
    assert SEMANTICS_READ_DATE in record.notes
    assert "runtime probe" in record.notes
    assert record.dtype is not None and "float64" in record.dtype
    assert record.optimizer is not None and "spring" in record.optimizer


# --------------------------------------------------------------------------
# wall clock: VMC stage only, or nothing at all
# --------------------------------------------------------------------------


def test_wall_clock_sums_step_times_and_says_which_stage_it_covers() -> None:
    """A wall clock covering only VMC must not read as end-to-end cost."""

    record = record_from_series(
        _plateau(40000),
        [0.05] * 40000,
        system_id="he_atom",
        walkers_per_step=WALKERS,
        step_times=[0.25] * 40000,
    )
    assert record.wall_clock_seconds == pytest.approx(10000.0)
    assert "VMC stage ONLY" in record.notes
    assert "pretraining" in record.notes


def test_absent_step_times_leave_wall_clock_unset_not_zero() -> None:
    """A missing measurement stays missing; zero would be a false efficiency."""

    record = record_from_series(
        _plateau(40000), [0.05] * 40000, system_id="he_atom", walkers_per_step=WALKERS
    )
    assert record.wall_clock_seconds is None
    assert "unset" in record.notes


def test_blank_step_time_cell_withholds_the_whole_wall_clock(tmp_path: Path) -> None:
    """A partial sum of per-step times is an undercount, not a measurement."""

    energies = _plateau(40000)
    times = [0.25] * 40000
    _write_log(tmp_path / "run", energies, [0.05] * 40000, times)
    path = tmp_path / "run" / TRAIN_LOG_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split(",")
    cells = lines[5].split(",")
    cells[columns.index(STEP_TIME_COLUMN)] = ""
    lines[5] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _, _, step_times = read_train_log(tmp_path / "run")
    assert step_times is None


def test_negative_step_time_is_rejected() -> None:
    """A negative duration means the column is not the timer we assumed."""

    with pytest.raises(AdapterError, match="negative"):
        record_from_series(
            _plateau(40000),
            [0.05] * 40000,
            system_id="he_atom",
            walkers_per_step=WALKERS,
            step_times=[0.25] * 39999 + [-1.0],
        )


# --------------------------------------------------------------------------
# disk round trip and CLI
# --------------------------------------------------------------------------


def test_build_record_reads_from_disk_and_round_trips_through_json(tmp_path: Path) -> None:
    """The written record must parse back under the shared schema."""

    run_dir = tmp_path / "np-he-run"
    _write_log(run_dir, _plateau(40000), [0.05] * 40000, [0.25] * 40000)
    record = build_record(
        run_dir,
        system_id="he_atom",
        walkers_per_step=WALKERS,
        code_commit="f711f08",
        device_type="cuda",
        gpu_model="NVIDIA A100-SXM4-40GB",
        seed=7,
    )
    assert record.run_id == "np-he-run"
    # run_dir is the collector's to stamp, relative to its own scan root.
    assert record.run_dir is None
    assert record.n_gpus == 1

    path = write_record(record, run_dir)
    assert path.name == RECORD_FILENAME
    reloaded = BaselineRecord.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded == record


def test_n_gpus_is_not_invented_for_an_unknown_device(tmp_path: Path) -> None:
    """No device claim means no GPU count, not a default of one."""

    run_dir = tmp_path / "run"
    _write_log(run_dir, _plateau(40000), [0.05] * 40000)
    record = build_record(run_dir, system_id="he_atom", walkers_per_step=WALKERS)
    assert record.device_type is None
    assert record.n_gpus is None


def test_main_dry_run_prints_and_writes_nothing(tmp_path: Path, capsys) -> None:
    """``--dry-run`` must leave the run directory untouched."""

    run_dir = tmp_path / "run"
    _write_log(run_dir, _plateau(40000), [0.05] * 40000)
    code = main(
        [
            "--run-dir",
            str(run_dir),
            "--system-id",
            "he_atom",
            "--walkers-per-step",
            str(WALKERS),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (run_dir / RECORD_FILENAME).exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == CODE_NAME


def test_main_returns_one_on_adapter_error(tmp_path: Path, capsys) -> None:
    """A bad run directory is a nonzero exit, not a traceback and not a record."""

    (tmp_path / "run").mkdir()
    code = main(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--system-id",
            "he_atom",
            "--walkers-per-step",
            str(WALKERS),
        ]
    )
    assert code == 1
    assert TRAIN_LOG_FILENAME in capsys.readouterr().err
