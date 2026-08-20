"""Tests for the DeepQMC run-directory adapter.

Most tests exercise :func:`record_from_series`, which takes an energy list and
therefore needs no HDF5 file and no ``h5py``. That split is deliberate: ``h5py``
is not a TPEN dependency, so tests that require it would be skipped in this
environment, and a skipped test protects nothing. The HDF5-reading tests are
gated on ``h5py`` and are the ones that must be run in the DeepQMC virtualenv on
the cluster.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.baselines.errors import AdapterError
from experiments.baselines.adapters.deepqmc import (
    DEFAULT_TAIL_FRACTION,
    ENERGY_DATASET,
    build_record,
    parse_device,
    parse_wall_clock_seconds,
    read_energies,
    record_from_series,
    result_path,
    write_record,
)

LOG_TEXT = """ansatz=lapnet host=holygpu8a25406 job=39639503 start=2026-08-16T20:13:16-04:00
NVIDIA H200, GPU-1a4626a2-1754-c017-fdb7-6d7d682f4165, 143771 MiB
jax 0.8.3 [CudaDevice(id=0)]
end=2026-08-17T15:43:32-04:00
"""


def _converged(count: int = 40000, seed: int = 5) -> list[float]:
    """A plateaued series: noise about a fixed mean, no drift."""

    rng = random.Random(seed)
    return [-2.9037 + rng.gauss(0.0, 1e-5) for _ in range(count)]


# --------------------------------------------------------------------------
# code identity -- the defect this adapter exists to avoid
# --------------------------------------------------------------------------


def test_code_is_deepqmc_never_ferminet() -> None:
    """A DeepQMC record must never claim the FermiNet codebase.

    DeepQMC ships an ansatz named ``ferminet``, but it is a reimplementation.
    ``code="ferminet", ansatz="ferminet"`` claims a run of google-deepmind's
    code; this must claim DeepQMC's. The mistake is invisible to a reader,
    because the ansatz field looks right either way.
    """

    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="ferminet", run_id="r"
    )
    assert record.code == "deepqmc"
    assert record.ansatz == "ferminet"


def test_ansatz_is_recorded_not_assumed() -> None:
    """Nothing about a DeepQMC run directory reveals which ansatz produced it."""

    for ansatz in ("lapnet", "psiformer", "deeperwin", "default"):
        record = record_from_series(
            _converged(), system_id="he_atom", batch_size=4096, ansatz=ansatz, run_id="r"
        )
        assert record.ansatz == ansatz


def test_reimplementation_is_flagged_in_notes_but_native_default_is_not() -> None:
    """`default` is PauliNet's own code; the others are reimplementations.

    DeepQMC *is* the PauliNet repository, so that one asymmetry must not be
    flattened into "everything here is a reimplementation".
    """

    reimpl = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    native = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="default", run_id="r"
    )
    assert "REIMPLEMENTATION" in (reimpl.notes or "")
    assert "native" in (native.notes or "")
    assert "REIMPLEMENTATION" not in (native.notes or "")


# --------------------------------------------------------------------------
# estimator and convergence reporting
# --------------------------------------------------------------------------


def test_default_tail_is_long() -> None:
    """A short tail produced impossible below-exact energies four times here."""

    assert DEFAULT_TAIL_FRACTION >= 0.25


def test_notes_report_a_monotone_tail_as_possibly_unconverged() -> None:
    """A still-descending run must be flagged, not reported as a clean number.

    This is the failure that invalidated the largest effect in this program's
    six-system comparison: a tail average looked perfectly stable while the run
    was still improving at its final step.
    """

    descending = [-7.4779 - 1e-9 * i for i in range(80000)]
    record = record_from_series(
        descending, system_id="li_atom", batch_size=4096, ansatz="ferminet", run_id="r"
    )
    assert "MONOTONE" in (record.notes or "")
    assert "may not have converged" in (record.notes or "")


def test_notes_report_a_plateaued_tail_as_noise() -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert "not monotone" in (record.notes or "")
    assert "MONOTONE" not in (record.notes or "")


def test_convergence_is_reported_unassessed_when_the_tail_is_too_short() -> None:
    """Silence would read as 'converged'. An unassessable tail says so."""

    record = record_from_series(
        [-2.9 + 1e-6 * i for i in range(6)],
        system_id="he_atom",
        batch_size=16,
        ansatz="lapnet",
        run_id="r",
        tail_fraction=1.0,
        allow_short_tail=True,
    )
    assert "UNASSESSED" in (record.notes or "")
    # A window under the floor must SAY so; silence would read as a full window.
    assert "provisional" in (record.notes or "")


def test_notes_carry_the_autocorrelation_inflation_ratio() -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert "inflation" in (record.notes or "")


def test_estimator_distinguishes_training_tail_from_inference() -> None:
    train = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    infer = record_from_series(
        _converged(),
        system_id="he_atom",
        batch_size=4096,
        ansatz="lapnet",
        run_id="r",
        estimator="inference",
        tail_fraction=1.0,
    )
    assert train.estimator == "training_tail"
    assert infer.estimator == "inference"
    assert "Training-tail average" in (train.notes or "")
    assert "Fixed-parameter inference" in (infer.notes or "")


# --------------------------------------------------------------------------
# arithmetic and validation
# --------------------------------------------------------------------------


def test_steps_and_samples_count_the_whole_run_not_the_tail() -> None:
    """`steps` is the run length; the tail is only the estimator window."""

    record = record_from_series(
        _converged(count=40000), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert record.steps == 40000
    assert record.samples == 40000 * 4096


def test_energy_is_the_tail_mean_not_the_final_value() -> None:
    """A final row is one step's walker mean and is pure noise at this scale."""

    series = [-1.0] * 39999 + [999.0]
    record = record_from_series(
        series, system_id="he_atom", batch_size=16, ansatz="lapnet", run_id="r", tail_fraction=0.5
    )
    assert record.energy_hartree == pytest.approx(-1.0, abs=0.1)


def test_rejects_out_of_range_tail_fraction() -> None:
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(AdapterError, match="tail fraction must be in"):
            record_from_series(
                _converged(), system_id="he_atom", batch_size=16, ansatz="lapnet", run_id="r",
                tail_fraction=bad,
            )


def test_rejects_a_run_shorter_than_the_floor() -> None:
    """A run too short for the minimum window must refuse, not quietly shrink.

    Naming both the run length and the floor is deliberate: "too short" is not
    actionable without them.
    """

    with pytest.raises(AdapterError, match="run of 3 steps cannot fill the 10000-step"):
        record_from_series(
            [-1.0, -1.1, -1.2],
            system_id="he_atom",
            batch_size=16,
            ansatz="lapnet",
            run_id="r",
        )


def test_floor_overrides_a_fraction_that_would_be_too_small() -> None:
    """0.25 of 20000 is 5000 steps, which produced impossible energies.

    The floor must win, so the window is 10000 and the notes say so in STEPS.
    """

    record = record_from_series(
        _converged(count=20000), system_id="he_atom", batch_size=4096,
        ansatz="psiformer", run_id="r", tail_fraction=0.25,
    )
    assert "last 10000 of 20000 steps" in (record.notes or "")
    assert "provisional" not in (record.notes or "")


def test_fraction_wins_when_it_exceeds_the_floor() -> None:
    """On a long run the fraction still governs; the floor is only a minimum."""

    record = record_from_series(
        _converged(count=200000), system_id="he_atom", batch_size=4096,
        ansatz="psiformer", run_id="r", tail_fraction=0.25,
    )
    assert "last 50000 of 200000 steps" in (record.notes or "")


# --------------------------------------------------------------------------
# operator caveats -- facts the numbers cannot carry
# --------------------------------------------------------------------------


def test_note_is_appended_without_displacing_the_generated_account() -> None:
    """A caveat extends the provenance text; it never replaces any of it."""

    kwargs = dict(
        system_id="lih_molecule", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    plain = record_from_series(_converged(), **kwargs)
    caveated = record_from_series(
        _converged(), **kwargs, note="Ran at R=3.09913 bohr, registry is 3.015 bohr."
    )

    assert (plain.notes or "") in (caveated.notes or "")
    assert (caveated.notes or "").endswith(
        " Ran at R=3.09913 bohr, registry is 3.015 bohr."
    )


def test_note_changes_no_number() -> None:
    """The caveat is documentation; every estimated field must be untouched."""

    kwargs = dict(
        system_id="lih_molecule", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    plain = record_from_series(_converged(), **kwargs)
    caveated = record_from_series(_converged(), **kwargs, note="geometry deviates")

    assert caveated.energy_hartree == plain.energy_hartree
    assert caveated.energy_stderr_hartree == plain.energy_stderr_hartree
    assert caveated.steps == plain.steps
    assert caveated.samples == plain.samples


def test_omitting_the_note_leaves_the_record_unchanged() -> None:
    """Default behaviour is the pre-change adapter's, character for character."""

    kwargs = dict(
        system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert (
        record_from_series(_converged(), **kwargs, note=None).notes
        == record_from_series(_converged(), **kwargs).notes
    )


def test_whitespace_only_note_is_refused_rather_than_dropped() -> None:
    """A caveat that silently vanishes is worse than no argument at all."""

    for blank in ("", "   ", "\t\n"):
        with pytest.raises(AdapterError, match="empty"):
            record_from_series(
                _converged(), system_id="he_atom", batch_size=4096,
                ansatz="lapnet", run_id="r", note=blank,
            )


def test_device_fields_stay_none_without_a_log() -> None:
    """Unknown provenance is None, never an invented value."""

    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert record.device_type is None
    assert record.gpu_model is None
    assert record.wall_clock_seconds is None
    assert record.dtype is None


def test_parse_device_reads_the_delivered_card() -> None:
    """`seas_gpu` mixes H200 and A100, so the partition does not imply hardware."""

    assert parse_device(LOG_TEXT) == ("cuda", "NVIDIA H200")


def test_parse_device_absent_yields_none_not_a_guess() -> None:
    assert parse_device("no device line here") == (None, None)


def test_parse_wall_clock_seconds() -> None:
    assert parse_wall_clock_seconds(LOG_TEXT) == pytest.approx(19 * 3600 + 30 * 60 + 16)


def test_record_round_trips_to_json(tmp_path: Path) -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    path = write_record(record, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["code"] == "deepqmc"
    assert payload["ansatz"] == "lapnet"
    assert payload["run_dir"] is None


# --------------------------------------------------------------------------
# HDF5 reading -- needs h5py, which is NOT a TPEN dependency
# --------------------------------------------------------------------------


def test_result_path_accepts_run_root_or_training_subdir(tmp_path: Path) -> None:
    nested = tmp_path / "training"
    nested.mkdir()
    (nested / "result.h5").write_bytes(b"")
    assert result_path(tmp_path) == nested / "result.h5"

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "result.h5").write_bytes(b"")
    assert result_path(flat) == flat / "result.h5"


def test_missing_result_file_raises(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    with pytest.raises(AdapterError, match="no result.h5"):
        read_energies(tmp_path)


def test_reads_the_three_dimensional_dataset_shape(tmp_path: Path) -> None:
    """DeepQMC writes ``(steps, 1, 1)``; it must be flattened, not indexed."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    values = numpy.asarray([-2.9, -2.91, -2.92], dtype="float32").reshape(3, 1, 1)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)

    assert read_energies(tmp_path) == pytest.approx([-2.9, -2.91, -2.92], abs=1e-6)


def test_missing_dataset_raises(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    run = tmp_path / "training"
    run.mkdir()
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset("something/else", data=[1.0])

    with pytest.raises(AdapterError, match="has no 'local_energy/mean'"):
        read_energies(tmp_path)


def test_empty_dataset_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    """An absent result is an error, never a record with a null energy."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=numpy.zeros((0, 1, 1), dtype="float32"))

    with pytest.raises(AdapterError, match="empty"):
        read_energies(tmp_path)


def test_build_record_end_to_end(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    values = numpy.asarray(_converged(count=4000), dtype="float32").reshape(4000, 1, 1)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)
    (tmp_path / "job.out").write_text(LOG_TEXT, encoding="utf-8")

    record = build_record(
        tmp_path,
        system_id="he_atom",
        batch_size=4096,
        ansatz="lapnet",
        log_path=tmp_path / "job.out",
        code_commit="cafe1234",
    )

    assert record.code == "deepqmc"
    assert record.ansatz == "lapnet"
    assert record.steps == 4000
    assert record.gpu_model == "NVIDIA H200"
    assert record.code_commit == "cafe1234"
    assert record.energy_hartree == pytest.approx(-2.9037, abs=1e-3)
