"""Tests for the common results record and the run-directory collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.baselines.collect import RECORD_FILENAME, collect, main, write_jsonl
from experiments.baselines.records import BaselineRecord, RecordValidationError


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    """Return a complete, valid record payload.

    Parameters
    ----------
    **overrides
        Fields to override on the base payload.

    Returns
    -------
    dict
        Payload suitable for :meth:`BaselineRecord.from_json_dict`.
    """

    payload: dict[str, Any] = {
        "system_id": "hooke_pair_singlet_omega0.5",
        "code": "tpen",
        "code_commit": "0123456789abcdef0123456789abcdef01234567",
        "ansatz": "tpen-pair-v1",
        "energy_hartree": 2.0004,
        "energy_stderr_hartree": 0.0003,
        "local_energy_variance_hartree2": 0.012,
        "steps": 25,
        "samples": 3200,
        "wall_clock_seconds": 91.5,
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "n_gpus": 1,
        "dtype": "float64",
        "optimizer": "adam",
        "parameter_count": 4321,
        "seed": 7,
        "run_id": "hooke-pair-v1-seed7",
        "run_dir": None,
        "collected_at": "2026-08-13T00:00:00+00:00",
        "notes": None,
    }
    payload.update(overrides)
    return payload


def _write_record(run_root: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write one record file into a run directory.

    Returns
    -------
    pathlib.Path
        The run directory that was created.
    """

    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RECORD_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_round_trip_preserves_every_field() -> None:
    """A record survives a JSON round trip unchanged."""

    payload = _valid_payload()
    record = BaselineRecord.from_json_dict(payload)
    assert record.to_json_dict() == payload
    assert BaselineRecord.from_json_dict(record.to_json_dict()) == record


def test_all_schema_fields_are_emitted() -> None:
    """Every field appears on every line, so the JSONL has a stable header set."""

    record = BaselineRecord(system_id="he_atom", code="ferminet")
    assert set(record.to_json_dict()) == set(BaselineRecord.field_names())


def test_scorecard_axes_are_present() -> None:
    """The README section-3 axes all have a home in the schema."""

    required = {
        "energy_hartree",
        "energy_stderr_hartree",
        "local_energy_variance_hartree2",
        "steps",
        "samples",
        "wall_clock_seconds",
        "gpu_model",
        "dtype",
        "optimizer",
        "parameter_count",
        "code",
        "code_commit",
    }
    assert required <= set(BaselineRecord.field_names())


def test_unknown_quantities_stay_none() -> None:
    """Unmeasured fields default to None rather than to a placeholder number."""

    record = BaselineRecord(system_id="he_atom", code="ferminet")
    assert record.energy_hartree is None
    assert record.wall_clock_seconds is None
    assert record.parameter_count is None


@pytest.mark.parametrize("field", ["system_id", "code"])
def test_identity_fields_are_required(field: str) -> None:
    """A record without a system or a code cannot be joined to anything."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(**{field: ""}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_energy_is_rejected(bad: float) -> None:
    """NaN and infinity never enter the results file."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(energy_hartree=bad))


@pytest.mark.parametrize(
    "field",
    ["energy_stderr_hartree", "local_energy_variance_hartree2", "wall_clock_seconds"],
)
def test_magnitudes_cannot_be_negative(field: str) -> None:
    """Error bars, variances, and wall clock are magnitudes."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(**{field: -1.0}))


@pytest.mark.parametrize("field", ["steps", "samples", "parameter_count", "n_gpus"])
def test_counts_must_be_non_negative_ints(field: str) -> None:
    """Counts are integers; a float count means the emitter guessed."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(**{field: 1.5}))
    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(**{field: -1}))


def test_energy_requires_an_error_bar() -> None:
    """An energy without a stated error bar is not a comparable measurement."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(energy_stderr_hartree=None))


def test_unknown_fields_are_rejected() -> None:
    """A typo in an emitter fails loudly instead of dropping a measurement."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict(_valid_payload(energy_hatree=1.0))


def test_non_object_payload_is_rejected() -> None:
    """A JSON array is not a record."""

    with pytest.raises(RecordValidationError):
        BaselineRecord.from_json_dict([1, 2, 3])


def test_collect_finds_nested_runs_and_stamps_relative_run_dir(tmp_path: Path) -> None:
    """Records are found at any depth and located relative to the run root."""

    _write_record(tmp_path, "seed7", _valid_payload(run_id="a"))
    _write_record(tmp_path, "nested/seed8", _valid_payload(run_id="b", seed=8))

    report = collect(tmp_path)

    assert report.failures == []
    assert [record.run_id for record in report.records] == ["b", "a"]
    assert sorted(record.run_dir for record in report.records) == ["nested/seed8", "seed7"]
    # No absolute path leaks into a collected record.
    for record in report.records:
        assert not Path(record.run_dir).is_absolute()


def test_collect_preserves_emitter_supplied_run_dir(tmp_path: Path) -> None:
    """The collector never overwrites provenance the emitter asserted."""

    _write_record(tmp_path, "seed7", _valid_payload(run_dir="asserted/by/emitter"))

    report = collect(tmp_path)

    assert [record.run_dir for record in report.records] == ["asserted/by/emitter"]


def test_collect_reports_invalid_records_without_dropping_them_silently(tmp_path: Path) -> None:
    """A malformed record is reported with its path and excluded from output."""

    _write_record(tmp_path, "good", _valid_payload())
    _write_record(tmp_path, "bad", _valid_payload(steps=-5))
    (tmp_path / "unparseable").mkdir()
    (tmp_path / "unparseable" / RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    report = collect(tmp_path)

    assert len(report.records) == 1
    assert sorted(path for path, _ in report.failures) == ["bad", "unparseable"]


def test_collect_requires_an_existing_run_root(tmp_path: Path) -> None:
    """Scanning a missing directory fails instead of reporting zero runs."""

    with pytest.raises(FileNotFoundError):
        collect(tmp_path / "absent")


def test_write_jsonl_emits_one_object_per_line(tmp_path: Path) -> None:
    """The output file is line-delimited JSON."""

    records = [
        BaselineRecord.from_json_dict(_valid_payload(run_id="a")),
        BaselineRecord.from_json_dict(_valid_payload(run_id="b")),
    ]
    output = tmp_path / "nested" / "results.jsonl"

    write_jsonl(records, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["a", "b"]


def test_cli_succeeds_on_clean_tree(tmp_path: Path) -> None:
    """A clean collection exits zero and writes the results file."""

    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_record(run_root, "seed7", _valid_payload())
    output = tmp_path / "results.jsonl"

    exit_code = main(["--run-root", str(run_root), "--output", str(output)])

    assert exit_code == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_fails_when_any_record_is_invalid(tmp_path: Path) -> None:
    """A partial pass is a failure, not a shorter results file."""

    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_record(run_root, "good", _valid_payload())
    _write_record(run_root, "bad", _valid_payload(system_id=""))
    output = tmp_path / "results.jsonl"

    exit_code = main(["--run-root", str(run_root), "--output", str(output)])

    assert exit_code == 1
