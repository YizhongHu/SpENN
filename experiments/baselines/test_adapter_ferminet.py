"""Tests for the FermiNet run-directory adapter."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from experiments.baselines.collect import collect

from experiments.baselines.adapters.ferminet import (
    AdapterError,
    blocking_stderr,
    build_record,
    parse_device,
    parse_wall_clock_seconds,
    read_energies,
    write_record,
)

LOG_TEXT = """host=holygpu7c26105 job=38985858 start=2026-08-13T12:45:00-04:00
NVIDIA A100-SXM4-40GB, GPU-39166c9d-3031-31cb-f77d-d62cb3f889f9, 40960 MiB
jax 0.9.2 [CudaDevice(id=0)]
end=2026-08-13T16:22:38-04:00
R2_FULL_DONE
"""


def _write_stats(run_dir: Path, energies: list[float]) -> Path:
    """Write a minimal FermiNet ``train_stats.csv`` and return the directory."""

    run_dir.mkdir(parents=True, exist_ok=True)
    lines = ["step,energy,ewmean,ewvar,pmove"]
    lines += [f"{i},{e},{e},0.0,0.95" for i, e in enumerate(energies)]
    (run_dir / "train_stats.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_missing_stats_file_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    """A run with no stats file is an error, never a null-energy record."""

    (tmp_path / "run").mkdir()
    with pytest.raises(AdapterError, match="no train_stats.csv"):
        read_energies(tmp_path / "run")


def test_header_only_stats_file_raises(tmp_path: Path) -> None:
    """Zero data rows must fail loudly; absence of data is not a clean result."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_stats.csv").write_text("step,energy,ewmean,ewvar,pmove\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="no data rows"):
        read_energies(run_dir)


def test_stats_file_without_energy_column_raises(tmp_path: Path) -> None:
    """A schema change upstream must surface, not silently yield nothing."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_stats.csv").write_text("step,ewmean\n0,1.0\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="no 'energy' column"):
        read_energies(run_dir)


def test_blocking_exceeds_naive_stderr_on_correlated_data() -> None:
    """Blocking must inflate the bar on autocorrelated input.

    This is the property that makes the adapter's error bar honest: an AR(1)
    series has fewer independent samples than points, and the naive standard
    error does not know that.
    """

    rng = random.Random(20260813)
    value, series = 0.0, []
    for _ in range(8192):
        value = 0.9 * value + rng.gauss(0.0, 1.0)
        series.append(value)

    naive = math.sqrt(sum((x - sum(series) / len(series)) ** 2 for x in series) / (len(series) - 1) / len(series))
    blocked, n_blocks = blocking_stderr(series)

    assert blocked > naive * 1.5, "blocking must widen the bar on correlated data"
    assert n_blocks <= len(series)


def test_blocking_matches_naive_on_independent_data() -> None:
    """On uncorrelated input the two estimates should be close."""

    rng = random.Random(11)
    series = [rng.gauss(0.0, 1.0) for _ in range(8192)]
    naive = 1.0 / math.sqrt(len(series))
    blocked, _ = blocking_stderr(series)
    assert 0.6 * naive < blocked < 1.8 * naive


def test_blocking_rejects_single_sample() -> None:
    with pytest.raises(AdapterError, match="at least two values"):
        blocking_stderr([1.0])


def test_parse_device_reads_delivered_card_not_partition() -> None:
    """Device identity comes from nvidia-smi inside the allocation."""

    device_type, gpu_model = parse_device(LOG_TEXT)
    assert device_type == "cuda"
    assert gpu_model == "NVIDIA A100-SXM4-40GB"


def test_parse_device_absent_yields_none_not_a_guess() -> None:
    assert parse_device("no device line here") == (None, None)


def test_parse_wall_clock_seconds() -> None:
    assert parse_wall_clock_seconds(LOG_TEXT) == pytest.approx(3 * 3600 + 37 * 60 + 38)


def test_build_record_round_trips_and_carries_estimator_caveat(tmp_path: Path) -> None:
    """A built record validates, serialises, and names its estimator."""

    energies = [-7.4779 + 0.0001 * math.sin(i / 7.0) for i in range(2000)]
    run_dir = _write_stats(tmp_path / "Li", energies)
    (tmp_path / "job.out").write_text(LOG_TEXT, encoding="utf-8")

    record = build_record(
        run_dir,
        system_id="li_atom",
        batch_size=4096,
        log_path=tmp_path / "job.out",
        code_commit="deadbeef",
    )

    assert record.system_id == "li_atom"
    assert record.code == "ferminet"
    assert record.steps == 2000
    assert record.samples == 2000 * 4096
    assert record.device_type == "cuda"
    assert record.gpu_model == "NVIDIA A100-SXM4-40GB"
    assert record.energy_hartree == pytest.approx(-7.4779, abs=1e-3)
    assert record.energy_stderr_hartree is not None and record.energy_stderr_hartree > 0.0
    # dtype is unknown for FermiNet and must not be invented.
    assert record.dtype is None
    # The estimator difference from the published table must travel with the row.
    assert "post-training evaluation" in (record.notes or "")

    path = write_record(record, run_dir)
    assert json.loads(path.read_text(encoding="utf-8"))["system_id"] == "li_atom"


def test_adapter_records_preserve_nested_run_provenance_on_collection(tmp_path: Path) -> None:
    """Collector stamps distinct root-relative paths for duplicate run basenames."""

    run_root = tmp_path / "runs"
    first = _write_stats(
        run_root / "system-a" / "seed-0",
        [-7.4 + 0.0001 * i for i in range(100)],
    )
    second = _write_stats(
        run_root / "system-b" / "seed-0",
        [-7.5 + 0.0001 * i for i in range(100)],
    )

    for run_dir in (first, second):
        record = build_record(run_dir, system_id="li_atom", batch_size=256)
        assert record.run_dir is None
        write_record(record, run_dir)

    report = collect(run_root)

    assert report.failures == []
    assert sorted(record.run_dir for record in report.records) == [
        "system-a/seed-0",
        "system-b/seed-0",
    ]


def test_build_record_rejects_a_tail_too_short_to_estimate(tmp_path: Path) -> None:
    """A tail of one sample cannot carry an error bar, so it must fail."""

    run_dir = _write_stats(tmp_path / "tiny", [-1.0, -1.1, -1.2])
    with pytest.raises(AdapterError, match="need >= 2"):
        build_record(run_dir, system_id="he_atom", batch_size=16, tail_fraction=0.01)


def test_build_record_rejects_out_of_range_tail_fraction(tmp_path: Path) -> None:
    run_dir = _write_stats(tmp_path / "run", [-1.0] * 100)
    with pytest.raises(AdapterError, match="tail_fraction"):
        build_record(run_dir, system_id="he_atom", batch_size=16, tail_fraction=0.0)


def test_missing_log_leaves_device_fields_null(tmp_path: Path) -> None:
    """Without a log, device fields stay None rather than being guessed."""

    run_dir = _write_stats(tmp_path / "run", [-2.9 + 0.001 * (i % 5) for i in range(500)])
    record = build_record(run_dir, system_id="he_atom", batch_size=256)
    assert record.device_type is None
    assert record.gpu_model is None
    assert record.wall_clock_seconds is None
