"""Tests for the FermiNet run-directory adapter."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from experiments.baselines.collect import collect

from experiments.baselines.records import BaselineRecord, RecordValidationError
from experiments.baselines.adapters.ferminet import (
    AdapterError,
    blocking_stderr,
    build_record,
    main,
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

    energies = [-7.4779 + 0.0001 * math.sin(i / 7.0) for i in range(20000)]
    run_dir = _write_stats(tmp_path / "Li", energies)
    (tmp_path / "job.out").write_text(LOG_TEXT, encoding="utf-8")

    record = build_record(
        run_dir,
        system_id="li_atom",
        batch_size=4096,
        ansatz="ferminet",
        log_path=tmp_path / "job.out",
        code_commit="deadbeef",
    )

    assert record.system_id == "li_atom"
    assert record.code == "ferminet"
    assert record.steps == 20000
    assert record.samples == 20000 * 4096
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
        [-7.4 + 0.0001 * i for i in range(20000)],
    )
    second = _write_stats(
        run_root / "system-b" / "seed-0",
        [-7.5 + 0.0001 * i for i in range(20000)],
    )

    for run_dir in (first, second):
        record = build_record(run_dir, system_id="li_atom", batch_size=256, ansatz="ferminet")
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
    with pytest.raises(AdapterError, match="cannot fill the 10000-step minimum"):
        build_record(run_dir, system_id="he_atom", batch_size=16, ansatz="ferminet", tail_fraction=0.01)


def test_build_record_rejects_out_of_range_tail_fraction(tmp_path: Path) -> None:
    run_dir = _write_stats(tmp_path / "run", [-1.0] * 20000)
    with pytest.raises(AdapterError, match="tail fraction must be in"):
        build_record(run_dir, system_id="he_atom", batch_size=16, ansatz="ferminet", tail_fraction=0.0)


def test_missing_log_leaves_device_fields_null(tmp_path: Path) -> None:
    """Without a log, device fields stay None rather than being guessed."""

    run_dir = _write_stats(tmp_path / "run", [-2.9 + 0.001 * (i % 5) for i in range(20000)])
    record = build_record(run_dir, system_id="he_atom", batch_size=256, ansatz="ferminet")
    assert record.device_type is None
    assert record.gpu_model is None
    assert record.wall_clock_seconds is None


def test_ansatz_is_recorded_not_assumed(tmp_path: Path) -> None:
    """A Psiformer run must not emit a record claiming FermiNet.

    `ansatz` was previously hardcoded, so every record said "ferminet"
    regardless of what ran, and two merged Psiformer records were mislabelled.
    """

    run_dir = _write_stats(tmp_path / "He", [-2.9 + 1e-4 * (i % 7) for i in range(20000)])
    record = build_record(run_dir, system_id="he_atom", batch_size=4096, ansatz="psiformer")
    assert record.ansatz == "psiformer"


def test_estimator_distinguishes_training_tail_from_inference(tmp_path: Path) -> None:
    """Two runs of one system must be separable by estimator, not by step count.

    Before this field, a training run and a fixed-parameter inference pass over
    the same system were structurally identical in the record; they could only
    be told apart by noticing `steps` differed, which is inference from a
    coincidence rather than a recorded fact.
    """

    run_dir = _write_stats(tmp_path / "Li", [-7.4779 + 1e-4 * (i % 5) for i in range(20000)])

    train = build_record(run_dir, system_id="li_atom", batch_size=4096, ansatz="ferminet")
    infer = build_record(
        run_dir,
        system_id="li_atom",
        batch_size=4096,
        ansatz="ferminet",
        estimator="inference",
        tail_fraction=1.0,
    )

    assert train.estimator == "training_tail"
    assert infer.estimator == "inference"
    # The caveat text must follow the estimator rather than always claiming a
    # training-tail average.
    assert "post-training evaluation" in (train.notes or "")
    assert "Fixed-parameter inference" in (infer.notes or "")


def test_notes_state_the_realized_window_not_the_requested_fraction(tmp_path: Path) -> None:
    """The floor widens the window past the requested fraction, so a notes string
    built from that fraction is false. Both rendered numbers must be measured.

    The old string interpolated the ARGUMENT for the percentage and the REALIZED
    window for the count, so the two halves of one sentence described different
    windows.
    """

    # 20000 steps at the 0.1 default asks for 2000; MIN_TAIL_STEPS=10000 wins, so
    # the old string said "10%" while averaging half the trace. This length is
    # deliberate: it is inside the band where the rendered percentage is false at
    # every precision. A length above the floor's reach would not discriminate.
    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(20000)]
    run_dir = _write_stats(tmp_path / "mid", energies)

    record = build_record(
        run_dir, system_id="li_atom", batch_size=256, ansatz="ferminet"
    )

    assert "10000 of 20000 steps" in (record.notes or "")
    # Assert the absence of ANY percent sign rather than the absence of "10%".
    # A value check cannot discriminate wherever the format specifier rounds a
    # false fraction back onto the true one; a shape check discriminates at every
    # length and every precision. See the companion test below.
    assert "%" not in (record.notes or "")


def test_notes_are_honest_when_the_floor_takes_the_whole_trace(tmp_path: Path) -> None:
    """At exactly MIN_TAIL_STEPS the window is the entire run, and the worst case
    of the old string described the whole trace as 10% of itself."""

    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(10000)]
    run_dir = _write_stats(tmp_path / "atfloor", energies)

    record = build_record(
        run_dir, system_id="li_atom", batch_size=256, ansatz="ferminet"
    )

    assert "10000 of 10000 steps" in (record.notes or "")
    assert "%" not in (record.notes or "")


def test_notes_render_no_percentage_even_where_one_would_round_true(
    tmp_path: Path,
) -> None:
    """Pin the band where a percentage is wrong but renders right.

    At 39216 steps with a 0.25 fraction the floor still widens the window, from a
    requested 9804 to 10000, so the realized fraction is 25.4998% - yet both it
    and the requested 25% render as "25%" at the precision this adapter used.
    That makes this length the only place where a percentage recomputed from the
    realized window would be byte-identical to the buggy one, so it is the only
    length at which asserting on the percentage's VALUE proves nothing and
    asserting on its ABSENCE proves the shape changed.

    The bounds are measured, not derived by hand: this band's upper end is 39997,
    because round(0.25 * 39998) is 10000 exactly and the floor stops binding two
    steps before 40000.
    """

    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(39216)]
    run_dir = _write_stats(tmp_path / "roundstrue", energies)

    record = build_record(
        run_dir,
        system_id="li_atom",
        batch_size=256,
        ansatz="ferminet",
        tail_fraction=0.25,
    )

    assert "10000 of 39216 steps" in (record.notes or "")
    assert "%" not in (record.notes or "")


def test_record_without_estimator_is_rejected() -> None:
    """An estimator-less record cannot be compared, so it must not validate."""

    with pytest.raises(RecordValidationError, match="estimator must be one of"):
        BaselineRecord(system_id="he_atom", code="ferminet")


def test_record_with_unknown_estimator_is_rejected() -> None:
    """The vocabulary is closed; a plausible-looking string is still wrong."""

    with pytest.raises(RecordValidationError, match="estimator must be one of"):
        BaselineRecord(system_id="he_atom", code="ferminet", estimator="training-tail")


def test_build_record_rejects_a_constant_tail_rather_than_publishing_a_zero_bar(
    tmp_path: Path,
) -> None:
    """A degenerate run must not publish a zero error bar. Two independent gates
    now refuse it, and this test pins WHICH one speaks so a future change to
    either layer is visible here rather than silently absorbed."""

    run_dir = _write_stats(tmp_path / "degenerate", [-7.5] * 20000)

    # Lower layer, statistics.blocking_stderr: refuses outright. It used to
    # return 0.0 for this series, and 0.0 still validates as a record field
    # (below), which is why the adapter keeps its own guard as well.
    with pytest.raises(AdapterError, match="has no measurable spread"):
        blocking_stderr([-7.5] * 20000)
    assert BaselineRecord(
        system_id="li_atom",
        code="ferminet",
        estimator="training_tail",
        energy_hartree=-7.5,
        energy_stderr_hartree=0.0,
    ).energy_stderr_hartree == 0.0

    # Through build_record the statistics raise is what surfaces, so assert on
    # its text. The adapter's own `stderr == 0.0` guard is therefore currently
    # unreachable by this route: it is retained as a second line of defence, not
    # as this test's subject.
    with pytest.raises(AdapterError, match="has no measurable spread"):
        build_record(run_dir, system_id="li_atom", batch_size=256, ansatz="ferminet")

    assert not (run_dir / "baseline_record.json").exists()


def test_allow_short_tail_accepts_a_run_below_the_floor(tmp_path: Path) -> None:
    """The short-tail escape hatch must actually work end to end: it was
    reachable from build_record but exercised by no test."""

    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(4000)]
    run_dir = _write_stats(tmp_path / "short", energies)

    record = build_record(
        run_dir,
        system_id="li_atom",
        batch_size=256,
        ansatz="ferminet",
        allow_short_tail=True,
    )

    # steps stays the length of the trace; the WINDOW is round(0.1 * 4000) = 400.
    # The floor no longer clips back up to the whole run when it cannot be met -
    # opting past the floor buys a short window, not a free full-trace one.
    assert record.steps == 4000
    assert record.samples == 4000 * 256
    assert record.energy_stderr_hartree is not None and record.energy_stderr_hartree > 0.0
    assert "400 of 4000 steps" in (record.notes or "")
    # 400 samples still fills the 32-block minimum, so the count is a real int
    # and the note is not quietly reporting an unblocked estimate.
    assert "from 200 blocks" in (record.notes or "")


def test_short_run_without_the_flag_still_raises(tmp_path: Path) -> None:
    """The counterpart: below the floor is an explicit decision, not a default."""

    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(4000)]
    run_dir = _write_stats(tmp_path / "short", energies)

    with pytest.raises(AdapterError, match="cannot fill the 10000-step minimum"):
        build_record(run_dir, system_id="li_atom", batch_size=256, ansatz="ferminet")


def test_window_too_short_to_block_raises_rather_than_reporting_none_blocks(
    tmp_path: Path,
) -> None:
    """A window below the 32-block minimum has no blocked error bar. The count
    comes back None, an f-string renders that as the literal "None", and a note
    reading "from None blocks" looks like a forgotten field rather than like
    "blocking never ran". Refuse instead of formatting it.

    Match on a substring unique to this adapter, not on statistics' block-floor
    message. Both messages appear on this path, and pinning the wrong one made
    the test pass whether the adapter refused or published: deleting the
    adapter's guard entirely left the suite green.
    """

    # 200 steps, 10% window -> 20 samples, which cannot fill 32 blocks.
    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(200)]
    run_dir = _write_stats(tmp_path / "unblockable", energies)

    # The count really is None at the layer below, and it really does format
    # silently - both halves of the hazard, executed rather than asserted about.
    # This is also the positive control for the assertion afterwards: with the
    # opt-in forwarded, statistics RETURNS here instead of raising, so the only
    # thing left that can refuse is the adapter.
    _, n_blocks = blocking_stderr(energies[-20:], allow_below_floor=True)
    assert n_blocks is None
    assert f"from {n_blocks} blocks" == "from None blocks"

    with pytest.raises(AdapterError, match="cannot say so in a number") as excinfo:
        build_record(
            run_dir,
            system_id="li_atom",
            batch_size=256,
            ansatz="ferminet",
            allow_short_tail=True,
        )

    # Attribute the refusal to the adapter rather than to the layer below.
    assert "cannot fill the 32-block minimum" not in str(excinfo.value)
    assert not (run_dir / "baseline_record.json").exists()


def test_short_but_blockable_window_still_publishes_a_real_block_count(
    tmp_path: Path,
) -> None:
    """Forwarding the opt-in must not turn every short run into a refusal.

    A window that is below the STEP floor but still has enough samples to block
    is a legitimate emission, and it now travels through
    ``blocking_stderr(..., allow_below_floor=True)``. Pin that it comes back
    with an integer count and notes that never say "None", otherwise the fix
    for the unreachable guard would have replaced a silent bad record with a
    blanket refusal.
    """

    # 1000 steps, 10% window -> 100 samples: under the 10000-step floor, over
    # the 32-block minimum.
    energies = [-7.4779 + 1e-4 * (i % 7) for i in range(1000)]
    record = build_record(
        _write_stats(tmp_path / "short-blockable", energies),
        system_id="li_atom",
        batch_size=256,
        ansatz="ferminet",
        allow_short_tail=True,
    )

    assert record.energy_stderr_hartree is not None
    assert record.energy_stderr_hartree > 0.0
    assert "None" not in (record.notes or ""), record.notes
    assert "100 of 1000 steps" in (record.notes or ""), record.notes


def test_published_notes_never_contain_the_string_none(tmp_path: Path) -> None:
    """Guard the whole notes field, not just today's known None. Any future
    Optional interpolated into this string would read as a missing value to a
    human and as a measurement to a parser."""

    # Both branches of the notes expression, and both a long run and a short
    # one. The short case is the one that reaches statistics with the opt-in
    # set, i.e. the only route on which a None count is producible at all.
    cases = (
        ("long", [-7.4779 + 1e-4 * (i % 5) for i in range(20000)], False),
        ("short", [-7.4779 + 1e-4 * (i % 7) for i in range(1000)], True),
    )
    for label, energies, allow_short_tail in cases:
        for estimator in ("training_tail", "inference"):
            record = build_record(
                _write_stats(tmp_path / f"{label}-{estimator}", energies),
                system_id="li_atom",
                batch_size=256,
                ansatz="ferminet",
                estimator=estimator,
                allow_short_tail=allow_short_tail,
            )
            assert "None" not in (record.notes or ""), (label, record.notes)


def test_cli_refuses_an_unblockable_window_cleanly_rather_than_tracebacking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal has to survive the CLI boundary, not just build_record.

    ``--allow-short-tail`` lowers the step floor, and the adapter forwards it to
    the blocking estimator, so a small enough window comes back unblocked and is
    refused here rather than one layer down. That refusal reaching a user as an
    uncaught AdapterError traceback would read as a broken tool rather than as a
    deliberate decision, so pin the exit code, the message on stderr, and the
    absence of an emitted record together.
    """

    # 200 steps, 10% window -> 20 samples, below the 32-block minimum.
    energies = [-7.4779 + 1e-4 * (i % 5) for i in range(200)]
    run_dir = _write_stats(tmp_path / "cli-unblockable", energies)

    rc = main(
        [
            "--run-dir",
            str(run_dir),
            "--system-id",
            "li_atom",
            "--batch-size",
            "256",
            "--ansatz",
            "ferminet",
            "--allow-short-tail",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "cannot say so in a number" in captured.err
    # The refusal must not be filed as a record, and must not print a record
    # shaped like a successful emission either. write_record's destination is
    # the run directory, so check there rather than at a caller-chosen path.
    assert not (run_dir / "baseline_record.json").exists()
    assert "energy_hartree" not in captured.out
