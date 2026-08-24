"""Fixture tests for the compact cost-projection tables."""

from __future__ import annotations

import pytest

from experiments.toolkit.cost import (
    artifact_timing_comparison,
    cost_by_axis_rows,
    cost_by_run_row,
    cost_by_task_rows,
)


def _train_metrics_rows():
    rows = [
        {"step": 0, "namespace": "runtime", "metric": "wall_time_sec", "value": 100.0},
        {"step": 0, "namespace": "runtime", "metric": "peak_memory_mb", "value": 512.0},
    ]
    for step, (step_time, sampling, forward) in enumerate([(1.0, 0.4, 0.2), (3.0, 1.2, 0.6), (2.0, 0.8, 0.4)]):
        rows.append({"step": step, "namespace": "train/perf", "metric": "step_time_sec", "value": step_time})
        rows.append({"step": step, "namespace": "train/perf", "metric": "sampling_time_sec", "value": sampling})
        rows.append({"step": step, "namespace": "train/perf", "metric": "forward_time_sec", "value": forward})
    return rows


def test_cost_by_run_row_projects_runtime_and_step_statistics() -> None:
    row = cost_by_run_row(
        _train_metrics_rows(),
        run_id="run-a",
        attempt_id="A0",
        stage="train",
        status="completed",
        device_type="cpu",
        axes={"basis": "B00", "lr": "3e-4"},
    )

    assert row["run_id"] == "run-a"
    assert row["stage"] == "train"
    assert row["wall_time_sec"] == 100.0
    assert row["peak_memory_mb"] == 512.0
    assert row["device_peak_memory_allocated_mb"] == ""
    assert row["device_peak_memory_reserved_mb"] == ""
    assert row["device_peak_memory_available"] is False
    assert row["n_steps"] == "3"
    assert float(row["mean_step_time_sec"]) == pytest.approx(2.0)
    assert float(row["median_step_time_sec"]) == pytest.approx(2.0)
    assert float(row["p95_step_time_sec"]) == pytest.approx(2.9)
    assert float(row["mean_sampling_time_sec"]) == pytest.approx(0.8)
    assert float(row["mean_forward_time_sec"]) == pytest.approx(0.4)
    assert row["mean_backward_time_sec"] == ""  # not emitted -> blank
    assert row["basis"] == "B00"
    assert row["lr"] == "3e-4"


def test_cost_by_run_row_default_warmup_excludes_nothing() -> None:
    default = cost_by_run_row(_train_metrics_rows(), run_id="run-a", attempt_id="A0", stage="train")
    explicit = cost_by_run_row(
        _train_metrics_rows(), run_id="run-a", attempt_id="A0", stage="train", warmup_steps=0
    )

    assert default == explicit
    assert default["warmup_steps"] == "0"
    assert default["n_steps"] == default["n_steps_measured"] == "3"
    assert float(default["mean_step_time_sec"]) == pytest.approx(2.0)


def test_cost_by_run_row_warmup_excludes_leading_samples() -> None:
    row = cost_by_run_row(
        _train_metrics_rows(), run_id="run-a", attempt_id="A0", stage="train", warmup_steps=1
    )

    # Dropping the first recorded sample leaves step times [3.0, 2.0].
    assert row["n_steps"] == "3"
    assert row["n_steps_measured"] == "2"
    assert row["warmup_steps"] == "1"
    assert float(row["mean_step_time_sec"]) == pytest.approx(2.5)
    assert float(row["median_step_time_sec"]) == pytest.approx(2.5)
    assert float(row["p95_step_time_sec"]) == pytest.approx(2.95)


def test_cost_by_run_row_warmup_excludes_phase_series_too() -> None:
    row = cost_by_run_row(
        _train_metrics_rows(), run_id="run-a", attempt_id="A0", stage="train", warmup_steps=1
    )

    # sampling [0.4, 1.2, 0.8] -> [1.2, 0.8]; forward [0.2, 0.6, 0.4] -> [0.6, 0.4].
    assert float(row["mean_sampling_time_sec"]) == pytest.approx(1.0)
    assert float(row["mean_forward_time_sec"]) == pytest.approx(0.5)


def test_cost_by_run_row_over_exclusion_blanks_aggregates() -> None:
    row = cost_by_run_row(
        _train_metrics_rows(), run_id="run-a", attempt_id="A0", stage="train", warmup_steps=5
    )

    # Blank aggregates beside an explicit zero count are visible; an unfiltered
    # mean silently substituted for the empty window would not be.
    assert row["n_steps"] == "3"
    assert row["n_steps_measured"] == "0"
    assert row["mean_step_time_sec"] == ""
    assert row["median_step_time_sec"] == ""
    assert row["p95_step_time_sec"] == ""
    assert row["mean_sampling_time_sec"] == ""


@pytest.mark.parametrize(
    ("warmup_steps", "match"),
    [
        (-1, "non-negative"),
        (1.5, "whole number"),
        ("two", "non-negative"),
        (None, "non-negative"),
    ],
)
def test_cost_by_run_row_rejects_unusable_warmup_steps(warmup_steps, match: str) -> None:
    # A broken warmup parameter must not silently report a window nobody asked for.
    with pytest.raises(ValueError, match=match):
        cost_by_run_row(
            _train_metrics_rows(),
            run_id="run-a",
            attempt_id="A0",
            stage="train",
            warmup_steps=warmup_steps,
        )


def test_cost_by_run_row_projects_explicit_timing_provenance() -> None:
    provenance = {
        "timing_mode": "device_event",
        "hostname": "holy8a24101",
        "slurm_job_id": "39221900",
        "device_uuid": "GPU-123",
    }

    row = cost_by_run_row(
        [],
        run_id="run-a",
        attempt_id="A0",
        stage="train",
        provenance=provenance,
    )

    assert {key: row[key] for key in provenance} == provenance


def test_cost_by_run_row_derives_gpu_seconds_from_delivered_allocation() -> None:
    allocation = {
        "device_name": "NVIDIA A100-SXM4-40GB",
        "device_count": 4,
        "allocated_wall_time_sec": 1800.0,
    }

    row = cost_by_run_row(
        [],
        run_id="run-a",
        attempt_id="A0",
        stage="train",
        allocation=allocation,
    )

    # Delivered facts are echoed, never normalized: the card string in particular
    # is the delivered device, which can differ from the advertised partition GRES.
    assert {key: row[key] for key in allocation} == allocation
    assert float(row["gpu_seconds"]) == pytest.approx(7200.0)


def test_cost_by_run_row_without_allocation_leaves_receipt_blank() -> None:
    row = cost_by_run_row([], run_id="run-a", attempt_id="A0", stage="train")

    assert row["device_name"] == ""
    assert row["device_count"] == ""
    assert row["allocated_wall_time_sec"] == ""
    assert row["gpu_seconds"] == ""


@pytest.mark.parametrize(
    "allocation",
    [
        {"device_count": 4},
        {"allocated_wall_time_sec": 1800.0},
        {"device_name": "NVIDIA A100-SXM4-40GB"},
    ],
)
def test_cost_by_run_row_partial_allocation_reports_no_gpu_seconds(allocation) -> None:
    row = cost_by_run_row([], run_id="run-a", attempt_id="A0", stage="train", allocation=allocation)

    # Half a receipt must not imply the missing half (e.g. a single device).
    assert row["gpu_seconds"] == ""
    assert {key: row[key] for key in allocation} == allocation


@pytest.mark.parametrize(
    ("allocation", "match"),
    [
        ({"device_count": 0, "allocated_wall_time_sec": 1800.0}, "device_count"),
        ({"device_count": -1, "allocated_wall_time_sec": 1800.0}, "device_count"),
        ({"device_count": 2.5, "allocated_wall_time_sec": 1800.0}, "whole number"),
        ({"device_count": "four", "allocated_wall_time_sec": 1800.0}, "device_count"),
        ({"device_count": 4, "allocated_wall_time_sec": 0}, "allocated_wall_time_sec"),
        ({"device_count": 4, "allocated_wall_time_sec": "30m"}, "allocated_wall_time_sec"),
        ({"gpu_seconds": 7200.0}, "derived"),
        ({"devices": 4}, "unsupported allocation field"),
    ],
)
def test_cost_by_run_row_rejects_unusable_allocation(allocation, match: str) -> None:
    # A broken or misspelled receipt understates cost if it silently blanks.
    with pytest.raises(ValueError, match=match):
        cost_by_run_row([], run_id="run-a", attempt_id="A0", stage="train", allocation=allocation)


def test_cost_by_run_row_without_metrics_leaves_blanks() -> None:
    row = cost_by_run_row([], run_id="run-a", attempt_id="A0", stage="validation")

    assert row["wall_time_sec"] == ""
    assert row["n_steps"] == ""
    assert row["mean_step_time_sec"] == ""


def test_cost_by_run_row_reports_device_peaks_only_when_both_were_measured() -> None:
    rows = [
        {
            "step": 0,
            "namespace": "runtime",
            "metric": "cuda_max_memory_allocated_mb",
            "value": 128.0,
        },
        {
            "step": 0,
            "namespace": "runtime",
            "metric": "cuda_max_memory_reserved_mb",
            "value": 256.0,
        },
    ]

    row = cost_by_run_row(rows, run_id="run-a", attempt_id="A0", stage="eval")

    assert row["device_peak_memory_available"] is True
    assert row["device_peak_memory_allocated_mb"] == 128.0
    assert row["device_peak_memory_reserved_mb"] == 256.0


def test_cost_by_run_row_rejects_partial_device_peak_receipt() -> None:
    rows = [
        {
            "step": 0,
            "namespace": "runtime",
            "metric": "cuda_max_memory_allocated_mb",
            "value": 128.0,
        }
    ]

    with pytest.raises(ValueError, match="peak-memory receipt is incomplete"):
        cost_by_run_row(rows, run_id="run-a", attempt_id="A0", stage="eval")


def test_cost_by_task_rows_merges_task_time_and_components() -> None:
    metrics_rows = [
        {"step": 0, "namespace": "diagnostics/tail", "metric": "time_sec", "value": 4.0},
        {"step": 0, "namespace": "eval/perf/tail", "metric": "generator_time_sec", "value": 1.0},
        {"step": 0, "namespace": "eval/perf/tail", "metric": "calculator/energy_time_sec", "value": 0.5},
        {"step": 0, "namespace": "eval/perf/tail", "metric": "calculator/variance_time_sec", "value": 0.25},
        {"step": 0, "namespace": "eval/perf/tail", "metric": "summary/histogram_time_sec", "value": 0.125},
        {"step": 0, "namespace": "diagnostics/cusp", "metric": "time_sec", "value": 2.0},
        {"step": 0, "namespace": "eval/tail", "metric": "outlier_fraction", "value": 0.1},
    ]

    rows = cost_by_task_rows(
        metrics_rows, run_id="run-a", attempt_id="A0", stage="validation", device_type="cpu"
    )

    assert [row["task_name"] for row in rows] == ["cusp", "tail"]
    cusp, tail = rows
    assert float(cusp["time_sec"]) == pytest.approx(2.0)
    assert cusp["generator_time_sec"] == ""
    assert float(tail["time_sec"]) == pytest.approx(4.0)
    assert float(tail["generator_time_sec"]) == pytest.approx(1.0)
    assert float(tail["calculator_time_sec"]) == pytest.approx(0.75)
    assert float(tail["summary_time_sec"]) == pytest.approx(0.125)


def test_cost_by_task_rows_reports_throughput_provenance_and_timing_reconciliation() -> None:
    metrics_rows = [
        {"step": 0, "namespace": "diagnostics/mcmc_energy", "metric": "time_sec", "value": 10.0},
        {"step": 0, "namespace": "eval/perf/mcmc_energy", "metric": "generator_time_sec", "value": 4.0},
        {
            "step": 0,
            "namespace": "eval/perf/mcmc_energy",
            "metric": "calculator/local_energy_time_sec",
            "value": 2.0,
        },
        {
            "step": 0,
            "namespace": "eval/perf/mcmc_energy",
            "metric": "summary/sampler_stats_time_sec",
            "value": 1.0,
        },
        {
            "step": 0,
            "namespace": "eval/perf/mcmc_energy",
            "metric": "summary/sampled_records_time_sec",
            "value": 0.5,
        },
        {
            "step": 0,
            "namespace": "eval/mcmc_energy",
            "metric": "sampler_trajectory_retained_value_count",
            "value": 12,
        },
        {
            "step": 0,
            "namespace": "eval/mcmc_energy",
            "metric": "sampler_trajectory_discarded_value_count",
            "value": 4,
        },
    ]
    provenance = {
        "timing_mode": "host_wall_unsynchronized",
        "hostname": "holy8a24101",
        "slurm_job_id": "123",
        "device_uuid": "cpu",
    }

    (row,) = cost_by_task_rows(
        metrics_rows,
        run_id="run-a",
        attempt_id="A0",
        stage="eval",
        device_type="cpu",
        provenance=provenance,
        allocation={
            "device_name": "cpu",
            "device_count": 4,
            "allocated_wall_time_sec": 30.0,
        },
    )

    # Both sampled_records and sampler_stats publish artifacts in He-v1.
    assert float(row["writer_summary_time_sec"]) == pytest.approx(1.5)
    assert float(row["component_time_sec"]) == pytest.approx(7.5)
    assert float(row["unattributed_time_sec"]) == pytest.approx(2.5)
    assert row["timing_reconciled"] is True
    assert float(row["value_count"]) == pytest.approx(16.0)
    assert float(row["values_per_sec"]) == pytest.approx(4.0)
    assert row["values_per_sec_denominator"] == "generator_time_sec"
    assert {key: row[key] for key in provenance} == provenance
    assert row["device_name"] == "cpu"
    assert row["device_count"] == 4
    assert row["allocated_wall_time_sec"] == 30.0
    # Four CPU workers are allocation provenance, never four GPUs.
    assert row["gpu_seconds"] == ""


def test_cost_by_task_rows_retains_complete_delivered_gpu_allocation() -> None:
    """Task throughput keeps the same delivered-allocation facts as run cost."""

    (row,) = cost_by_task_rows(
        [
            {
                "step": 0,
                "namespace": "diagnostics/energy",
                "metric": "time_sec",
                "value": 1.0,
            }
        ],
        run_id="run-a",
        attempt_id="A0",
        stage="eval",
        device_type="cuda",
        allocation={
            "device_name": "NVIDIA A100-SXM4-40GB",
            "device_count": 2,
            "allocated_wall_time_sec": 30.0,
        },
    )

    assert row["device_name"] == "NVIDIA A100-SXM4-40GB"
    assert row["device_count"] == 2
    assert row["allocated_wall_time_sec"] == 30.0
    assert float(row["gpu_seconds"]) == pytest.approx(60.0)


def test_cost_by_task_rows_preserves_negative_timing_residual() -> None:
    metrics_rows = [
        {"step": 0, "namespace": "diagnostics/energy", "metric": "time_sec", "value": 1.0},
        {"step": 0, "namespace": "eval/perf/energy", "metric": "generator_time_sec", "value": 2.0},
    ]

    (row,) = cost_by_task_rows(
        metrics_rows,
        run_id="run-a",
        attempt_id="A0",
        stage="eval",
    )

    assert float(row["unattributed_time_sec"]) == pytest.approx(-1.0)
    assert row["timing_reconciled"] is False


def test_cost_by_task_rows_rejects_partial_trajectory_value_counts() -> None:
    """Throughput is unavailable when either measurement region is missing."""

    metrics_rows = [
        {
            "step": 0,
            "namespace": "eval/perf/energy",
            "metric": "generator_time_sec",
            "value": 2.0,
        },
        {
            "step": 0,
            "namespace": "eval/energy",
            "metric": "sampler_trajectory_retained_value_count",
            "value": 8,
        },
    ]

    with pytest.raises(ValueError, match="value-count receipt.*incomplete"):
        cost_by_task_rows(
            metrics_rows,
            run_id="run-a",
            attempt_id="A0",
            stage="eval",
        )


def test_repeated_interleaved_artifact_timing_keeps_signed_deltas() -> None:
    rows = [
        {"measurement_index": 0, "measurement_region": "measured", "artifact_mode": "off", "time_sec": 10.0, "shape": "4x8", "host": "h1"},
        {"measurement_index": 1, "measurement_region": "measured", "artifact_mode": "on", "time_sec": 9.0, "shape": "4x8", "host": "h1"},
        {"measurement_index": 2, "measurement_region": "measured", "artifact_mode": "off", "time_sec": 10.0, "shape": "4x8", "host": "h1"},
        {"measurement_index": 3, "measurement_region": "measured", "artifact_mode": "on", "time_sec": 12.0, "shape": "4x8", "host": "h1"},
        {"measurement_index": 4, "measurement_region": "measured", "artifact_mode": "off", "time_sec": 11.0, "shape": "4x8", "host": "h1"},
        {"measurement_index": 5, "measurement_region": "measured", "artifact_mode": "on", "time_sec": 12.0, "shape": "4x8", "host": "h1"},
    ]

    result = artifact_timing_comparison(rows, comparison_fields=("shape", "host"))

    assert result["status"] == "repeated_comparable_delta"
    assert result["repeated_comparable"] is True
    assert result["measurement_region"] == "measured"
    assert result["comparison_fields"] == {"shape": "4x8", "host": "h1"}
    assert result["artifact_delta_sec_by_pair"] == (-1.0, 2.0, 1.0)
    assert result["artifact_delta_sec_min"] == pytest.approx(-1.0)
    assert result["artifact_delta_sec_median"] == pytest.approx(1.0)
    assert result["artifact_delta_sec_spread"] == pytest.approx(3.0)


def test_single_artifact_pair_is_a_signed_delta_not_an_overhead_claim() -> None:
    result = artifact_timing_comparison(
        [
            {"measurement_index": 0, "measurement_region": "measured", "artifact_mode": "off", "time_sec": 5.0, "shape": "same"},
            {"measurement_index": 1, "measurement_region": "measured", "artifact_mode": "on", "time_sec": 4.0, "shape": "same"},
        ],
        comparison_fields=("shape",),
    )

    assert result["status"] == "single_observation_delta"
    assert result["repeated_comparable"] is False
    assert result["artifact_delta_sec_by_pair"] == (-1.0,)


def test_artifact_timing_rejects_noninterleaved_or_noncomparable_measurements() -> None:
    base = [
        {"measurement_index": 0, "measurement_region": "measured", "artifact_mode": "off", "time_sec": 5.0, "host": "h1"},
        {"measurement_index": 1, "measurement_region": "measured", "artifact_mode": "on", "time_sec": 6.0, "host": "h2"},
    ]
    with pytest.raises(ValueError, match="not comparable"):
        artifact_timing_comparison(base, comparison_fields=("host",))
    with pytest.raises(ValueError, match="alternate"):
        artifact_timing_comparison(
            [dict(base[0]), {**base[1], "artifact_mode": "off", "host": "h1"}],
            comparison_fields=("host",),
        )


def test_artifact_timing_rejects_warmup_rows_as_measurement_evidence() -> None:
    """Warmup stays visible and cannot qualify an artifact comparison."""

    with pytest.raises(ValueError, match="only rows explicitly marked"):
        artifact_timing_comparison(
            [
                {
                    "measurement_index": 0,
                    "measurement_region": "warmup",
                    "artifact_mode": "off",
                    "time_sec": 5.0,
                    "shape": "same",
                },
                {
                    "measurement_index": 1,
                    "measurement_region": "measured",
                    "artifact_mode": "on",
                    "time_sec": 6.0,
                    "shape": "same",
                },
            ],
            comparison_fields=("shape",),
        )


def test_cost_by_axis_rows_groups_by_each_axis_value() -> None:
    cost_rows = [
        {
            "stage": "train",
            "basis": "B00",
            "wall_time_sec": "100",
            "median_step_time_sec": "2.0",
            "mean_local_energy_time_sec": "0.5",
            "mean_forward_time_sec": "0.2",
            "mean_backward_time_sec": "0.3",
            "peak_memory_mb": "512",
        },
        {
            "stage": "train",
            "basis": "B00",
            "wall_time_sec": "300",
            "median_step_time_sec": "4.0",
            "mean_local_energy_time_sec": "1.5",
            "mean_forward_time_sec": "0.6",
            "mean_backward_time_sec": "0.9",
            "peak_memory_mb": "1024",
        },
        {
            "stage": "train",
            "basis": "B01",
            "wall_time_sec": "50",
            "median_step_time_sec": "1.0",
            "mean_local_energy_time_sec": "0.25",
            "mean_forward_time_sec": "0.1",
            "mean_backward_time_sec": "0.15",
            "peak_memory_mb": "256",
        },
    ]

    rows = cost_by_axis_rows(cost_rows, axis_names=["basis", "absent_axis"])

    assert [(row["axis_name"], row["axis_value"]) for row in rows] == [("basis", "B00"), ("basis", "B01")]
    b00 = rows[0]
    assert b00["n_runs"] == "2"
    assert float(b00["wall_time_sec_median"]) == pytest.approx(200.0)
    assert float(b00["wall_time_sec_q25"]) == pytest.approx(150.0)
    assert float(b00["wall_time_sec_q75"]) == pytest.approx(250.0)
    assert float(b00["step_time_sec_median"]) == pytest.approx(3.0)
    assert float(b00["peak_memory_mb_median"]) == pytest.approx(768.0)


def test_cost_by_axis_rows_handles_blank_metrics() -> None:
    rows = cost_by_axis_rows(
        [{"stage": "validation", "basis": "B00", "wall_time_sec": ""}],
        axis_names=["basis"],
    )

    assert rows[0]["wall_time_sec_median"] == ""
    assert rows[0]["n_runs"] == "1"


def test_cost_by_axis_rows_rolls_up_gpu_seconds_for_fully_populated_group() -> None:
    rows = cost_by_axis_rows(
        [
            {"stage": "train", "basis": "B00", "gpu_seconds": "400"},
            {"stage": "train", "basis": "B00", "gpu_seconds": "1200"},
            {"stage": "train", "basis": "B00", "gpu_seconds": "800"},
        ],
        axis_names=["basis"],
    )

    assert rows[0]["n_runs"] == "3"
    assert rows[0]["n_runs_with_gpu_seconds"] == "3"
    assert float(rows[0]["gpu_seconds_median"]) == pytest.approx(800.0)


def test_cost_by_axis_rows_counts_only_runs_carrying_gpu_seconds() -> None:
    rows = cost_by_axis_rows(
        [
            {"stage": "train", "basis": "B00", "gpu_seconds": "100"},
            {"stage": "train", "basis": "B00", "gpu_seconds": ""},
            {"stage": "train", "basis": "B00", "gpu_seconds": "300"},
            {"stage": "train", "basis": "B00"},
        ],
        axis_names=["basis"],
    )

    # The median uses only the two finite cells; the count is what tells a reader
    # the median describes a minority of the group.
    assert float(rows[0]["gpu_seconds_median"]) == pytest.approx(200.0)
    assert rows[0]["n_runs"] == "4"
    assert rows[0]["n_runs_with_gpu_seconds"] == "2"
    assert int(rows[0]["n_runs_with_gpu_seconds"]) < int(rows[0]["n_runs"])


def test_cost_by_axis_rows_reports_zero_gpu_seconds_contributors_when_all_blank() -> None:
    rows = cost_by_axis_rows(
        [
            {"stage": "train", "basis": "B00", "gpu_seconds": ""},
            {"stage": "train", "basis": "B00", "wall_time_sec": "100"},
        ],
        axis_names=["basis"],
    )

    assert rows[0]["gpu_seconds_median"] == ""
    assert rows[0]["n_runs_with_gpu_seconds"] == "0"
    assert rows[0]["n_runs"] == "2"
