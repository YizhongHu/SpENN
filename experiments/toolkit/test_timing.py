"""Deterministic fixtures for the fail-loud timing reducer."""

from __future__ import annotations

import pytest

from experiments.toolkit.timing import (
    IDENTITY_FIELDS,
    REQUIRED_TRAIN_METRICS,
    TimingReductionError,
    convergence_receipt,
    provenance_from_metadata,
    reduce_attempt,
    validate_attempts,
)


def _rows(steps=(0, 2, 4), *, phase=0.1):
    rows = []
    for step in steps:
        rows.append({"step": step, "namespace": "train/perf", "metric": "step_time_sec", "value": 1.0})
        for name in ("sampling", "batch_build", "local_energy", "forward", "objective", "backward", "optimizer_step", "post_step_metrics"):
            rows.append({"step": step, "namespace": "train/perf", "metric": f"{name}_time_sec", "value": phase})
    return rows


def test_sparse_cadence_aligns_by_durable_step_and_excludes_positional_warmup():
    result = reduce_attempt(_rows(), run_id="r", attempt_id="a", stage="train", warmup_steps=1)
    assert result["n_steps"] == 3
    assert result["n_steps_measured"] == 2
    assert result["step_time_sec_median"] == pytest.approx(1.0)


def test_missing_metric_and_empty_phases_fail_loudly():
    with pytest.raises(TimingReductionError, match="required timing metric"):
        reduce_attempt([_rows()[1]], run_id="r", attempt_id="a", stage="train", warmup_steps=0)
    with pytest.raises(TimingReductionError, match="empty training phase"):
        reduce_attempt([], run_id="r", attempt_id="a", stage="train", warmup_steps=0)


def test_comparable_training_requires_all_eight_phases():
    rows = [row for row in _rows(steps=(0,)) if row["metric"] != "post_step_metrics_time_sec"]
    with pytest.raises(TimingReductionError, match="required timing metric absent"):
        reduce_attempt(rows, run_id="r", attempt_id="a", stage="train", warmup_steps=0)
    result = reduce_attempt(_rows(steps=(0,)), run_id="r", attempt_id="a", stage="train", warmup_steps=0)
    assert all(f"{phase}_time_sec_median" in result for phase in (
        "sampling", "batch_build", "local_energy", "forward", "objective",
        "backward", "optimizer_step", "post_step_metrics",
    ))


def test_missing_authoritative_identity_fails_and_explicit_metadata_maps():
    with pytest.raises(TimingReductionError, match="required timing identity absent"):
        from experiments.toolkit.timing import require_identity
        require_identity({}, {})
    metadata = {
        "git_sha": "abc", "resolved_timing_mode": "device_event", "hostname": "node-a",
        "device_uuid": "GPU-a", "device_model": "A100", "process_packing": "1xGPU",
        "partition": "test", "allocation": {"device_count": 1, "allocated_wall_time_sec": 30},
    }
    provenance, allocation = provenance_from_metadata(metadata)
    assert set(IDENTITY_FIELDS) <= set(provenance) | set(allocation)


def test_negative_residual_fails_loudly():
    rows = _rows(steps=(0,), phase=0.2)
    rows[1]["value"] = 0.2
    with pytest.raises(TimingReductionError, match="negative unclassified"):
        reduce_attempt(rows, run_id="r", attempt_id="a", stage="train", warmup_steps=0, clocks_comparable=True)


def test_normalized_throughput_and_mixed_hardware_retry_rejection():
    result = reduce_attempt(_rows(), run_id="r", attempt_id="a", stage="train", warmup_steps=0, sample_count=100, walker_count=4)
    assert result["samples_per_sec"] == pytest.approx(100.0)
    assert result["samples_per_walker_sec"] == pytest.approx(25.0)
    with pytest.raises(TimingReductionError, match="mixed device_uuid"):
        validate_attempts([{"run_id": "r", "attempt_id": "a", "device_uuid": "x"}, {"run_id": "s", "attempt_id": "b", "device_uuid": "y"}])
    with pytest.raises(TimingReductionError, match="duplicate attempt"):
        validate_attempts([{"run_id": "r", "attempt_id": "a"}, {"run_id": "r", "attempt_id": "a"}])


def test_censored_target_is_preserved_without_inventing_precision():
    receipt = convergence_receipt(target="mcse<=0.01", reached=False)
    assert receipt["target_status"] == "censored"
    assert "time_to_target_sec" not in receipt
    assert convergence_receipt(target=None, reached=False)["target_status"] == "not_declared"
