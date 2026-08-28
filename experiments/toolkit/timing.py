"""Fail-loud timing reduction for explicitly supplied experiment artifacts.

This module owns timing semantics.  It accepts expanded metrics rows only; it
never discovers run directories or fills missing measurements with zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

TRAIN_PHASES = (
    "sampling", "batch_build", "local_energy", "forward", "objective",
    "backward", "optimizer_step", "post_step_metrics",
)
REQUIRED_TRAIN_METRICS = ("step_time_sec", *[f"{phase}_time_sec" for phase in TRAIN_PHASES])
IDENTITY_FIELDS = (
    "git_sha", "timing_mode", "device_model", "device_uuid", "hostname",
    "process_packing", "partition", "device_count", "allocated_wall_time_sec",
)


class TimingReductionError(ValueError):
    """Raised when an explicit timing artifact cannot support a summary."""


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TimingReductionError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise TimingReductionError(f"{name} must be finite, got {value!r}")
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TimingReductionError("cannot summarize an empty timing table")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise TimingReductionError("cannot summarize an empty timing table")
    return {
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.5),
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "p95": _quantile(values, 0.95),
    }


def _warmup(value: Any) -> int:
    count = _number(value, "warmup_steps")
    if count < 0 or count != int(count):
        raise TimingReductionError(f"warmup_steps must be a non-negative integer, got {value!r}")
    return int(count)


def reduce_attempt(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    attempt_id: str,
    stage: str,
    warmup_steps: int,
    required_metrics: Sequence[str] = REQUIRED_TRAIN_METRICS,
    provenance: Mapping[str, Any] | None = None,
    sample_count: Any | None = None,
    walker_count: Any | None = None,
    clocks_comparable: bool = False,
    residual_tolerance_sec: float = 1e-9,
) -> dict[str, Any]:
    """Reduce one explicit metrics table, aligned by durable step.

    ``warmup_steps`` is positional in the recorded-step order and is required
    deliberately: the reducer never guesses a burn-in boundary.
    """
    warmup = _warmup(warmup_steps)
    by_step: dict[Any, dict[str, float]] = defaultdict(dict)
    for row in metrics_rows:
        namespace = str(row.get("namespace", "")).strip("/")
        metric = str(row.get("metric", ""))
        if namespace != "train/perf":
            continue
        step = row.get("step")
        if step in (None, ""):
            raise TimingReductionError("training timing row has no durable step")
        key = metric.removesuffix("_sec")
        value = _number(row.get("value"), f"train/perf/{metric}")
        if value < 0:
            raise TimingReductionError(f"negative timing for {metric} at step {step}")
        by_step[step][metric] = value
    if not by_step:
        raise TimingReductionError("empty training phase table")
    ordered = sorted(by_step, key=lambda value: (isinstance(value, str), value))
    missing = [metric for step in ordered for metric in required_metrics if metric not in by_step[step]]
    if missing:
        raise TimingReductionError(f"required timing metric absent: {missing[0]}")
    measured_steps = ordered[warmup:]
    if not measured_steps:
        raise TimingReductionError("warmup excludes every recorded step")
    values = {metric: [by_step[step][metric] for step in measured_steps if metric in by_step[step]]
              for metric in ("step_time_sec", *[f"{phase}_time_sec" for phase in TRAIN_PHASES])}
    result: dict[str, Any] = {
        "run_id": run_id, "attempt_id": attempt_id, "stage": stage,
        "n_steps": len(ordered), "warmup_steps": warmup,
        "n_steps_measured": len(measured_steps),
        "phase_table_steps": len(ordered),
    }
    for metric, series in values.items():
        if not series:
            continue
        result[f"{metric}_mean"] = _stats(series)["mean"]
        result[f"{metric}_median"] = _stats(series)["median"]
        result[f"{metric}_iqr"] = _stats(series)["iqr"]
        result[f"{metric}_p95"] = _stats(series)["p95"]
    total = values["step_time_sec"]
    phases = [values[f"{phase}_time_sec"] for phase in TRAIN_PHASES]
    if clocks_comparable and all(len(series) == len(measured_steps) for series in phases):
        residuals = [total[index] - sum(series[index] for series in phases) for index in range(len(total))]
        if any(value < -residual_tolerance_sec for value in residuals):
            raise TimingReductionError("negative unclassified step residual")
        result["unclassified_step_residual_median"] = _stats(residuals)["median"]
    if sample_count is not None:
        samples = _number(sample_count, "sample_count")
        if samples <= 0:
            raise TimingReductionError("sample_count must be positive")
        result["samples_per_sec"] = samples / result["step_time_sec_median"]
        if walker_count is not None:
            walkers = _number(walker_count, "walker_count")
            if walkers <= 0:
                raise TimingReductionError("walker_count must be positive")
            result["samples_per_walker_sec"] = samples / walkers / result["step_time_sec_median"]
    result.update({key: value for key, value in (provenance or {}).items()})
    return result


def _first_explicit(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def provenance_from_metadata(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract only explicitly recorded provenance and allocation receipts.

    Nested sections are accepted because they are named metadata contracts;
    filesystem names, hostnames, device types, and partition names are never
    inferred.  Missing required values remain absent for strict validation.
    """
    runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), Mapping) else {}
    scheduler = metadata.get("scheduler") if isinstance(metadata.get("scheduler"), Mapping) else {}
    receipt = metadata.get("allocation") if isinstance(metadata.get("allocation"), Mapping) else {}
    provenance = {
        "git_sha": _first_explicit(metadata, ("git_sha", "git_commit", "commit_sha")),
        "timing_mode": _first_explicit(metadata, ("resolved_timing_mode", "timing_mode")),
        "device_model": _first_explicit(metadata, ("device_model", "device_name")) or _first_explicit(runtime, ("device_model", "device_name")),
        "device_uuid": _first_explicit(metadata, ("device_uuid",)) or _first_explicit(runtime, ("device_uuid",)),
        "hostname": _first_explicit(metadata, ("hostname", "host")) or _first_explicit(runtime, ("hostname", "host")),
        "process_packing": _first_explicit(metadata, ("process_packing", "packing")),
        "partition": _first_explicit(metadata, ("partition", "slurm_partition")) or _first_explicit(scheduler, ("partition", "slurm_partition")),
        "slurm_job_id": _first_explicit(metadata, ("slurm_job_id", "job_id")) or _first_explicit(scheduler, ("slurm_job_id", "job_id")),
    }
    allocation = {
        "device_name": _first_explicit(receipt, ("device_name", "device_model")) or provenance["device_model"],
        "device_count": _first_explicit(receipt, ("device_count", "delivered_device_count")) or _first_explicit(metadata, ("device_count", "delivered_device_count")),
        "allocated_wall_time_sec": _first_explicit(receipt, ("allocated_wall_time_sec", "allocation_wall_time_sec")) or _first_explicit(metadata, ("allocated_wall_time_sec", "allocation_wall_time_sec")),
    }
    return {key: value for key, value in provenance.items() if value not in (None, "")}, {key: value for key, value in allocation.items() if value not in (None, "")}


def require_identity(provenance: Mapping[str, Any], allocation: Mapping[str, Any]) -> None:
    """Fail closed when a comparable timing row lacks its authoritative join."""
    def present(source: Mapping[str, Any], field: str) -> bool:
        value = source.get(field)
        return value is not None and str(value).strip() != ""

    missing = [field for field in IDENTITY_FIELDS if not present(provenance, field) and not present(allocation, field)]
    if missing:
        raise TimingReductionError(f"required timing identity absent: {', '.join(missing)}")
    for field in ("device_count", "allocated_wall_time_sec"):
        if not present(allocation, field):
            raise TimingReductionError(f"required allocation receipt absent: {field}")


def validate_attempts(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate attempts and mixed comparison identity fields."""
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("run_id", "")), str(row.get("attempt_id", "")))
        if key in seen:
            raise TimingReductionError(f"duplicate attempt: {key[0]}/{key[1]}")
        seen.add(key)
    for field in ("git_sha", "timing_mode", "device_uuid", "device_model"):
        values = {str(row[field]) for row in rows if str(row.get(field, "")).strip()}
        if len(values) > 1:
            raise TimingReductionError(f"mixed {field} values")


def convergence_receipt(
    *,
    target: Any | None,
    reached: bool,
    elapsed_sec: Any | None = None,
    mcse: Any | None = None,
) -> dict[str, Any]:
    """Return an explicit convergence join without fabricating a target.

    A missing target is represented as ``target_status='not_declared'``; a
    declared but unreached target is censored rather than converted to a false
    elapsed time or an invented precision threshold.
    """
    if target is None or not str(target).strip():
        return {"target_status": "not_declared", "reached": ""}
    result: dict[str, Any] = {"target": target, "reached": bool(reached)}
    if reached:
        if elapsed_sec is None:
            raise TimingReductionError("reached convergence target has no elapsed time")
        result["time_to_target_sec"] = _number(elapsed_sec, "elapsed_sec")
        if mcse is not None:
            result["mcse"] = _number(mcse, "mcse")
        result["target_status"] = "reached"
    else:
        result["target_status"] = "censored"
    return result
