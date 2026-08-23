"""Compact cost-projection tables from run metrics.

Projects declared timing/resource metrics (``runtime/*``, ``train/perf/*``,
``diagnostics/*/time_sec``, ``eval/perf/<task>/*``) from per-run metrics rows
into the compact cost tables recommended by the experiment profiling spec:
``cost_by_run.csv``, ``cost_by_axis.csv``, and ``cost_by_task.csv``. Pure
projection: callers read metrics rows with :mod:`experiments.toolkit.artifacts`
readers and pass them in; nothing here scans run directories, and unknown or
absent metrics simply leave blank cells.

Execution provenance and the delivered-hardware allocation receipt are supplied
by the caller and preserved verbatim; ``gpu_seconds`` is the one derived cost
column, computed as delivered device count times allocated wall time. Nothing
here contacts a scheduler, inspects the host, or infers a device count.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.toolkit.artifacts import metric_map
from experiments.toolkit.timing import (
    REQUIRED_TRAIN_METRICS,
    TRAIN_PHASES,
    reduce_attempt,
    require_identity,
)

# Delivered-hardware receipt fields a caller may supply per run/attempt. The
# delivered card can disagree with a partition's advertised GRES, so identity
# and count are recorded facts, never inferred from ``device_type`` or hostname.
ALLOCATION_FIELDS = ("device_name", "device_count", "allocated_wall_time_sec")

# ``gpu_seconds`` is derived from the receipt, so it is a column but not a field.
ALLOCATION_COLUMNS = (*ALLOCATION_FIELDS, "gpu_seconds")

# These component names are the He-v1 summaries whose normal invocation owns
# artifact publication. Streamed trajectory CSV writes remain generator work.
DEFAULT_WRITER_SUMMARY_NAMES = ("sampled_records", "sampler_stats")

COST_BY_RUN_BASE_COLUMNS = [
    "run_id",
    "attempt_id",
    "stage",
    "device_type",
    "status",
    "wall_time_sec",
    "peak_memory_mb",
    "device_peak_memory_allocated_mb",
    "device_peak_memory_reserved_mb",
    "device_peak_memory_available",
    "timing_mode",
    "git_sha",
    "hostname",
    "device_model",
    "partition",
    "process_packing",
    "slurm_job_id",
    "device_uuid",
    "device_name",
    "device_count",
    "allocated_wall_time_sec",
    "gpu_seconds",
    "n_steps",
    "warmup_steps",
    "n_steps_measured",
    "mean_step_time_sec",
    "median_step_time_sec",
    "iqr_step_time_sec",
    "p95_step_time_sec",
    *(name for phase in TRAIN_PHASES for name in (
        f"mean_{phase}_time_sec", f"median_{phase}_time_sec",
        f"iqr_{phase}_time_sec", f"p95_{phase}_time_sec",
    )),
]

COST_BY_AXIS_COLUMNS = [
    "stage",
    "axis_name",
    "axis_value",
    "n_runs",
    "wall_time_sec_median",
    "wall_time_sec_q25",
    "wall_time_sec_q75",
    "step_time_sec_median",
    "local_energy_time_sec_median",
    "forward_time_sec_median",
    "backward_time_sec_median",
    "peak_memory_mb_median",
    "gpu_seconds_median",
    "n_runs_with_gpu_seconds",
]

COST_BY_TASK_COLUMNS = [
    "run_id",
    "attempt_id",
    "stage",
    "task_name",
    "device_type",
    "time_sec",
    "generator_time_sec",
    "calculator_time_sec",
    "summary_time_sec",
    "writer_summary_time_sec",
    "component_time_sec",
    "unattributed_time_sec",
    "timing_reconciled",
    "value_count",
    "values_per_sec",
    "values_per_sec_denominator",
    "timing_mode",
    "hostname",
    "slurm_job_id",
    "device_uuid",
    "device_name",
    "device_count",
    "allocated_wall_time_sec",
    "gpu_seconds",
]


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def _finite(values: Sequence[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _format(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def _mean(values: Sequence[float]) -> float:
    finite = _finite(values)
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _quantile(values: Sequence[float], fraction: float) -> float:
    finite = sorted(_finite(values))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    position = fraction * (len(finite) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _per_step_values(metrics_rows: Sequence[Mapping[str, Any]], namespace: str, metric: str) -> list[float]:
    return [
        _as_float(row.get("value"))
        for row in metrics_rows
        if str(row.get("namespace", "")).strip("/") == namespace and row.get("metric") == metric
    ]


def _warmup_count(value: Any) -> int:
    """Return the number of leading samples to exclude, rejecting broken input.

    A negative, fractional, or non-numeric warmup count cannot describe a
    sample position, and silently coercing it would report an aggregate over a
    window nobody asked for.
    """
    parsed = _as_float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"warmup_steps must be a non-negative finite number, got {value!r}")
    if parsed != int(parsed):
        raise ValueError(f"warmup_steps must be a whole number, got {value!r}")
    return int(parsed)


def _steady_state(values: Sequence[float], warmup_steps: int) -> list[float]:
    """Return ``values`` with the first ``warmup_steps`` recorded samples dropped.

    Warmup is defined by POSITION in metrics-file order, never by comparing the
    ``step`` field against a threshold: cadences differ on whether the first
    recorded step is 0 or 1, so a positional rule is unambiguous under both.
    Over-exclusion is not an error; it yields an empty window, which surfaces as
    blank aggregates next to an explicit zero measured-step count.
    """
    return list(values[warmup_steps:])


def _allocated_quantity(field: str, value: Any) -> float:
    """Return a delivered allocation quantity, rejecting unusable receipts.

    Absence is a blank cell elsewhere in this module, but a present-yet-unusable
    quantity is a broken receipt: silently blanking it would understate cost.
    """
    parsed = _as_float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"allocation {field} must be a positive finite number, got {value!r}")
    if field == "device_count" and parsed != int(parsed):
        raise ValueError(f"allocation device_count must be a whole number, got {value!r}")
    return parsed


def _allocation_columns(
    allocation: Mapping[str, Any] | None,
    *,
    device_type: str = "",
) -> dict[str, Any]:
    """Return the delivered-allocation columns, deriving ``gpu_seconds``.

    ``gpu_seconds`` is arithmetic on two supplied facts, delivered device count
    times allocated wall time. Nothing here queries a scheduler, reads a run
    directory, or guesses a device count when only one of the two is supplied.
    """
    supplied = dict(allocation or {})
    if "gpu_seconds" in supplied:
        raise ValueError(
            "gpu_seconds is derived from device_count and allocated_wall_time_sec; do not supply it"
        )
    unknown = sorted(set(supplied) - set(ALLOCATION_FIELDS))
    if unknown:
        # A dropped receipt field would understate cost, so typos must not pass.
        raise ValueError(f"unsupported allocation field(s): {', '.join(unknown)}")
    columns: dict[str, Any] = dict.fromkeys(ALLOCATION_COLUMNS, "")
    columns["device_name"] = supplied.get("device_name", "")
    devices = seconds = None
    if supplied.get("device_count") is not None:
        devices = _allocated_quantity("device_count", supplied["device_count"])
        columns["device_count"] = supplied["device_count"]
    if supplied.get("allocated_wall_time_sec") is not None:
        seconds = _allocated_quantity("allocated_wall_time_sec", supplied["allocated_wall_time_sec"])
        columns["allocated_wall_time_sec"] = supplied["allocated_wall_time_sec"]
    # A CPU allocation may still carry its delivered worker count and wall
    # limit, but those facts are not GPU-seconds. Preserve them without
    # manufacturing a device-memory/cost reading for hardware that was absent.
    is_cpu = str(device_type).strip().lower().startswith("cpu")
    if devices is not None and seconds is not None and not is_cpu:
        columns["gpu_seconds"] = _format(devices * seconds)
    return columns


def cost_by_run_row(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    attempt_id: str,
    stage: str,
    status: str = "",
    device_type: str = "",
    axes: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    warmup_steps: int = 0,
    identity_required: bool = False,
) -> dict[str, Any]:
    """Return one ``cost_by_run.csv`` row projected from run metrics rows.

    Parameters
    ----------
    metrics_rows
        Expanded metrics rows for one run, as returned by
        :func:`experiments.toolkit.artifacts.read_metrics_jsonl`.
    axes
        Optional configured-axis columns appended verbatim to the row.
    provenance
        Optional execution metadata. The supported timing and placement fields
        are projected verbatim; absent fields remain blank.
    allocation
        Optional delivered-hardware receipt (``device_name``, ``device_count``,
        ``allocated_wall_time_sec``) echoed verbatim, from which ``gpu_seconds``
        is derived. A present-yet-unusable receipt raises rather than blanking.
    warmup_steps
        Number of leading recorded samples excluded from every ``train/perf``
        per-step series before aggregation, so the reported statistics describe
        steady state rather than compile and cache-warming cost. Counted by
        position in metrics-file order, not by ``step`` value. ``n_steps`` still
        reports the total recorded samples; ``n_steps_measured`` reports how
        many survived. A negative, fractional, or non-numeric value raises.
    """

    warmup = _warmup_count(warmup_steps)
    metric_values = metric_map(metrics_rows)
    device_peak_keys = (
        "runtime/cuda_max_memory_allocated_mb",
        "runtime/cuda_max_memory_reserved_mb",
    )
    device_peak_present = [key in metric_values for key in device_peak_keys]
    if any(device_peak_present) and not all(device_peak_present):
        raise ValueError("device peak-memory receipt is incomplete")
    device_peak_available = all(device_peak_present)
    step_times = _per_step_values(metrics_rows, "train/perf", "step_time_sec")
    measured_step_times = _steady_state(step_times, warmup)
    timing = (
        reduce_attempt(
            metrics_rows, run_id=run_id, attempt_id=attempt_id, stage=stage,
            warmup_steps=warmup,
            required_metrics=REQUIRED_TRAIN_METRICS if identity_required else ("step_time_sec",),
        )
        if step_times and measured_step_times else {}
    )
    has_training_timing = bool(step_times) or any(
        str(row.get("namespace", "")).strip("/") == "train/perf" for row in metrics_rows
    )
    if identity_required and has_training_timing:
        require_identity(provenance or {}, allocation or {})
    row: dict[str, Any] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "device_type": device_type,
        "status": status,
        "wall_time_sec": metric_values.get("runtime/wall_time_sec", ""),
        "peak_memory_mb": metric_values.get("runtime/peak_memory_mb", ""),
        # CUDA peaks are conditional measurements. On CPU these cells stay
        # blank and availability is false; zero would falsely claim a reading.
        "device_peak_memory_allocated_mb": metric_values.get(
            "runtime/cuda_max_memory_allocated_mb", ""
        ),
        "device_peak_memory_reserved_mb": metric_values.get(
            "runtime/cuda_max_memory_reserved_mb", ""
        ),
        "device_peak_memory_available": device_peak_available,
        "n_steps": str(len(step_times)) if step_times else "",
        "warmup_steps": str(warmup),
        # Blank only when nothing was recorded; an exhausted window reads "0".
        "n_steps_measured": str(len(measured_step_times)) if step_times else "",
        "mean_step_time_sec": _format(timing.get("step_time_sec_mean", _mean(measured_step_times))),
        "median_step_time_sec": _format(timing.get("step_time_sec_median", _median(measured_step_times))),
        "iqr_step_time_sec": _format(timing.get("step_time_sec_iqr", math.nan)),
        "p95_step_time_sec": _format(timing.get("step_time_sec_p95", _quantile(measured_step_times, 0.95))),
        **{
            key: (provenance or {}).get(key, "")
            for key in (
                "timing_mode", "git_sha", "hostname", "device_model", "partition",
                "process_packing", "slurm_job_id", "device_uuid",
            )
        },
        **_allocation_columns(allocation, device_type=device_type),
    }
    for phase in TRAIN_PHASES:
        values = _per_step_values(metrics_rows, "train/perf", f"{phase}_time_sec")
        # Same exclusion as the total step series, so phase means stay comparable.
        row[f"mean_{phase}_time_sec"] = _format(timing.get(f"{phase}_time_sec_mean", _mean(_steady_state(values, warmup))))
        row[f"median_{phase}_time_sec"] = _format(timing.get(f"{phase}_time_sec_median", math.nan))
        row[f"iqr_{phase}_time_sec"] = _format(timing.get(f"{phase}_time_sec_iqr", math.nan))
        row[f"p95_{phase}_time_sec"] = _format(timing.get(f"{phase}_time_sec_p95", math.nan))
    for axis, value in (axes or {}).items():
        row[str(axis)] = value
    return row


def cost_by_task_rows(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    attempt_id: str,
    stage: str,
    device_type: str = "",
    writer_summary_names: Sequence[str] = DEFAULT_WRITER_SUMMARY_NAMES,
    provenance: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    reconciliation_tolerance_sec: float = 1.0e-9,
) -> list[dict[str, Any]]:
    """Return per-evaluation-task ``cost_by_task.csv`` rows from run metrics.

    Task wall time comes from ``diagnostics/<task>/time_sec``; component times
    come from ``eval/perf/<task>/{generator_time_sec, calculator/<name>_time_sec,
    summary/<name>_time_sec}`` with calculator/summary components summed per
    task. Tasks appear if either source is present. Draw-value throughput is
    divided by generator wall time only when the task also emitted the complete
    typed trajectory value-count receipt. Writer time refers only to explicitly
    named summary components (by default ``sampled_records`` and
    ``sampler_stats``); streamed record I/O inside the generator remains in
    generator time and is assessed by an artifact on/off comparison instead of
    being silently reattributed.
    """

    tolerance = _as_float(reconciliation_tolerance_sec)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("reconciliation_tolerance_sec must be finite and non-negative")
    writer_names = {str(name).strip() for name in writer_summary_names}
    if "" in writer_names:
        raise ValueError("writer_summary_names must not contain blanks")
    metric_values = metric_map(metrics_rows)
    tasks: dict[str, dict[str, float]] = {}

    def task_entry(task_name: str) -> dict[str, float]:
        return tasks.setdefault(
            task_name,
            {
                "time_sec": math.nan,
                "generator": math.nan,
                "calculator": math.nan,
                "summary": math.nan,
                "writer_summary": math.nan,
                "retained_values": math.nan,
                "discarded_values": math.nan,
            },
        )

    for key, value in metric_values.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "diagnostics" and parts[2] == "time_sec":
            task_entry(parts[1])["time_sec"] = _as_float(value)
            continue
        if len(parts) >= 4 and parts[0] == "eval" and parts[1] == "perf":
            task_name = parts[2]
            component = "/".join(parts[3:])
            entry = task_entry(task_name)
            if component == "generator_time_sec":
                entry["generator"] = _as_float(value)
            elif component.startswith("calculator/") and component.endswith("_time_sec"):
                addend = _as_float(value)
                if math.isfinite(addend):
                    previous = entry["calculator"] if math.isfinite(entry["calculator"]) else 0.0
                    entry["calculator"] = addend + previous
            elif component.startswith("summary/") and component.endswith("_time_sec"):
                addend = _as_float(value)
                if math.isfinite(addend):
                    entry["summary"] = addend + (entry["summary"] if math.isfinite(entry["summary"]) else 0.0)
                    summary_name = component[len("summary/") : -len("_time_sec")]
                    if summary_name in writer_names:
                        entry["writer_summary"] = addend + (
                            entry["writer_summary"]
                            if math.isfinite(entry["writer_summary"])
                            else 0.0
                        )
            continue
        if len(parts) == 3 and parts[0] == "eval":
            task_name = parts[1]
            if parts[2] == "sampler_trajectory_retained_value_count":
                task_entry(task_name)["retained_values"] = _as_float(value)
            elif parts[2] == "sampler_trajectory_discarded_value_count":
                task_entry(task_name)["discarded_values"] = _as_float(value)

    provenance_values = {
        key: (provenance or {}).get(key, "")
        for key in ("timing_mode", "hostname", "slurm_job_id", "device_uuid")
    }
    allocation_values = _allocation_columns(allocation, device_type=device_type)
    output: list[dict[str, Any]] = []
    for task_name, entry in sorted(tasks.items()):
        components = _finite(
            [entry["generator"], entry["calculator"], entry["summary"]]
        )
        component_time = sum(components) if components else math.nan
        task_time = entry["time_sec"]
        unattributed = (
            task_time - component_time
            if math.isfinite(task_time) and math.isfinite(component_time)
            else math.nan
        )
        timing_reconciled: bool | str = ""
        if math.isfinite(unattributed):
            timing_reconciled = unattributed >= -tolerance
        retained_available = math.isfinite(entry["retained_values"])
        discarded_available = math.isfinite(entry["discarded_values"])
        if retained_available != discarded_available:
            raise ValueError(
                f"trajectory value-count receipt for task {task_name!r} is incomplete"
            )
        value_count = math.nan
        if retained_available:
            for label in ("retained_values", "discarded_values"):
                count = entry[label]
                if count < 0.0 or count != int(count):
                    raise ValueError(
                        f"trajectory {label} for task {task_name!r} must be a "
                        "non-negative whole number"
                    )
            value_count = entry["retained_values"] + entry["discarded_values"]
        values_per_sec = (
            value_count / entry["generator"]
            if math.isfinite(value_count) and entry["generator"] > 0.0
            else math.nan
        )
        output.append({
            "run_id": run_id,
            "attempt_id": attempt_id,
            "stage": stage,
            "task_name": task_name,
            "device_type": device_type,
            "time_sec": _format(entry["time_sec"]),
            "generator_time_sec": _format(entry["generator"]),
            "calculator_time_sec": _format(entry["calculator"]),
            "summary_time_sec": _format(entry["summary"]),
            "writer_summary_time_sec": _format(entry["writer_summary"]),
            "component_time_sec": _format(component_time),
            # Signed on purpose. A negative residual exposes inconsistent
            # boundary data instead of being clamped into a plausible zero.
            "unattributed_time_sec": _format(unattributed),
            "timing_reconciled": timing_reconciled,
            "value_count": _format(value_count),
            "values_per_sec": _format(values_per_sec),
            "values_per_sec_denominator": (
                "generator_time_sec" if math.isfinite(values_per_sec) else ""
            ),
            **provenance_values,
            "device_name": allocation_values["device_name"],
            "device_count": allocation_values["device_count"],
            "allocated_wall_time_sec": allocation_values["allocated_wall_time_sec"],
            "gpu_seconds": allocation_values["gpu_seconds"],
        })
    return output


def artifact_timing_comparison(
    measurements: Sequence[Mapping[str, Any]],
    *,
    comparison_fields: Sequence[str],
    order_key: str = "measurement_index",
    mode_key: str = "artifact_mode",
    time_key: str = "time_sec",
    region_key: str = "measurement_region",
) -> dict[str, Any]:
    """Compare interleaved artifact-off/on timings without hiding noise.

    The sorted sequence must alternate ``off`` and ``on`` and contain complete
    adjacent pairs. All declared comparison fields must match across the whole
    sequence, which is how shape and delivered-host/device provenance are kept
    comparable. Signed deltas are always ``on - off`` and are never clamped.
    Every input row must identify itself as part of the measured region; warmup
    rows cannot qualify as comparison evidence. Three pairs are required to
    meet the repetition/comparability floor. The result remains a signed delta
    measurement: whether its magnitude exceeds its dispersion is a reader's
    claim, not something inferred from pair count alone.
    """

    if not measurements:
        raise ValueError("artifact timing comparison requires measurements")
    ordered = sorted(measurements, key=lambda row: _as_float(row.get(order_key)))
    if len(ordered) % 2:
        raise ValueError("artifact timing comparison requires complete off/on pairs")
    orders = [_as_float(row.get(order_key)) for row in ordered]
    if not all(math.isfinite(value) for value in orders) or len(set(orders)) != len(orders):
        raise ValueError("artifact timing measurement indices must be finite and unique")
    modes = [str(row.get(mode_key, "")).strip().lower() for row in ordered]
    if any(mode not in {"off", "on"} for mode in modes):
        raise ValueError("artifact_mode must be 'off' or 'on'")
    if any(left == right for left, right in zip(modes, modes[1:])):
        raise ValueError("artifact timing measurements must alternate off and on")
    regions = [str(row.get(region_key, "")).strip().lower() for row in ordered]
    if any(region != "measured" for region in regions):
        raise ValueError(
            "artifact timing comparison accepts only rows explicitly marked "
            "measurement_region='measured'"
        )
    fields = tuple(str(field).strip() for field in comparison_fields)
    if not fields or any(not field for field in fields) or len(set(fields)) != len(fields):
        raise ValueError("comparison_fields must contain unique non-empty names")
    comparison: dict[str, Any] = {}
    for field in fields:
        first = ordered[0].get(field)
        if first is None or (isinstance(first, str) and not first.strip()):
            raise ValueError(
                f"artifact timing comparison field {field!r} must be present and non-empty"
            )
        if any(row.get(field) != first for row in ordered[1:]):
            raise ValueError(f"artifact timing comparison field {field!r} is not comparable")
        comparison[field] = first

    deltas: list[float] = []
    for first, second in zip(ordered[::2], ordered[1::2]):
        pair = {
            str(first.get(mode_key)).strip().lower(): first,
            str(second.get(mode_key)).strip().lower(): second,
        }
        off_time = _as_float(pair["off"].get(time_key))
        on_time = _as_float(pair["on"].get(time_key))
        if not math.isfinite(off_time) or not math.isfinite(on_time):
            raise ValueError("artifact timing values must be finite")
        deltas.append(on_time - off_time)

    n_pairs = len(deltas)
    repeated_comparable = n_pairs >= 3
    status = (
        "repeated_comparable_delta"
        if repeated_comparable
        else "single_observation_delta"
        if n_pairs == 1
        else "insufficient_repetition_delta"
    )
    return {
        "status": status,
        "n_pairs": n_pairs,
        "repeated_comparable": repeated_comparable,
        "measurement_region": "measured",
        "comparison_fields": comparison,
        "artifact_delta_sec_by_pair": tuple(deltas),
        "artifact_delta_sec_min": min(deltas),
        "artifact_delta_sec_median": _median(deltas),
        "artifact_delta_sec_max": max(deltas),
        "artifact_delta_sec_spread": max(deltas) - min(deltas),
    }


def cost_by_axis_rows(
    cost_rows: Sequence[Mapping[str, Any]],
    *,
    axis_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return per-(stage, axis, value) medians over ``cost_by_run`` rows.

    ``gpu_seconds`` is aggregated as supplied by :func:`cost_by_run_row`; it is
    never recomputed here from ``device_count`` and ``allocated_wall_time_sec``.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in cost_rows:
        stage = str(row.get("stage", ""))
        for axis in axis_names:
            if axis not in row:
                continue
            key = (stage, str(axis), str(row.get(axis, "")))
            grouped.setdefault(key, []).append(row)

    def column(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
        return [_as_float(row.get(name)) for row in rows]

    output = []
    for (stage, axis_name, axis_value), rows in sorted(grouped.items()):
        wall = column(rows, "wall_time_sec")
        gpu_seconds = column(rows, "gpu_seconds")
        output.append(
            {
                "stage": stage,
                "axis_name": axis_name,
                "axis_value": axis_value,
                "n_runs": str(len(rows)),
                "wall_time_sec_median": _format(_median(wall)),
                "wall_time_sec_q25": _format(_quantile(wall, 0.25)),
                "wall_time_sec_q75": _format(_quantile(wall, 0.75)),
                "step_time_sec_median": _format(_median(column(rows, "median_step_time_sec"))),
                "local_energy_time_sec_median": _format(_median(column(rows, "mean_local_energy_time_sec"))),
                "forward_time_sec_median": _format(_median(column(rows, "mean_forward_time_sec"))),
                "backward_time_sec_median": _format(_median(column(rows, "mean_backward_time_sec"))),
                "peak_memory_mb_median": _format(_median(column(rows, "peak_memory_mb"))),
                "gpu_seconds_median": _format(_median(gpu_seconds)),
                # A blank cell parses to NaN and is dropped, so a median over two
                # of nine runs would otherwise be indistinguishable from one over
                # all nine. ``n_runs`` is the group size; this is how many rows
                # actually carried a finite ``gpu_seconds``.
                "n_runs_with_gpu_seconds": str(len(_finite(gpu_seconds))),
            }
        )
    return output
