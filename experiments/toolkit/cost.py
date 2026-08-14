"""Compact cost-projection tables from run metrics.

Projects declared timing/resource metrics (``runtime/*``, ``train/perf/*``,
``diagnostics/*/time_sec``, ``eval/perf/<task>/*``) from per-run metrics rows
into the compact cost tables recommended by the experiment profiling spec:
``cost_by_run.csv``, ``cost_by_axis.csv``, and ``cost_by_task.csv``. Pure
projection: callers read metrics rows with :mod:`experiments.toolkit.artifacts`
readers and pass them in; nothing here scans run directories, and unknown or
absent metrics simply leave blank cells.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.toolkit.artifacts import metric_map

TRAIN_PHASES = (
    "sampling",
    "batch_build",
    "local_energy",
    "forward",
    "objective",
    "backward",
    "optimizer_step",
    "post_step_metrics",
)

COST_BY_RUN_BASE_COLUMNS = [
    "run_id",
    "attempt_id",
    "stage",
    "device_type",
    "status",
    "wall_time_sec",
    "peak_memory_mb",
    "timing_mode",
    "hostname",
    "slurm_job_id",
    "device_uuid",
    "n_steps",
    "mean_step_time_sec",
    "median_step_time_sec",
    "p95_step_time_sec",
    *(f"mean_{phase}_time_sec" for phase in TRAIN_PHASES),
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
    """

    metric_values = metric_map(metrics_rows)
    step_times = _per_step_values(metrics_rows, "train/perf", "step_time_sec")
    row: dict[str, Any] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "device_type": device_type,
        "status": status,
        "wall_time_sec": metric_values.get("runtime/wall_time_sec", ""),
        "peak_memory_mb": metric_values.get("runtime/peak_memory_mb", ""),
        "n_steps": str(len(step_times)) if step_times else "",
        "mean_step_time_sec": _format(_mean(step_times)),
        "median_step_time_sec": _format(_median(step_times)),
        "p95_step_time_sec": _format(_quantile(step_times, 0.95)),
        **{
            key: (provenance or {}).get(key, "")
            for key in ("timing_mode", "hostname", "slurm_job_id", "device_uuid")
        },
    }
    for phase in TRAIN_PHASES:
        values = _per_step_values(metrics_rows, "train/perf", f"{phase}_time_sec")
        row[f"mean_{phase}_time_sec"] = _format(_mean(values))
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
) -> list[dict[str, Any]]:
    """Return per-evaluation-task ``cost_by_task.csv`` rows from run metrics.

    Task wall time comes from ``diagnostics/<task>/time_sec``; component times
    come from ``eval/perf/<task>/{generator_time_sec, calculator/<name>_time_sec,
    summary/<name>_time_sec}`` with calculator/summary components summed per
    task. Tasks appear if either source is present.
    """

    metric_values = metric_map(metrics_rows)
    tasks: dict[str, dict[str, float]] = {}

    def task_entry(task_name: str) -> dict[str, float]:
        return tasks.setdefault(
            task_name,
            {"time_sec": math.nan, "generator": math.nan, "calculator": math.nan, "summary": math.nan},
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
                    entry["calculator"] = addend + (entry["calculator"] if math.isfinite(entry["calculator"]) else 0.0)
            elif component.startswith("summary/") and component.endswith("_time_sec"):
                addend = _as_float(value)
                if math.isfinite(addend):
                    entry["summary"] = addend + (entry["summary"] if math.isfinite(entry["summary"]) else 0.0)
    return [
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "stage": stage,
            "task_name": task_name,
            "device_type": device_type,
            "time_sec": _format(entry["time_sec"]),
            "generator_time_sec": _format(entry["generator"]),
            "calculator_time_sec": _format(entry["calculator"]),
            "summary_time_sec": _format(entry["summary"]),
        }
        for task_name, entry in sorted(tasks.items())
    ]


def cost_by_axis_rows(
    cost_rows: Sequence[Mapping[str, Any]],
    *,
    axis_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return per-(stage, axis, value) medians over ``cost_by_run`` rows."""

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
            }
        )
    return output
