"""Immutable, stdlib-only selector verification semantics for V1 references.

Never modify this module's behavior in place. Add a new version and register it
in selector_verifiers.py when selection semantics change.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def _key_text(value: Any) -> str:
    """Return a stable text representation for grouping and CSV output."""

    if value is None:
        return ""
    return str(value)


def _csv_number(value: float) -> str:
    """Return a compact CSV/JSON-safe numeric string."""

    if not math.isfinite(value):
        return ""
    return f"{value:.16g}"


def normalize_champion_specs(configured: Sequence[Any]) -> list[dict[str, Any]]:
    """Return normalized champion selector specs."""

    specs: list[dict[str, Any]] = []
    for entry in configured:
        if isinstance(entry, dict):
            spec = dict(entry)
        else:
            raise ValueError(f"champion entries must be mappings, got {entry!r}")
        name = str(spec.get("name", "")).strip()
        selector = str(spec.get("selector", "")).strip()
        if not name:
            raise ValueError("champion specs require a non-empty name")
        if not selector:
            raise ValueError(f"champion {name!r} requires selector")
        spec["name"] = name
        spec["selector"] = selector
        specs.append(spec)
    if not specs:
        raise ValueError("champion_specs must contain at least one selector")
    return specs


def _normalize_reference_metrics(configured: Sequence[Any] | None) -> list[tuple[str, str]]:
    """Return ``(label, source_metric)`` pairs copied into champions.csv."""

    if configured is None:
        return []
    metrics = []
    for entry in configured:
        if isinstance(entry, dict):
            label = str(entry.get("label", "")).strip()
            metric = str(entry.get("metric", "")).strip()
        else:
            try:
                label, metric = entry
            except (TypeError, ValueError) as exc:
                raise ValueError("reference metrics require label and metric") from exc
            label = str(label).strip()
            metric = str(metric).strip()
        if not label or not metric:
            raise ValueError("reference metrics require non-empty label and metric")
        metrics.append((label, metric))
    return metrics


def _as_float(value: Any, *, default: float = math.inf) -> float:
    """Return ``value`` as a finite float, or ``default``."""

    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _median(values: Sequence[float]) -> float:
    """Return the median of finite/non-finite numeric values."""

    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean for finite values."""

    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return math.inf
    return sum(finite) / len(finite)


def _stderr(values: Sequence[float]) -> float:
    """Return the sample standard error across seed-level values."""

    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return math.inf
    if len(finite) == 1:
        return 0.0
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance / len(finite))


def _seed_metric(metric: str, statistic: str) -> str:
    return f"{metric}_seed_{statistic}"


def reference_metrics(
    configured: Sequence[Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return stable champions.csv reference labels and source metrics."""

    return tuple(_normalize_reference_metrics(configured))


def reference_columns(
    reference_metrics: Sequence[tuple[str, str]],
    *,
    reference_statistics: Sequence[str],
) -> list[str]:
    """Return champions.csv columns for seed-aggregated reference metrics."""

    return [
        f"{label}_seed_{statistic}"
        for label, _metric in reference_metrics
        for statistic in reference_statistics
    ]


def _metric_value(row: dict[str, Any], metric: str, *, mode: str) -> float:
    """Return a sortable metric value, sending missing/non-finite to the worst end."""

    worst = math.inf if mode == "min" else -math.inf
    value = _as_float(row.get(metric), default=worst)
    return value if math.isfinite(value) else worst


def _task_has_metric(rows: Sequence[dict[str, Any]], source_metric: str) -> bool:
    """Return whether any row has a finite seed-median value for ``source_metric``."""

    metric = _seed_metric(source_metric, "median")
    return any(math.isfinite(_as_float(row.get(metric))) for row in rows)


def _clearly_beats(a: dict[str, Any], b: dict[str, Any], source_metric: str) -> bool:
    """Return whether row ``a`` beats row ``b`` by non-overlapping error bars."""

    mean_metric = _seed_metric(source_metric, "mean")
    stderr_metric = _seed_metric(source_metric, "stderr")
    a_mean = _as_float(a.get(mean_metric))
    b_mean = _as_float(b.get(mean_metric))
    if not math.isfinite(a_mean) or not math.isfinite(b_mean):
        return math.isfinite(a_mean) and not math.isfinite(b_mean)
    a_stderr = max(0.0, _as_float(a.get(stderr_metric), default=0.0))
    b_stderr = max(0.0, _as_float(b.get(stderr_metric), default=0.0))
    return a_mean + a_stderr < b_mean - b_stderr


def _wall_time(row: dict[str, Any], *, wall_time_metrics: Sequence[str]) -> float:
    for metric in wall_time_metrics:
        value = _as_float(row.get(_seed_metric(metric, "median")))
        if math.isfinite(value):
            return value
    return math.inf


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("config_id") or row.get("run_id", ""))


def group_key(row: dict[str, Any], group_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(_key_text(row.get(key)) for key in group_keys)


def group_label_from_key(group_keys: Sequence[str], key: Sequence[str]) -> str:
    return "|".join(f"{name}={value}" for name, value in zip(group_keys, key, strict=True))


def parse_group_by(group_by: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_by, str):
        keys = tuple(part.strip() for part in group_by.split(",") if part.strip())
    else:
        keys = tuple(str(part).strip() for part in group_by if str(part).strip())
    if not keys:
        raise ValueError("group_by must contain at least one column")
    return keys


def _select_by_metric_ladder(
    rows: Sequence[dict[str, Any]],
    *,
    tasks: Sequence[str],
    metric_template: str,
    mode: str = "min",
    fallback_metric: str,
    fallback_mode: str = "min",
) -> tuple[dict[str, Any], list[str], str, str]:
    """Select an aggregated config by an ordered metric ladder."""

    if mode != "min":
        raise ValueError("metric_ladder currently supports mode='min' only")
    if not metric_template:
        raise ValueError("metric_ladder requires metric_template")
    remaining = list(rows)
    decisions: list[str] = []
    selected_metric = ""
    selected_value = ""

    for task in tasks:
        source_metric = metric_template.format(task=task)
        if not _task_has_metric(remaining, source_metric):
            decisions.append(f"{task}: skipped, no finite metric {source_metric!r} in the current cohort")
            continue
        metric = _seed_metric(source_metric, "median")
        finite_rows = [row for row in remaining if math.isfinite(_as_float(row.get(metric)))]
        if not finite_rows:
            decisions.append(f"{task}: skipped, no finite metric {source_metric!r} in the current cohort")
            continue
        leader = min(finite_rows, key=lambda row: (_as_float(row.get(metric)), _row_label(row)))
        next_remaining = [
            row for row in finite_rows if row is leader or not _clearly_beats(leader, row, source_metric)
        ]
        selected_metric = metric
        selected_value = str(leader.get(metric, ""))
        if len(next_remaining) == 1:
            decisions.append(f"{task}: {_row_label(leader)} clearly wins by non-overlapping seed error bars")
            return leader, decisions, selected_metric, selected_value
        decisions.append(
            f"{task}: {len(next_remaining)} configs remain because their seed error bars overlap the leader"
        )
        remaining = next_remaining

    selected_metric = fallback_metric if fallback_metric.endswith("_seed_median") else _seed_metric(fallback_metric, "median")
    if fallback_mode not in {"min", "max"}:
        raise ValueError(f"fallback_mode must be 'min' or 'max', got {fallback_mode!r}")
    leader = min(
        remaining,
        key=lambda row: (
            _metric_value(row, selected_metric, mode=fallback_mode)
            if fallback_mode == "min"
            else -_metric_value(row, selected_metric, mode=fallback_mode),
            _row_label(row),
        ),
    )
    value = _metric_value(leader, selected_metric, mode=fallback_mode)
    selected_value = "" if not math.isfinite(value) else str(value)
    if len(remaining) == 1:
        decisions.append("all metric-ladder tie-breakers reduced the cohort to one row")
    else:
        decisions.append(f"metric ladder exhausted; selected by fallback {selected_metric!r} ({fallback_mode})")
    return leader, decisions, selected_metric, selected_value


def _select_by_single_metric(
    rows: Sequence[dict[str, Any]], *, metric: str, mode: str
) -> tuple[dict[str, Any], list[str], str, str]:
    """Select an aggregated config by one scalar metric for CLI overrides."""

    selected_metric = metric if metric.endswith("_seed_median") else _seed_metric(metric, "median")

    def sort_key(row: dict[str, Any]) -> tuple[float, str]:
        value = _metric_value(row, selected_metric, mode=mode)
        return (value if mode == "min" else -value, _row_label(row))

    best = min(rows, key=sort_key)
    return (
        best,
        [f"selected by explicit seed-aggregated scalar metric {selected_metric!r} ({mode})"],
        selected_metric,
        str(best.get(selected_metric, "")),
    )


def _select_metric_champion(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    excluded_config_id: str | None = None,
) -> tuple[dict[str, Any] | None, list[str], str, str]:
    """Select a champion by a scalar seed-aggregated metric."""

    selected_metric = metric if metric.endswith("_seed_median") else _seed_metric(metric, "median")
    finite_rows = [
        row for row in rows if math.isfinite(_metric_value(row, selected_metric, mode=mode))
    ]
    finite_rows.sort(
        key=lambda row: (
            _metric_value(row, selected_metric, mode=mode)
            if mode == "min"
            else -_metric_value(row, selected_metric, mode=mode),
            _row_label(row),
        )
    )
    if not finite_rows:
        return None, [f"no finite metric {selected_metric!r} found"], selected_metric, ""

    best = finite_rows[0]
    if excluded_config_id is not None and _row_label(best) == excluded_config_id:
        alternatives = [row for row in finite_rows if _row_label(row) != excluded_config_id]
        if not alternatives:
            return (
                None,
                ["best metric config is excluded; no distinct alternative is available"],
                selected_metric,
                "",
            )
        best = alternatives[0]
        decisions = ["best metric config is excluded; selected the next best distinct config"]
    else:
        decisions = [f"selected by scalar metric {selected_metric!r} ({mode})"]
    return best, decisions, selected_metric, str(best.get(selected_metric, ""))


def champion_record(
    row: dict[str, Any] | None,
    *,
    group_keys: Sequence[str],
    group_key: Sequence[str],
    config_keys: Sequence[str],
    winner_kind: str,
    metric: str,
    metric_value: str,
    reference_metrics: Sequence[tuple[str, str]],
    reference_statistics: Sequence[str],
) -> dict[str, Any]:
    """Return one row for ``champions.csv``."""

    record = {key: value for key, value in zip(group_keys, group_key, strict=True)}
    record["winner_kind"] = winner_kind
    record["metric"] = metric
    record["metric_value"] = metric_value
    record["metric_seed_mean"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_mean"), ""))
    record["metric_seed_stderr"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_stderr"), ""))
    record["metric_seed_n"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_n"), ""))
    for label, source_metric in reference_metrics:
        for statistic in reference_statistics:
            column = f"{label}_seed_{statistic}"
            record[column] = "" if row is None else str(row.get(_seed_metric(source_metric, statistic), ""))
    if row is None:
        for key in (
            "config_id",
            "major_id",
            "minor_id",
            "run_ids",
            "seeds",
            "n_expected",
            "n_present",
            "n_success",
            "n_failed",
            "n_missing_seed",
            *(key for key in config_keys if key not in group_keys),
        ):
            record[key] = ""
        return record
    for key in config_keys:
        record[key] = row.get(key, "")
    record.update(
        config_id=row.get("config_id", ""),
        major_id=row.get("major_id", ""),
        minor_id=row.get("minor_id", ""),
        run_ids=row.get("run_ids", ""),
        seeds=row.get("seeds", ""),
        n_expected=row.get("n_expected", ""),
        n_present=row.get("n_present", ""),
        n_success=row.get("n_success", ""),
        n_failed=row.get("n_failed", ""),
        n_missing_seed=row.get("n_missing_seed", ""),
    )
    return record


def select_by_spec(
    rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    *,
    selected_by_name: dict[str, dict[str, Any]],
    default_fallback_metric: str,
    metric_override: str | None = None,
    mode_override: str = "min",
) -> tuple[dict[str, Any] | None, list[str], str, str]:
    """Select one champion according to a normalized selector spec."""

    selector = str(spec.get("selector", ""))
    if selector in {"metric_ladder", "energy_ladder"}:
        if metric_override is not None:
            return _select_by_single_metric(rows, metric=metric_override, mode=mode_override)
        tasks = [str(task) for task in spec.get("tasks", [])]
        if not tasks:
            raise ValueError(f"champion {spec['name']!r} metric_ladder requires tasks")
        return _select_by_metric_ladder(
            rows,
            tasks=tasks,
            metric_template=str(spec.get("metric_template", "")),
            mode=str(spec.get("mode", "min")),
            fallback_metric=str(spec.get("fallback_metric", default_fallback_metric)),
            fallback_mode=str(spec.get("fallback_mode", "min")),
        )
    if selector in {"metric", "scalar_metric"}:
        metric = str(spec.get("metric", "")).strip()
        if not metric:
            raise ValueError(f"champion {spec['name']!r} metric selector requires metric")
        excluded_name = str(spec.get("exclude", "")).strip()
        excluded_config_id = None
        if excluded_name and excluded_name in selected_by_name:
            excluded_config_id = _row_label(selected_by_name[excluded_name])
        return _select_metric_champion(
            rows,
            metric=metric,
            mode=str(spec.get("mode", "min")),
            excluded_config_id=excluded_config_id,
        )
    raise ValueError(f"unsupported champion selector {selector!r} for {spec.get('name', '<unnamed>')!r}")


# ---------------------------------------------------------------------------
# Seed -> trial aggregation (COLLECTOR-owned, co-located here temporarily).
#
# In the incoming experiment-stack design these helpers belong to the metric
# collector (future ``MetricCollector.collect_trial_results``), not the
# selector: they fold seed rows into one row per configuration and are NOT part
# of the selector concept. They live here only until that collector contract
# exists.
# ---------------------------------------------------------------------------


def _numeric_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    config_keys: Sequence[str],
    seed_key: str,
) -> list[str]:
    metrics = []
    non_metrics = {
        *config_keys,
        "major_id",
        "minor_id",
        "config_id",
        seed_key,
        "status",
        "run_id",
        "validation_attempt_id",
        "validation_attempt_dir",
        "train_attempt_id",
        "checkpoint_path",
    }
    for key in sorted({key for row in rows for key in row}):
        if key in non_metrics:
            continue
        if any(math.isfinite(_as_float(row.get(key))) for row in rows):
            metrics.append(key)
    return metrics


def _aggregate_metric(
    row: dict[str, Any],
    metric: str,
    *,
    median_values: Sequence[float],
    moment_values: Sequence[float],
) -> None:
    """Write seed aggregate statistics for one metric into ``row``."""

    finite = [value for value in moment_values if math.isfinite(value)]
    row[_seed_metric(metric, "median")] = _csv_number(_median(median_values))
    row[_seed_metric(metric, "mean")] = _csv_number(_mean(finite))
    row[_seed_metric(metric, "stderr")] = _csv_number(_stderr(finite))
    row[_seed_metric(metric, "n")] = str(len(finite))


def aggregate_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    config_keys: Sequence[str],
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    seed_key: str,
    axis_id_labels: dict[str, str],
    success_statuses: Any,
    id_for_axes: Callable[[Mapping[str, Any], Sequence[str], Mapping[str, str]], str],
) -> tuple[list[dict[str, Any]], bool]:
    """Aggregate seed rows into one row per non-seed configuration."""

    successes = [row for row in rows if str(row.get("status", "")) in success_statuses]
    used_status_fallback = not successes
    value_rows = list(rows) if used_status_fallback else successes
    metrics = _numeric_metrics(value_rows, config_keys=config_keys, seed_key=seed_key)
    expected_seeds = sorted({_key_text(row.get(seed_key)) for row in rows if _key_text(row.get(seed_key))})

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row, config_keys), []).append(row)

    candidates = []
    for key, group_rows in grouped.items():
        first = group_rows[0]
        seed_rows = {_key_text(row.get(seed_key)): row for row in group_rows if _key_text(row.get(seed_key))}
        seed_order = expected_seeds or sorted(seed_rows)
        run_ids = sorted({_key_text(row.get("run_id")) for row in group_rows if _key_text(row.get("run_id"))})
        n_success = sum(1 for row in seed_rows.values() if str(row.get("status", "")) in success_statuses)
        n_expected = len(seed_order)
        n_missing_seed = sum(1 for seed in seed_order if seed not in seed_rows)
        point = {config_key: first.get(config_key, key[index]) for index, config_key in enumerate(config_keys)}
        candidate: dict[str, Any] = {
            "config_id": first.get("config_id") or id_for_axes(point, config_keys, axis_id_labels),
            "run_id": first.get("config_id") or id_for_axes(point, config_keys, axis_id_labels),
            "major_id": first.get("major_id") or id_for_axes(point, major_axes, axis_id_labels),
            "minor_id": first.get("minor_id") or id_for_axes(point, minor_axes, axis_id_labels),
            "run_ids": ";".join(run_ids),
            "seeds": ",".join(seed_order),
            seed_key: ",".join(seed_order),
            "n_expected": n_expected,
            "n_present": len(seed_rows),
            "n_success": n_success,
            "n_failed": n_expected - n_success,
            "n_missing_seed": n_missing_seed,
        }
        for index, config_key in enumerate(config_keys):
            candidate[config_key] = key[index]
        for metric in metrics:
            median_values: list[float] = []
            moment_values: list[float] = []
            for seed in seed_order:
                source_row = seed_rows.get(seed)
                if source_row is None:
                    median_values.append(math.inf)
                    continue
                is_success = str(source_row.get("status", "")) in success_statuses
                value = _as_float(source_row.get(metric))
                if used_status_fallback or is_success:
                    median_values.append(value)
                    if math.isfinite(value):
                        moment_values.append(value)
                else:
                    median_values.append(math.inf)
            _aggregate_metric(
                candidate,
                metric,
                median_values=median_values,
                moment_values=moment_values,
            )
        candidates.append(candidate)
    return sorted(candidates, key=lambda row: _row_label(row)), used_status_fallback


VERIFIER_ID = "pair-stability-v4/selector-verifier/v1"
CONTRACT_SCHEMA_VERSION = "pair-stability-v4/selection-verification/v1"
SOURCE_PATH = "experiments/hooke/pair_stability_v4/selector_verifier_v1.py"
TRAIN_WALL_TIME_METRIC = "train/runtime/wall_time_sec"
SELECTION_REFERENCE_STATISTICS = ("median", "mean", "stderr")
SELECTION_SUCCESS_STATUSES = {"completed", "success"}
SELECTION_DEFAULT_FALLBACK_METRIC = TRAIN_WALL_TIME_METRIC


def replay_selection(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the immutable V1 aggregation and champion policy."""

    major_axes = tuple(str(value) for value in grid.get("major_axes", ()))
    minor_axes = tuple(str(value) for value in grid.get("minor_axes", ()))
    config_keys = (*major_axes, *minor_axes)
    seed_key = str(grid.get("scan_seed_axis") or "")
    labels_value = grid.get("axis_id_labels")
    if not isinstance(labels_value, Mapping):
        raise ValueError("source grid axis labels are missing")
    labels = {
        str(key): str(value) for key, value in labels_value.items()
    }
    specs = normalize_champion_specs(grid.get("champions"))
    references = reference_metrics(grid.get("champion_reference_metrics"))

    def missing_id(
        _point: Mapping[str, Any],
        axes: Sequence[str],
        _labels: Mapping[str, str],
    ) -> str:
        raise ValueError(
            f"selector replay requires persisted ids for axes {list(axes)!r}"
        )

    ordered_rows = sorted(
        (dict(row) for row in summary_rows),
        key=lambda row: (
            str(row.get("config_id") or ""),
            str(row.get(seed_key) or ""),
            str(row.get("run_id") or ""),
        ),
    )
    candidates, used_fallback = aggregate_candidates(
        ordered_rows,
        config_keys=config_keys,
        major_axes=major_axes,
        minor_axes=minor_axes,
        seed_key=seed_key,
        axis_id_labels=labels,
        success_statuses=SELECTION_SUCCESS_STATUSES,
        id_for_axes=missing_id,
    )
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = group_key(candidate, major_axes)
        groups.setdefault(key, []).append(candidate)

    champions: list[dict[str, Any]] = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    for grouped_key, group_rows in sorted(groups.items()):
        group_decisions: dict[str, list[str]] = {}
        selected_by_name: dict[str, dict[str, Any]] = {}
        for spec in specs:
            kind = str(spec["name"])
            winner, decisions, metric, metric_value = select_by_spec(
                group_rows,
                spec,
                selected_by_name=selected_by_name,
                default_fallback_metric=SELECTION_DEFAULT_FALLBACK_METRIC,
                metric_override=None,
                mode_override="min",
            )
            group_decisions[kind] = decisions
            if winner is not None:
                selected_by_name[kind] = winner
            champions.append(
                champion_record(
                    winner,
                    group_keys=major_axes,
                    group_key=grouped_key,
                    config_keys=config_keys,
                    winner_kind=kind,
                    metric=metric,
                    metric_value=metric_value,
                    reference_metrics=references,
                    reference_statistics=SELECTION_REFERENCE_STATISTICS,
                )
            )
        decisions_by_group[
            group_label_from_key(major_axes, grouped_key)
        ] = group_decisions

    if candidates:
        overall, overall_decisions, overall_metric, overall_value = (
            select_by_spec(
                candidates,
                specs[0],
                selected_by_name={},
                default_fallback_metric=SELECTION_DEFAULT_FALLBACK_METRIC,
                metric_override=None,
                mode_override="min",
            )
        )
    else:
        overall, overall_decisions, overall_metric, overall_value = (
            None,
            [],
            "",
            "",
        )
    overall_selected = (
        {str(specs[0]["name"]): overall} if overall is not None else {}
    )
    secondary_spec = specs[1] if len(specs) > 1 else specs[0]
    if candidates:
        secondary, secondary_decisions, secondary_metric, secondary_value = (
            select_by_spec(
                candidates,
                secondary_spec,
                selected_by_name=overall_selected,
                default_fallback_metric=SELECTION_DEFAULT_FALLBACK_METRIC,
                metric_override=None,
                mode_override="min",
            )
        )
    else:
        secondary, secondary_decisions, secondary_metric, secondary_value = (
            None,
            [],
            "",
            "",
        )
    return {
        "champions": champions,
        "configs": candidates,
        "overall_champion": (
            None if overall is None else overall.get("config_id", "")
        ),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_value,
        "overall_decisions": overall_decisions,
        "secondary_champion_kind": str(secondary_spec["name"]),
        "secondary_champion": (
            None if secondary is None else secondary.get("config_id", "")
        ),
        "secondary_metric": secondary_metric,
        "secondary_metric_value": secondary_value,
        "secondary_decisions": secondary_decisions,
        "decisions_by_group": decisions_by_group,
        "used_status_fallback": used_fallback,
        "group_by": list(major_axes),
        "champion_kinds": [str(spec["name"]) for spec in specs],
        "champion_specs": specs,
        "config_keys": list(config_keys),
        "reference_metrics": [
            {"label": label, "metric": metric}
            for label, metric in references
        ],
        "n_candidates": len(candidates),
    }


def champion_columns(replay: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact V1 champions.csv header."""

    group_keys = tuple(str(value) for value in replay["group_by"])
    config_keys = tuple(str(value) for value in replay["config_keys"])
    references = reference_metrics(replay["reference_metrics"])
    return (
        *group_keys,
        "winner_kind",
        "config_id",
        "major_id",
        "minor_id",
        *(key for key in config_keys if key not in group_keys),
        "seeds",
        "n_expected",
        "n_present",
        "n_success",
        "n_failed",
        "n_missing_seed",
        "metric",
        "metric_value",
        "metric_seed_mean",
        "metric_seed_stderr",
        "metric_seed_n",
        *reference_columns(
            references,
            reference_statistics=SELECTION_REFERENCE_STATISTICS,
        ),
        "run_ids",
    )


def report_projection(
    *,
    grid: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Return every selection-report field governed by V1 semantics."""

    return {
        "metric": None,
        "mode": "min",
        "group_by": replay["group_by"],
        "major_axes": list(grid.get("major_axes") or ()),
        "minor_axes": list(grid.get("minor_axes") or ()),
        "scan_seed_axis": grid.get("scan_seed_axis"),
        "axis_id_labels": grid.get("axis_id_labels"),
        "champion_kinds": replay["champion_kinds"],
        "champion_specs": replay["champion_specs"],
        "config_keys": replay["config_keys"],
        "seed_aggregation": {
            "value": "median of successful seed rows",
            "error_bar": (
                "sample standard error across successful seed rows"
            ),
            "mean": "arithmetic mean across successful seed rows",
        },
        "reference_metrics": replay["reference_metrics"],
        "reference_statistics": list(SELECTION_REFERENCE_STATISTICS),
        "wall_time_metrics": [TRAIN_WALL_TIME_METRIC],
        "n_candidates": replay["n_candidates"],
        "n_configs": replay["n_candidates"],
        "n_champions": len(replay["champions"]),
        "overall_champion": replay["overall_champion"],
        "overall_metric": replay["overall_metric"],
        "overall_metric_value": replay["overall_metric_value"],
        "overall_decisions": replay["overall_decisions"],
        "secondary_champion_kind": replay["secondary_champion_kind"],
        "secondary_metric": replay["secondary_metric"],
        "secondary_champion": replay["secondary_champion"],
        "secondary_metric_value": replay["secondary_metric_value"],
        "secondary_decisions": replay["secondary_decisions"],
        "decisions_by_group": replay["decisions_by_group"],
        "used_status_fallback": replay["used_status_fallback"],
        "champions": replay["champions"],
        "configs": replay["configs"],
    }


def csv_scalar(value: object) -> str:
    """Return the stable CSV text representation used by the V1 producer."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, sort_keys=True)


def policy_projection(
    *,
    grid: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Return normalized immutable policy inputs."""

    return {
        "champion_specs": replay["champion_specs"],
        "champion_reference_metrics": replay["reference_metrics"],
        "config_keys": replay["config_keys"],
        "group_by": replay["group_by"],
        "seed_key": str(grid.get("scan_seed_axis") or ""),
        "axis_id_labels": grid.get("axis_id_labels"),
        "success_statuses": sorted(SELECTION_SUCCESS_STATUSES),
        "default_fallback_metric": SELECTION_DEFAULT_FALLBACK_METRIC,
        "reference_statistics": list(SELECTION_REFERENCE_STATISTICS),
        "metric_override": None,
        "mode": "min",
    }


def input_projection(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return normalized selector inputs, including every consumed summary cell."""

    grid_fields = {
        key: grid.get(key)
        for key in (
            "major_axes",
            "minor_axes",
            "scan_seed_axis",
            "axis_id_labels",
            "champions",
            "champion_reference_metrics",
        )
    }
    seed_key = str(grid.get("scan_seed_axis") or "")
    rows = sorted(
        ({str(key): value for key, value in row.items()} for row in summary_rows),
        key=lambda row: (
            str(row.get("config_id") or ""),
            str(row.get(seed_key) or ""),
            str(row.get("run_id") or ""),
        ),
    )
    return {"grid": grid_fields, "summary_rows": rows}


def source_sha256() -> str:
    """Return the raw SHA of this immutable verifier module."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    """Return SHA-256 over canonical JSON."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_ARTIFACT_DIGEST_KEYS = {
    "grid_manifest",
    "summary_csv",
    "selection_report",
    "champions_csv",
}
_CONTRACT_FIELDS = {
    "schema_version",
    "verifier",
    "producer",
    "artifact_sha256",
    "input_projection",
    "input_projection_sha256",
    "policy",
    "policy_sha256",
    "replay",
    "replay_sha256",
    "report_projection",
    "report_projection_sha256",
    "champions_csv_projection",
    "champions_csv_projection_sha256",
    "contract_sha256",
}


def build_contract(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    producer_replay: Mapping[str, Any],
    producer_source_path: str,
    producer_source_sha256: str,
) -> dict[str, Any]:
    """Build a V1 contract after independent producer/verifier agreement."""

    replay = replay_selection(grid=grid, summary_rows=summary_rows)
    if canonical_sha256(replay) != canonical_sha256(producer_replay):
        raise ValueError(
            "current producer replay differs from immutable V1 verifier"
        )
    report = report_projection(grid=grid, replay=replay)
    for key, expected in report.items():
        if selection_report.get(key) != expected:
            raise ValueError(
                f"selection report differs from V1 projection for {key}"
            )
    columns = champion_columns(replay)
    observed_columns = tuple(champion_rows[0]) if champion_rows else ()
    if observed_columns != columns:
        raise ValueError("champions.csv header differs from V1 projection")
    if len(champion_rows) != len(replay["champions"]):
        raise ValueError("champions.csv population differs from V1 projection")
    expected_rows: list[dict[str, str]] = []
    for index, (actual, expected) in enumerate(
        zip(champion_rows, replay["champions"], strict=True)
    ):
        projected = {
            column: csv_scalar(expected.get(column, ""))
            for column in columns
        }
        if dict(actual) != projected:
            raise ValueError(
                f"champions.csv row {index} differs from V1 projection"
            )
        expected_rows.append(projected)
    if set(artifact_sha256) != _ARTIFACT_DIGEST_KEYS:
        raise ValueError("selection artifact digest fields mismatch")
    normalized_artifacts = {
        key: _require_digest(value)
        for key, value in sorted(artifact_sha256.items())
    }
    if producer_source_path != "experiments/toolkit/selection.py":
        raise ValueError("selection producer source path is invalid")
    producer_digest = _require_digest(producer_source_sha256)
    inputs = input_projection(grid=grid, summary_rows=summary_rows)
    policy = policy_projection(grid=grid, replay=replay)
    champions = {
        "columns": list(columns),
        "rows": expected_rows,
    }
    body = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "verifier": {
            "id": VERIFIER_ID,
            "source_path": SOURCE_PATH,
            "source_sha256": source_sha256(),
        },
        "producer": {
            "source_path": producer_source_path,
            "source_sha256": producer_digest,
        },
        "artifact_sha256": normalized_artifacts,
        "input_projection": inputs,
        "input_projection_sha256": canonical_sha256(inputs),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "replay": replay,
        "replay_sha256": canonical_sha256(replay),
        "report_projection": report,
        "report_projection_sha256": canonical_sha256(report),
        "champions_csv_projection": champions,
        "champions_csv_projection_sha256": canonical_sha256(champions),
    }
    return {
        **body,
        "contract_sha256": canonical_sha256(body),
    }


def verify_contract(
    contract: object,
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> tuple[str, ...]:
    """Verify a V1 contract without importing the mutable producer."""

    if not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS:
        return ("selection V1 contract fields mismatch",)
    verifier = contract.get("verifier")
    if not isinstance(verifier, dict) or verifier != {
        "id": VERIFIER_ID,
        "source_path": SOURCE_PATH,
        "source_sha256": source_sha256(),
    }:
        return ("selection V1 verifier identity/source digest mismatch",)
    producer = contract.get("producer")
    if not isinstance(producer, dict) or set(producer) != {
        "source_path",
        "source_sha256",
    }:
        return ("selection V1 producer provenance fields mismatch",)
    try:
        expected = build_contract(
            grid=grid,
            summary_rows=summary_rows,
            selection_report=selection_report,
            champion_rows=champion_rows,
            artifact_sha256=artifact_sha256,
            # Verification is intentionally independent of the producer.
            producer_replay=replay_selection(
                grid=grid,
                summary_rows=summary_rows,
            ),
            producer_source_path=str(producer.get("source_path") or ""),
            producer_source_sha256=str(
                producer.get("source_sha256") or ""
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return (f"cannot rebuild selection V1 contract: {exc}",)
    if contract != expected:
        return ("selection V1 contract/projection digest mismatch",)
    return ()


def _require_digest(value: object) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("invalid SHA-256 digest")
    return text
