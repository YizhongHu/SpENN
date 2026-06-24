"""Select champions from a collection attempt.

Reads a ``03_collect`` summary table, aggregates seed rows into non-seed
configs, and selects configured winner kinds per configured major-grid bucket.
Local energy ranking uses seed medians, while overlap tests use the
seed-combined mean and standard error. An explicit scalar metric can still be
passed for debugging overrides.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

from run_utils import (
    STAGE_COLLECT,
    STAGE_SELECT,
    axis_id_labels_from_manifest,
    grid_axes_from_manifest,
    id_for_axes,
    latest_attempt_id,
    load_study_module,
    log_prefix,
    new_attempt_id,
    read_json,
    smoke_attempt_id,
    stage_dir,
    study_name_from_manifest,
    write_json,
    write_latest,
)

_ancestry = load_study_module("ancestry", __file__)
source_grid_from_attempt = _ancestry.source_grid_from_attempt

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
REFERENCE_STATISTICS = ("median", "mean", "stderr")
WALL_TIME_METRICS = ("train/runtime/wall_time_sec",)
SUCCESS_STATUSES = {"completed", "success"}


def read_summary(collection_attempt_dir: Path) -> list[dict[str, Any]]:
    """Read the collection ``summary.csv`` rows."""

    summary = collection_attempt_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"collection attempt has no summary.csv: {summary}")
    with summary.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def _normalize_champion_specs(configured: Sequence[Any]) -> list[dict[str, Any]]:
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


def _reference_metrics(
    configured: Sequence[Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return stable champions.csv reference labels and source metrics."""

    return tuple(_normalize_reference_metrics(configured))


def _reference_columns(reference_metrics: Sequence[tuple[str, str]]) -> list[str]:
    """Return champions.csv columns for seed-aggregated reference metrics."""

    return [
        f"{label}_seed_{statistic}"
        for label, _metric in reference_metrics
        for statistic in REFERENCE_STATISTICS
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


def _wall_time(row: dict[str, Any]) -> float:
    for metric in WALL_TIME_METRICS:
        value = _as_float(row.get(_seed_metric(metric, "median")))
        if math.isfinite(value):
            return value
    return math.inf


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("config_id") or row.get("run_id", ""))


def _group_key(row: dict[str, Any], group_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(_key_text(row.get(key)) for key in group_keys)


def _group_label_from_key(group_keys: Sequence[str], key: Sequence[str]) -> str:
    return "|".join(f"{name}={value}" for name, value in zip(group_keys, key, strict=True))


def _parse_group_by(group_by: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_by, str):
        keys = tuple(part.strip() for part in group_by.split(",") if part.strip())
    else:
        keys = tuple(str(part).strip() for part in group_by if str(part).strip())
    if not keys:
        raise ValueError("group_by must contain at least one column")
    return keys


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


def _aggregate_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    config_keys: Sequence[str],
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    seed_key: str,
    axis_id_labels: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    """Aggregate seed rows into one row per non-seed configuration."""

    successes = [row for row in rows if str(row.get("status", "")) in SUCCESS_STATUSES]
    used_status_fallback = not successes
    value_rows = list(rows) if used_status_fallback else successes
    metrics = _numeric_metrics(value_rows, config_keys=config_keys, seed_key=seed_key)
    expected_seeds = sorted({_key_text(row.get(seed_key)) for row in rows if _key_text(row.get(seed_key))})

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row, config_keys), []).append(row)

    candidates = []
    for key, group_rows in grouped.items():
        first = group_rows[0]
        seed_rows = {_key_text(row.get(seed_key)): row for row in group_rows if _key_text(row.get(seed_key))}
        seed_order = expected_seeds or sorted(seed_rows)
        run_ids = sorted({_key_text(row.get("run_id")) for row in group_rows if _key_text(row.get("run_id"))})
        n_success = sum(1 for row in seed_rows.values() if str(row.get("status", "")) in SUCCESS_STATUSES)
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
                is_success = str(source_row.get("status", "")) in SUCCESS_STATUSES
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


def _select_by_metric_ladder(
    rows: Sequence[dict[str, Any]],
    *,
    tasks: Sequence[str],
    metric_template: str,
    mode: str = "min",
    fallback_metric: str = "train/runtime/wall_time_sec",
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


def _champion_record(
    row: dict[str, Any] | None,
    *,
    group_keys: Sequence[str],
    group_key: Sequence[str],
    config_keys: Sequence[str],
    winner_kind: str,
    metric: str,
    metric_value: str,
    reference_metrics: Sequence[tuple[str, str]],
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
        for statistic in REFERENCE_STATISTICS:
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


def _select_by_spec(
    rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    *,
    selected_by_name: dict[str, dict[str, Any]],
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
            fallback_metric=str(spec.get("fallback_metric", "train/runtime/wall_time_sec")),
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


def select_champions(
    rows: Sequence[dict[str, Any]],
    *,
    config_keys: Sequence[str],
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    seed_key: str,
    axis_id_labels: dict[str, str],
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] | None = None,
    champion_specs: Sequence[Any] | None = None,
    reference_metrics: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Select configured winner kinds per major grid point."""

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    group_keys = _parse_group_by(group_by if group_by is not None else major_axes)
    if champion_specs is None:
        raise ValueError("champion selector specs are required")
    champion_specs = _normalize_champion_specs(champion_specs)
    champion_kinds = [str(spec["name"]) for spec in champion_specs]
    reference_metric_pairs = _reference_metrics(reference_metrics)
    candidates, used_fallback = _aggregate_candidates(
        rows,
        config_keys=config_keys,
        major_axes=major_axes,
        minor_axes=minor_axes,
        seed_key=seed_key,
        axis_id_labels=axis_id_labels,
    )

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(_group_key(row, group_keys), []).append(row)

    champions = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    for group_key, group_rows in sorted(groups.items()):
        group_decisions: dict[str, list[str]] = {}
        selected_by_name: dict[str, dict[str, Any]] = {}
        for spec in champion_specs:
            kind = str(spec["name"])
            winner, decisions, selected_metric, selected_value = _select_by_spec(
                group_rows,
                spec,
                selected_by_name=selected_by_name,
                metric_override=metric,
                mode_override=mode,
            )
            group_decisions[kind] = decisions
            if winner is not None:
                selected_by_name[kind] = winner
            champions.append(
                _champion_record(
                    winner,
                    group_keys=group_keys,
                    group_key=group_key,
                    config_keys=config_keys,
                    winner_kind=kind,
                    metric=selected_metric,
                    metric_value=selected_value,
                    reference_metrics=reference_metric_pairs,
                )
            )
        decisions_by_group[_group_label_from_key(group_keys, group_key)] = group_decisions

    if candidates:
        overall, overall_decisions, overall_metric, overall_metric_value = _select_by_spec(
            candidates,
            champion_specs[0],
            selected_by_name={},
            metric_override=metric,
            mode_override=mode,
        )
    else:
        overall = None
        overall_decisions = []
        overall_metric = ""
        overall_metric_value = ""

    overall_selected: dict[str, dict[str, Any]] = {}
    if overall is not None:
        overall_selected[str(champion_specs[0]["name"])] = overall
    secondary_spec = champion_specs[1] if len(champion_specs) > 1 else champion_specs[0]
    secondary_name = str(secondary_spec["name"])
    if candidates:
        secondary, secondary_decisions, secondary_metric, secondary_metric_value = _select_by_spec(
            candidates,
            secondary_spec,
            selected_by_name=overall_selected,
            metric_override=None,
            mode_override=mode,
        )
    else:
        secondary = None
        secondary_decisions = []
        secondary_metric = ""
        secondary_metric_value = ""
    return {
        "champions": champions,
        "configs": candidates,
        "overall_champion": None if overall is None else overall.get("config_id", ""),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_metric_value,
        "overall_decisions": overall_decisions,
        "secondary_champion_kind": secondary_name,
        "secondary_champion": None if secondary is None else secondary.get("config_id", ""),
        "secondary_metric": secondary_metric,
        "secondary_metric_value": secondary_metric_value,
        "secondary_decisions": secondary_decisions,
        "decisions_by_group": decisions_by_group,
        "used_status_fallback": used_fallback,
        "group_by": list(group_keys),
        "champion_kinds": list(champion_kinds),
        "champion_specs": champion_specs,
        "config_keys": list(config_keys),
        "reference_metrics": [
            {"label": label, "metric": source_metric}
            for label, source_metric in reference_metric_pairs
        ],
        "n_candidates": len(candidates),
    }


def _resolve_collection_attempt(results_root: Path, collection_attempt_id: str | None, *, smoke: bool) -> str:
    if collection_attempt_id is not None:
        return collection_attempt_id
    collect_dir = stage_dir(results_root, STAGE_COLLECT)
    attempt_id = latest_attempt_id(collect_dir, smoke=smoke)
    if attempt_id is None:
        raise FileNotFoundError(f"no collection attempts under {collect_dir}")
    return attempt_id


def _champion_specs_from_grid(
    results_root: Path,
    collection_dir: Path,
    requested: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return champion specs from source grid manifest, optionally filtered by CLI."""

    manifest = _source_grid_manifest(results_root, collection_dir)
    configured = manifest.get("champions") if isinstance(manifest, dict) else None
    if not configured:
        raise ValueError("source grid manifest must define explicit champion selector specs")
    specs = _normalize_champion_specs(configured)
    if requested is None:
        return specs
    requested_names = {str(kind) for kind in requested}
    filtered = [spec for spec in specs if str(spec.get("name")) in requested_names]
    missing = requested_names - {str(spec.get("name")) for spec in filtered}
    if missing:
        raise ValueError(f"requested champions are not defined by the source grid: {', '.join(sorted(missing))}")
    return filtered


def _reference_metrics_from_grid(
    results_root: Path,
    collection_dir: Path,
) -> list[tuple[str, str]]:
    """Return reference metrics from the source grid manifest, or defaults."""

    manifest = _source_grid_manifest(results_root, collection_dir)
    configured = manifest.get("champion_reference_metrics") if isinstance(manifest, dict) else None
    if configured:
        return list(_reference_metrics(configured))
    return list(_reference_metrics(None))


def _source_grid_manifest(results_root: Path, collection_dir: Path) -> dict[str, Any] | None:
    """Return the source grid manifest for a collection attempt, if available."""

    source_grid = source_grid_from_attempt(results_root, collection_dir)
    if source_grid is None or not source_grid.manifest_path.is_file():
        return None
    return source_grid.read_manifest()


def _axis_metadata_from_collection(results_root: Path, collection_dir: Path) -> dict[str, Any]:
    """Return axis metadata inherited by a collection attempt."""

    report_path = collection_dir / "collection_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    if isinstance(report, dict) and report.get("config_keys"):
        major_axes = tuple(str(axis) for axis in report.get("major_axes", ()))
        minor_axes = tuple(str(axis) for axis in report.get("minor_axes", ()))
        seed_key = str(report.get("scan_seed_axis", "seed"))
        config_keys = tuple(str(axis) for axis in report.get("config_keys", (*major_axes, *minor_axes)))
        labels = report.get("axis_id_labels") if isinstance(report.get("axis_id_labels"), dict) else {}
        return {
            "major_axes": major_axes,
            "minor_axes": minor_axes,
            "config_keys": config_keys,
            "scan_seed_axis": seed_key,
            "axis_id_labels": {axis: str(labels.get(axis, axis)) for axis in (*config_keys, seed_key)},
        }

    manifest = _source_grid_manifest(results_root, collection_dir)
    axes = grid_axes_from_manifest(manifest)
    config_keys = tuple(axes["config_axes"])
    seed_key = str(axes["scan_seed_axis"])
    labels = axis_id_labels_from_manifest(manifest, (*config_keys, seed_key))
    return {
        "major_axes": tuple(axes["major_axes"]),
        "minor_axes": tuple(axes["minor_axes"]),
        "config_keys": config_keys,
        "scan_seed_axis": seed_key,
        "axis_id_labels": labels,
    }


def _study_from_collection(results_root: Path, collection_dir: Path) -> str:
    """Return the study name inherited by a collection attempt."""

    report_path = collection_dir / "collection_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if isinstance(report, dict) and report.get("study"):
            return study_name_from_manifest(report)
    manifest = _source_grid_manifest(results_root, collection_dir)
    if manifest is not None:
        return study_name_from_manifest(manifest)
    return study_name_from_manifest(None)


def _parse_champion_args(values: Sequence[str] | None) -> list[str] | None:
    """Parse repeated or comma-separated champion-kind CLI values."""

    if values is None:
        return None
    parsed = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parsed


def select(
    *,
    results_root: str | Path,
    collection_attempt_id: str | None = None,
    select_attempt_id: str | None = None,
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] | None = None,
    champion_kinds: Sequence[str] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Select champions from a collection attempt and write a ``04_select`` attempt."""

    results_root = Path(results_root)
    collection_attempt_id = _resolve_collection_attempt(results_root, collection_attempt_id, smoke=smoke)
    select_attempt_id = select_attempt_id or new_attempt_id()
    if smoke:
        select_attempt_id = smoke_attempt_id(select_attempt_id)
    collection_dir = stage_dir(results_root, STAGE_COLLECT) / collection_attempt_id

    rows = read_summary(collection_dir)
    study = _study_from_collection(results_root, collection_dir)
    axis_metadata = _axis_metadata_from_collection(results_root, collection_dir)
    config_keys = tuple(axis_metadata["config_keys"])
    champion_specs = _champion_specs_from_grid(results_root, collection_dir, champion_kinds)
    reference_metrics = _reference_metrics_from_grid(results_root, collection_dir)
    selection = select_champions(
        rows,
        metric=metric,
        mode=mode,
        config_keys=config_keys,
        major_axes=tuple(axis_metadata["major_axes"]),
        minor_axes=tuple(axis_metadata["minor_axes"]),
        seed_key=str(axis_metadata["scan_seed_axis"]),
        axis_id_labels=dict(axis_metadata["axis_id_labels"]),
        group_by=group_by,
        champion_specs=champion_specs,
        reference_metrics=reference_metrics,
    )

    attempt = stage_dir(results_root, STAGE_SELECT) / select_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    champions = selection["champions"]
    group_keys = tuple(selection["group_by"])
    non_group_config_keys = [key for key in config_keys if key not in group_keys]
    columns = [
        *group_keys,
        "winner_kind",
        "config_id",
        "major_id",
        "minor_id",
        *non_group_config_keys,
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
        *_reference_columns(reference_metrics),
        "run_ids",
    ]
    with (attempt / "champions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for champion in champions:
            writer.writerow(champion)

    write_json(
        attempt / "source_collection_attempt.json",
        {
            "collection_attempt_id": collection_attempt_id,
            "collection_attempt_dir": str(collection_dir),
        },
    )
    report = {
        "study": study,
        "stage": STAGE_SELECT,
        "attempt_id": select_attempt_id,
        "smoke": bool(smoke),
        "collection_attempt_id": collection_attempt_id,
        "metric": metric,
        "mode": mode,
        "group_by": selection["group_by"],
        "major_axes": list(axis_metadata["major_axes"]),
        "minor_axes": list(axis_metadata["minor_axes"]),
        "scan_seed_axis": axis_metadata["scan_seed_axis"],
        "axis_id_labels": axis_metadata["axis_id_labels"],
        "champion_kinds": selection["champion_kinds"],
        "champion_specs": selection["champion_specs"],
        "config_keys": selection["config_keys"],
        "seed_aggregation": {
            "value": "median of successful seed rows",
            "error_bar": "sample standard error across successful seed rows",
            "mean": "arithmetic mean across successful seed rows",
        },
        "reference_metrics": selection["reference_metrics"],
        "reference_statistics": list(REFERENCE_STATISTICS),
        "wall_time_metrics": list(WALL_TIME_METRICS),
        "n_candidates": selection["n_candidates"],
        "n_configs": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
        "overall_metric": selection["overall_metric"],
        "overall_metric_value": selection["overall_metric_value"],
        "overall_decisions": selection["overall_decisions"],
        "secondary_champion_kind": selection["secondary_champion_kind"],
        "secondary_metric": selection["secondary_metric"],
        "secondary_champion": selection["secondary_champion"],
        "secondary_metric_value": selection["secondary_metric_value"],
        "secondary_decisions": selection["secondary_decisions"],
        "decisions_by_group": selection["decisions_by_group"],
        "used_status_fallback": selection["used_status_fallback"],
        "champions": champions,
        "configs": selection["configs"],
    }
    write_json(attempt / "selection_report.json", report)
    write_latest(stage_dir(results_root, STAGE_SELECT), select_attempt_id, smoke=smoke)
    return {"attempt_dir": str(attempt), "report": report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse select command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--collection-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None, help="Select attempt id (defaults to now).")
    parser.add_argument("--smoke", action="store_true", help="Select champions from a smoke collection attempt.")
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional scalar metric override. By default, use the ordered local-energy tie-breaker ladder.",
    )
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument(
        "--group-by",
        default=None,
        help="Comma-separated grouping columns for winner buckets (default: source grid major_axes).",
    )
    parser.add_argument(
        "--champions",
        nargs="+",
        default=None,
        help="Champion kinds to select, e.g. 'energy stability'. Defaults to source grid manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Select champions from the command line."""

    args = parse_args(argv)
    result = select(
        results_root=args.results_root,
        collection_attempt_id=args.collection_attempt_id,
        select_attempt_id=args.attempt_id,
        metric=args.metric,
        mode=args.mode,
        group_by=args.group_by,
        champion_kinds=_parse_champion_args(args.champions),
        smoke=args.smoke,
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} selected {report['n_champions']} champions "
        f"(overall {report['overall_champion']}, "
        f"{report['secondary_champion_kind']} {report['secondary_champion']}) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
