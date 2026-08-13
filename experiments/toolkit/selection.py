"""Generic, layout-agnostic champion-selection engine for staged runs.

Operates purely on the ``summary.csv`` row-dict contract produced by a collect
stage: a champion is chosen per grouping bucket by an ordered metric ladder or a
single scalar metric, with non-overlapping seed error bars breaking ties and a
configurable fallback metric closing them out. Spec/reference normalization,
group-by parsing, overlap logic, split-sample seed partitioning, and the
per-bucket distribution diagnostic live here too. Nothing in this module
imports study code or ``spenn``; study-specific defaults (success statuses,
reference statistics, wall-time metric, fallback metric name) and the
``id_for_axes`` id-builder are passed in explicitly by the caller.

Seed -> trial aggregation (the ``_numeric_metrics`` / ``_aggregate_metric`` /
``aggregate_candidates`` section below) is COLLECTOR-owned in the incoming
experiment-stack design: it will move to ``MetricCollector.collect_trial_results``
once that contract exists. It is co-located here only until then and is NOT part
of the selector concept.
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Split-sample selection and per-bucket distributions
#
# Both exist for the same reason: `min` over many noisy candidates is a winner's
# curse. The argmin is biased toward whichever candidate's noise fell favorably,
# and when buckets are COMPARED by their argmins that bias is differential -- the
# noisier bucket wins by an artifact. Re-measuring the selected candidate on fresh
# seeds fixes its reported value and does nothing for its identity, because
# selection already happened on the noisy data.
#
# Split-sample selection fixes the identity: choose on one set of seed rows,
# report on a set that had no vote. The distribution summary is the diagnostic
# that says whether it mattered -- if a bucket ranking flips between champion and
# median, the champion ranking was noise.
# ---------------------------------------------------------------------------


def split_sample_seeds(spec: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return one spec's ``(selection_seeds, holdout_seeds)``, or ``None``.

    Seed labels are returned as text because they are matched against
    ``summary.csv`` cells, which are text regardless of how the grid spelled them.

    Returns
    -------
    tuple of tuple of str, or None
        ``None`` when the spec declares no split, i.e. selection reads every seed
        row -- the pre-split-sample behaviour, kept so a grid that predates this
        surface still selects exactly as it did.

    Raises
    ------
    ValueError
        When a holdout is declared without a selection sample, or when the two
        sets intersect. An intersecting holdout is not a degraded split; it is a
        leak that silently restores the bias the split exists to remove, so it
        fails loudly rather than being trimmed to something valid.
    """

    selection = tuple(
        _key_text(seed) for seed in (spec.get("selection_seeds") or ()) if _key_text(seed)
    )
    holdout = tuple(_key_text(seed) for seed in (spec.get("holdout_seeds") or ()) if _key_text(seed))
    name = spec.get("name", "<unnamed>")
    if not selection:
        if holdout:
            raise ValueError(
                f"champion {name!r} declares holdout_seeds without selection_seeds; "
                "a holdout is only meaningful against an explicit selection sample"
            )
        return None
    overlap = sorted(set(selection) & set(holdout))
    if overlap:
        raise ValueError(
            f"champion {name!r} split-sample seeds overlap on {', '.join(overlap)}; "
            "a seed that selects the champion cannot also evaluate it"
        )
    return selection, holdout


def rows_for_seeds(
    rows: Sequence[dict[str, Any]], *, seed_key: str, seeds: Sequence[str]
) -> list[dict[str, Any]]:
    """Return the rows whose ``seed_key`` cell is one of ``seeds``.

    Raises
    ------
    ValueError
        When ``seeds`` is empty, or when no row carries any of them. A silently
        empty sample would make the selector fall through to its fallback on a
        cohort of zero rows, which reads as "the metrics were missing" rather than
        as "the requested seeds are not in this collection".
    """

    wanted = {_key_text(seed) for seed in seeds if _key_text(seed)}
    if not wanted:
        raise ValueError("rows_for_seeds requires at least one seed")
    selected = [row for row in rows if _key_text(row.get(seed_key)) in wanted]
    if not selected:
        raise ValueError(
            f"no summary row carries {seed_key} in {{{', '.join(sorted(wanted))}}}; "
            "the collection does not contain the requested seed rows"
        )
    return selected


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Return the linearly interpolated ``q`` quantile of a sorted sequence."""

    if not ordered:
        return math.inf
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * float(q)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low]) * (1.0 - weight) + float(ordered[high]) * weight


def metric_distribution(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    best_k: int = 3,
    mode: str = "min",
) -> dict[str, Any]:
    """Summarize one metric's distribution across a bucket's aggregated configs.

    Parameters
    ----------
    rows
        Aggregated per-configuration rows for ONE bucket, as produced by
        :func:`aggregate_candidates`.
    metric
        Source metric name; its seed-median column is read.
    best_k
        How many leading configs to name, in the direction ``mode`` implies.
    mode
        ``"min"`` or ``"max"``, defining which end of the distribution is "best".

    Returns
    -------
    dict
        ``n``, ``median``, ``q1``, ``q3``, ``best``, ``worst`` and a ``best_k``
        list of ``{"config_id", "value"}`` entries. Numbers are formatted with
        :func:`_csv_number` so the block is JSON- and CSV-safe.
    """

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    column = metric if metric.endswith("_seed_median") else _seed_metric(metric, "median")
    pairs = [
        (_as_float(row.get(column)), _row_label(row))
        for row in rows
        if math.isfinite(_as_float(row.get(column)))
    ]
    if not pairs:
        return {"metric": metric, "column": column, "n": 0, "best_k": []}
    values = sorted(value for value, _label in pairs)
    leading = sorted(pairs, key=lambda pair: (pair[0] if mode == "min" else -pair[0], pair[1]))
    return {
        "metric": metric,
        "column": column,
        "mode": mode,
        "n": len(values),
        "median": _csv_number(_quantile(values, 0.5)),
        "q1": _csv_number(_quantile(values, 0.25)),
        "q3": _csv_number(_quantile(values, 0.75)),
        "best": _csv_number(values[0] if mode == "min" else values[-1]),
        "worst": _csv_number(values[-1] if mode == "min" else values[0]),
        "best_k": [
            {"config_id": label, "value": _csv_number(value)}
            for value, label in leading[: max(0, int(best_k))]
        ],
    }


def ladder_metrics(spec: Mapping[str, Any]) -> list[str]:
    """Return the ordered ladder rungs one ``metric_ladder`` spec declares.

    Exactly one of two spellings, because they answer different needs and a spec
    that used both would have two ladders:

    ``metrics``
        Full metric names, in declared order. Required when the rungs are
        heterogeneous -- a study whose primary criterion is a MEAN on one task and
        whose secondary is a VARIANCE on another cannot be spelled by any single
        ``{task}`` template, and forcing it into one is how a selector ends up
        ranking on a key nobody chose.
    ``tasks`` + ``metric_template``
        One template applied to each task name, for a ladder whose rungs really
        are the same metric on different tasks.

    Raises
    ------
    ValueError
        When both spellings are present, when neither is, or when a declared
        spelling is empty.
    """

    metrics = [str(metric).strip() for metric in (spec.get("metrics") or []) if str(metric).strip()]
    tasks = [str(task).strip() for task in (spec.get("tasks") or []) if str(task).strip()]
    template = str(spec.get("metric_template", "")).strip()
    name = spec.get("name", "<unnamed>")
    if metrics and (tasks or template):
        raise ValueError(
            f"champion {name!r} metric_ladder declares both metrics and tasks/metric_template; "
            "use one spelling"
        )
    if metrics:
        return metrics
    if not tasks:
        raise ValueError(f"champion {name!r} metric_ladder requires metrics or tasks")
    if not template:
        raise ValueError(f"champion {name!r} metric_ladder requires metric_template")
    return [template.format(task=task) for task in tasks]


def _select_by_metric_ladder(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    mode: str = "min",
    fallback_metric: str,
    fallback_mode: str = "min",
) -> tuple[dict[str, Any], list[str], str, str]:
    """Select an aggregated config by an ordered metric ladder."""

    if mode != "min":
        raise ValueError("metric_ladder currently supports mode='min' only")
    if not metrics:
        raise ValueError("metric_ladder requires at least one metric")
    remaining = list(rows)
    decisions: list[str] = []
    selected_metric = ""
    selected_value = ""

    for source_metric in metrics:
        if not _task_has_metric(remaining, source_metric):
            decisions.append(f"{source_metric}: skipped, no finite value in the current cohort")
            continue
        metric = _seed_metric(source_metric, "median")
        finite_rows = [row for row in remaining if math.isfinite(_as_float(row.get(metric)))]
        if not finite_rows:
            decisions.append(f"{source_metric}: skipped, no finite value in the current cohort")
            continue
        leader = min(finite_rows, key=lambda row: (_as_float(row.get(metric)), _row_label(row)))
        next_remaining = [
            row for row in finite_rows if row is leader or not _clearly_beats(leader, row, source_metric)
        ]
        selected_metric = metric
        selected_value = str(leader.get(metric, ""))
        if len(next_remaining) == 1:
            decisions.append(
                f"{source_metric}: {_row_label(leader)} clearly wins by non-overlapping seed error bars"
            )
            return leader, decisions, selected_metric, selected_value
        decisions.append(
            f"{source_metric}: {len(next_remaining)} configs remain because their seed error bars "
            "overlap the leader"
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
        return _select_by_metric_ladder(
            rows,
            metrics=ladder_metrics(spec),
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
