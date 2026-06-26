"""Select pair-stability champions from a collection attempt (PR8.8).

Reads a ``03_collect`` summary table, aggregates seed rows into non-seed
configs, and selects two winners per architecture/normalization bucket: one by
the ordered local-energy hierarchy and one by feature-trace stability. Local
energy ranking uses seed medians, while overlap tests use the seed-combined
mean and standard error. An explicit scalar metric can still be passed for
debugging overrides.
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
    attempt_ids,
    new_attempt_id,
    read_json,
    stage_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
CONFIG_KEYS = ("architecture", "normalization", "lr", "channels")
DEFAULT_GROUP_KEYS = ("architecture", "normalization")
ENERGY_TASK_ORDER = ("stratified_geometry", "tail", "cusp", "hooke_orbital")
FEATURE_TRACE_METRIC = "eval/feature_trace_stability/feature_rms_q95"
READOUT_TRACE_METRIC = "eval/readout_trace_stability/condition_number_q95"
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


def _source_energy_metric(task: str) -> str:
    return f"eval/{task}/local_energy_mean"


def _seed_metric(metric: str, statistic: str) -> str:
    return f"{metric}_seed_{statistic}"


def _reference_metrics() -> tuple[tuple[str, str], ...]:
    """Return stable champions.csv reference labels and source metrics."""

    energy_metrics = tuple(
        (f"{task}_energy", _source_energy_metric(task)) for task in ENERGY_TASK_ORDER
    )
    return (
        *energy_metrics,
        ("feature_stability", FEATURE_TRACE_METRIC),
        ("readout_stability", READOUT_TRACE_METRIC),
    )


def _reference_columns() -> list[str]:
    """Return champions.csv columns for seed-aggregated reference metrics."""

    return [
        f"{label}_seed_{statistic}"
        for label, _metric in _reference_metrics()
        for statistic in REFERENCE_STATISTICS
    ]


def _energy_value_metric(task: str) -> str:
    return _seed_metric(_source_energy_metric(task), "median")


def _energy_errorbar_center_metric(task: str) -> str:
    return _seed_metric(_source_energy_metric(task), "mean")


def _energy_stderr_metric(task: str) -> str:
    return _seed_metric(_source_energy_metric(task), "stderr")


def _metric_value(row: dict[str, Any], metric: str, *, mode: str) -> float:
    """Return a sortable metric value, sending missing/non-finite to the worst end."""

    worst = math.inf if mode == "min" else -math.inf
    value = _as_float(row.get(metric), default=worst)
    return value if math.isfinite(value) else worst


def _task_has_energy(rows: Sequence[dict[str, Any]], task: str) -> bool:
    """Return whether any row has a finite energy for ``task``."""

    metric = _energy_value_metric(task)
    return any(math.isfinite(_as_float(row.get(metric))) for row in rows)


def _clearly_beats(a: dict[str, Any], b: dict[str, Any], task: str) -> bool:
    """Return whether row ``a`` beats row ``b`` by non-overlapping error bars."""

    mean_metric = _energy_errorbar_center_metric(task)
    stderr_metric = _energy_stderr_metric(task)
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


def _format_config_id(row: dict[str, Any]) -> str:
    return (
        f"arch-{_key_text(row.get('architecture'))}"
        f"_norm-{_key_text(row.get('normalization'))}"
        f"_lr-{_key_text(row.get('lr'))}"
        f"_ch-{_key_text(row.get('channels'))}"
    )


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


def _numeric_metrics(rows: Sequence[dict[str, Any]]) -> list[str]:
    metrics = []
    for key in sorted({key for row in rows for key in row}):
        if key in {*CONFIG_KEYS, "seed", "status", "run_id"}:
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


def _aggregate_candidates(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Aggregate seed rows into one row per non-seed configuration."""

    successes = [row for row in rows if str(row.get("status", "")) in SUCCESS_STATUSES]
    used_status_fallback = not successes
    value_rows = list(rows) if used_status_fallback else successes
    metrics = _numeric_metrics(value_rows)
    expected_seeds = sorted({_key_text(row.get("seed")) for row in rows if _key_text(row.get("seed"))})

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row, CONFIG_KEYS), []).append(row)

    candidates = []
    for key, group_rows in grouped.items():
        first = group_rows[0]
        seed_rows = {_key_text(row.get("seed")): row for row in group_rows if _key_text(row.get("seed"))}
        seed_order = expected_seeds or sorted(seed_rows)
        run_ids = sorted({_key_text(row.get("run_id")) for row in group_rows if _key_text(row.get("run_id"))})
        n_success = sum(1 for row in seed_rows.values() if str(row.get("status", "")) in SUCCESS_STATUSES)
        n_expected = len(seed_order)
        n_missing_seed = sum(1 for seed in seed_order if seed not in seed_rows)
        candidate: dict[str, Any] = {
            "config_id": _format_config_id(first),
            "run_id": _format_config_id(first),
            "run_ids": ";".join(run_ids),
            "seeds": ",".join(seed_order),
            "seed": ",".join(seed_order),
            "n_expected": n_expected,
            "n_present": len(seed_rows),
            "n_success": n_success,
            "n_failed": n_expected - n_success,
            "n_missing_seed": n_missing_seed,
        }
        for index, config_key in enumerate(CONFIG_KEYS):
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


def _select_by_energy_ladder(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str], str, str]:
    """Select an aggregated config by ordered local-energy diagnostics."""

    remaining = list(rows)
    decisions: list[str] = []
    selected_metric = ""
    selected_value = ""

    for task in ENERGY_TASK_ORDER:
        if not _task_has_energy(remaining, task):
            decisions.append(f"{task}: skipped, no finite local-energy metric in the current cohort")
            continue
        metric = _energy_value_metric(task)
        finite_rows = [row for row in remaining if math.isfinite(_as_float(row.get(metric)))]
        if not finite_rows:
            decisions.append(f"{task}: skipped, no finite local-energy metric in the current cohort")
            continue
        leader = min(finite_rows, key=lambda row: (_as_float(row.get(metric)), _row_label(row)))
        next_remaining = [row for row in finite_rows if row is leader or not _clearly_beats(leader, row, task)]
        selected_metric = metric
        selected_value = str(leader.get(metric, ""))
        if len(next_remaining) == 1:
            decisions.append(f"{task}: {_row_label(leader)} clearly wins by non-overlapping seed error bars")
            return leader, decisions, selected_metric, selected_value
        decisions.append(
            f"{task}: {len(next_remaining)} configs remain because their seed error bars overlap the leader"
        )
        remaining = next_remaining

    selected_metric = "train/runtime/wall_time_sec_seed_median"
    leader = min(remaining, key=lambda row: (_wall_time(row), _row_label(row)))
    selected_value = "" if not math.isfinite(_wall_time(leader)) else str(_wall_time(leader))
    if len(remaining) == 1:
        decisions.append("all energy tie-breakers reduced the cohort to one row")
    else:
        decisions.append("energy tie-breakers exhausted; selected the shortest available wall time")
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


def _select_feature_trace_champion(
    rows: Sequence[dict[str, Any]], *, excluded_config_id: str | None
) -> tuple[dict[str, Any] | None, list[str], str, str]:
    """Select the lowest feature-trace RMS q95, avoiding the energy winner."""

    metric = _seed_metric(FEATURE_TRACE_METRIC, "median")
    finite_rows = [
        row for row in rows if math.isfinite(_metric_value(row, metric, mode="min"))
    ]
    finite_rows.sort(key=lambda row: (_metric_value(row, metric, mode="min"), _row_label(row)))
    if not finite_rows:
        return None, ["no finite feature-trace stability metric found"], metric, ""

    best = finite_rows[0]
    if excluded_config_id is not None and _row_label(best) == excluded_config_id:
        alternatives = [row for row in finite_rows if _row_label(row) != excluded_config_id]
        if not alternatives:
            return (
                None,
                ["best feature-trace config is the energy winner; no distinct alternative is available"],
                metric,
                "",
            )
        best = alternatives[0]
        decisions = [
            "best feature-trace config is the energy winner; selected the next best distinct config"
        ]
    else:
        decisions = ["selected the lowest finite feature-trace RMS q95"]
    return best, decisions, metric, str(best.get(metric, ""))


def _champion_record(
    row: dict[str, Any] | None,
    *,
    group_keys: Sequence[str],
    group_key: Sequence[str],
    winner_kind: str,
    metric: str,
    metric_value: str,
) -> dict[str, Any]:
    """Return one row for ``champions.csv``."""

    record = {key: value for key, value in zip(group_keys, group_key, strict=True)}
    record["winner_kind"] = winner_kind
    record["metric"] = metric
    record["metric_value"] = metric_value
    record["metric_seed_mean"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_mean"), ""))
    record["metric_seed_stderr"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_stderr"), ""))
    record["metric_seed_n"] = "" if row is None else str(row.get(metric.replace("_seed_median", "_seed_n"), ""))
    for label, source_metric in _reference_metrics():
        for statistic in REFERENCE_STATISTICS:
            column = f"{label}_seed_{statistic}"
            record[column] = "" if row is None else str(row.get(_seed_metric(source_metric, statistic), ""))
    if row is None:
        for key in (
            "config_id",
            "run_ids",
            "lr",
            "channels",
            "seeds",
            "n_expected",
            "n_present",
            "n_success",
            "n_failed",
            "n_missing_seed",
        ):
            record[key] = ""
        return record
    record.update(
        config_id=row.get("config_id", ""),
        run_ids=row.get("run_ids", ""),
        lr=row.get("lr", ""),
        channels=row.get("channels", ""),
        seeds=row.get("seeds", ""),
        n_expected=row.get("n_expected", ""),
        n_present=row.get("n_present", ""),
        n_success=row.get("n_success", ""),
        n_failed=row.get("n_failed", ""),
        n_missing_seed=row.get("n_missing_seed", ""),
    )
    return record


def select_champions(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] = DEFAULT_GROUP_KEYS,
) -> dict[str, Any]:
    """Select energy and feature-trace winners per architecture/normalization group."""

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    group_keys = _parse_group_by(group_by)
    candidates, used_fallback = _aggregate_candidates(rows)

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(_group_key(row, group_keys), []).append(row)

    champions = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    for group_key, group_rows in sorted(groups.items()):
        if metric is None:
            energy, energy_decisions, energy_metric, energy_value = _select_by_energy_ladder(group_rows)
        else:
            energy, energy_decisions, energy_metric, energy_value = _select_by_single_metric(
                group_rows, metric=metric, mode=mode
            )
        feature, feature_decisions, feature_metric, feature_value = _select_feature_trace_champion(
            group_rows,
            excluded_config_id=_row_label(energy),
        )
        decisions_by_group[_group_label_from_key(group_keys, group_key)] = {
            "energy": energy_decisions,
            "feature_trace": feature_decisions,
        }
        champions.append(
            _champion_record(
                energy,
                group_keys=group_keys,
                group_key=group_key,
                winner_kind="energy",
                metric=energy_metric,
                metric_value=energy_value,
            )
        )
        champions.append(
            _champion_record(
                feature,
                group_keys=group_keys,
                group_key=group_key,
                winner_kind="feature_trace",
                metric=feature_metric,
                metric_value=feature_value,
            )
        )

    if candidates:
        if metric is None:
            overall, overall_decisions, overall_metric, overall_metric_value = _select_by_energy_ladder(candidates)
        else:
            overall, overall_decisions, overall_metric, overall_metric_value = _select_by_single_metric(
                candidates, metric=metric, mode=mode
            )
    else:
        overall = None
        overall_decisions = []
        overall_metric = ""
        overall_metric_value = ""

    if not candidates:
        excluded_energy_config_id = None
    elif metric is None:
        excluded_energy_config_id = _row_label(overall) if overall is not None else None
    else:
        energy_overall, _, _, _ = _select_by_energy_ladder(candidates)
        excluded_energy_config_id = _row_label(energy_overall)
    feature_trace, feature_trace_decisions, feature_trace_metric, feature_trace_metric_value = (
        _select_feature_trace_champion(candidates, excluded_config_id=excluded_energy_config_id)
    )
    return {
        "champions": champions,
        "configs": candidates,
        "overall_champion": None if overall is None else overall.get("config_id", ""),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_metric_value,
        "overall_decisions": overall_decisions,
        "feature_trace_champion": None if feature_trace is None else feature_trace.get("config_id", ""),
        "feature_trace_metric": feature_trace_metric,
        "feature_trace_metric_value": feature_trace_metric_value,
        "feature_trace_decisions": feature_trace_decisions,
        "decisions_by_group": decisions_by_group,
        "used_status_fallback": used_fallback,
        "group_by": list(group_keys),
        "config_keys": list(CONFIG_KEYS),
        "n_candidates": len(candidates),
    }


def _resolve_collection_attempt(results_root: Path, collection_attempt_id: str | None) -> str:
    if collection_attempt_id is not None:
        return collection_attempt_id
    collect_dir = stage_dir(results_root, STAGE_COLLECT)
    latest = collect_dir / "latest.json"
    if latest.is_file():
        return str(read_json(latest).get("attempt_id"))
    ids = attempt_ids(collect_dir)
    if not ids:
        raise FileNotFoundError(f"no collection attempts under {collect_dir}")
    return ids[-1]


def select(
    *,
    results_root: str | Path,
    collection_attempt_id: str | None = None,
    select_attempt_id: str | None = None,
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] = DEFAULT_GROUP_KEYS,
) -> dict[str, Any]:
    """Select champions from a collection attempt and write a ``04_select`` attempt."""

    results_root = Path(results_root)
    collection_attempt_id = _resolve_collection_attempt(results_root, collection_attempt_id)
    select_attempt_id = select_attempt_id or new_attempt_id()
    collection_dir = stage_dir(results_root, STAGE_COLLECT) / collection_attempt_id

    rows = read_summary(collection_dir)
    selection = select_champions(rows, metric=metric, mode=mode, group_by=group_by)

    attempt = stage_dir(results_root, STAGE_SELECT) / select_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    champions = selection["champions"]
    group_keys = tuple(selection["group_by"])
    columns = [
        *group_keys,
        "winner_kind",
        "config_id",
        "lr",
        "channels",
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
        *_reference_columns(),
        "run_ids",
    ]
    with (attempt / "champions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for champion in champions:
            writer.writerow(champion)

    write_json(attempt / "source_collection_attempt.json", {"collection_attempt_id": collection_attempt_id})
    report = {
        "study": "pair_stability",
        "stage": STAGE_SELECT,
        "attempt_id": select_attempt_id,
        "collection_attempt_id": collection_attempt_id,
        "metric": metric,
        "mode": mode,
        "group_by": selection["group_by"],
        "config_keys": selection["config_keys"],
        "seed_aggregation": {
            "value": "median of successful seed rows",
            "error_bar": "sample standard error across successful seed rows",
            "mean": "arithmetic mean across successful seed rows",
        },
        "energy_task_order": list(ENERGY_TASK_ORDER),
        "reference_metrics": {
            label: source_metric for label, source_metric in _reference_metrics()
        },
        "reference_statistics": list(REFERENCE_STATISTICS),
        "wall_time_metrics": list(WALL_TIME_METRICS),
        "n_candidates": selection["n_candidates"],
        "n_configs": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
        "overall_metric": selection["overall_metric"],
        "overall_metric_value": selection["overall_metric_value"],
        "overall_decisions": selection["overall_decisions"],
        "feature_trace_metric": selection["feature_trace_metric"],
        "feature_trace_champion": selection["feature_trace_champion"],
        "feature_trace_metric_value": selection["feature_trace_metric_value"],
        "feature_trace_decisions": selection["feature_trace_decisions"],
        "decisions_by_group": selection["decisions_by_group"],
        "used_status_fallback": selection["used_status_fallback"],
        "champions": champions,
        "configs": selection["configs"],
    }
    write_json(attempt / "selection_report.json", report)
    return {"attempt_dir": str(attempt), "report": report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse select command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--collection-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None, help="Select attempt id (defaults to now).")
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional scalar metric override. By default, use the ordered local-energy tie-breaker ladder.",
    )
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument(
        "--group-by",
        default=",".join(DEFAULT_GROUP_KEYS),
        help="Comma-separated grouping columns for winner buckets (default: architecture,normalization).",
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
    )
    report = result["report"]
    print(
        f"[pair_stability] selected {report['n_champions']} champions "
        f"(overall {report['overall_champion']}, "
        f"feature-trace {report['feature_trace_champion']}) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
