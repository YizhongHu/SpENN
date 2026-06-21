"""Select pair-stability champions from a collection attempt (PR8.8).

Reads a ``03_collect`` summary table and selects one champion per architecture
(plus an overall champion) by the study's ordered local-energy hierarchy. Energy
means whose standard-error bars overlap stay tied and advance to the next
diagnostic; wall time breaks any remaining tie. An explicit scalar metric can
still be passed for debugging overrides.
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
ENERGY_TASK_ORDER = ("stratified_geometry", "tail", "cusp", "hooke_orbital")
WALL_TIME_METRICS = ("runtime/wall_time_sec", "eval/perf/wall_time_sec")
SUCCESS_STATUSES = {"completed", "success"}


def read_summary(collection_attempt_dir: Path) -> list[dict[str, Any]]:
    """Read the collection ``summary.csv`` rows."""

    summary = collection_attempt_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"collection attempt has no summary.csv: {summary}")
    with summary.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_value(row: dict[str, Any], metric: str, *, mode: str) -> float:
    """Return a sortable metric value, sending missing/non-finite to the worst end."""

    worst = math.inf if mode == "min" else -math.inf
    raw = row.get(metric, "")
    if raw is None or str(raw).strip() == "":
        return worst
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return worst
    if not math.isfinite(value):
        return worst
    return value


def _as_float(value: Any, *, default: float = math.inf) -> float:
    """Return ``value`` as a finite float, or ``default``."""

    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _energy_mean_metric(task: str) -> str:
    return f"eval/{task}/local_energy_mean"


def _energy_stderr_metric(task: str) -> str:
    return f"eval/{task}/local_energy_stderr"


def _task_has_energy(rows: Sequence[dict[str, Any]], task: str) -> bool:
    """Return whether any row has a finite energy for ``task``."""

    metric = _energy_mean_metric(task)
    return any(math.isfinite(_as_float(row.get(metric))) for row in rows)


def _clearly_beats(a: dict[str, Any], b: dict[str, Any], task: str) -> bool:
    """Return whether row ``a`` beats row ``b`` by non-overlapping error bars."""

    mean_metric = _energy_mean_metric(task)
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
        value = _as_float(row.get(metric))
        if math.isfinite(value):
            return value
    return math.inf


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("run_id", ""))


def _select_by_energy_ladder(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str], str, str]:
    """Select a row by ordered local-energy diagnostics and wall-time fallback."""

    remaining = list(rows)
    decisions: list[str] = []
    selected_metric = ""
    selected_value = ""

    for task in ENERGY_TASK_ORDER:
        if not _task_has_energy(remaining, task):
            decisions.append(f"{task}: skipped, no finite local-energy metric in the current cohort")
            continue
        metric = _energy_mean_metric(task)
        finite_rows = [row for row in remaining if math.isfinite(_as_float(row.get(metric)))]
        if not finite_rows:
            decisions.append(f"{task}: skipped, no finite local-energy metric in the current cohort")
            continue
        leader = min(finite_rows, key=lambda row: (_as_float(row.get(metric)), _row_label(row)))
        next_remaining = [row for row in finite_rows if row is leader or not _clearly_beats(leader, row, task)]
        selected_metric = metric
        selected_value = str(leader.get(metric, ""))
        if len(next_remaining) == 1:
            decisions.append(f"{task}: {_row_label(leader)} clearly wins by non-overlapping error bars")
            return leader, decisions, selected_metric, selected_value
        decisions.append(
            f"{task}: {len(next_remaining)} rows remain because their error bars overlap the leader"
        )
        remaining = next_remaining

    selected_metric = "wall_time_sec"
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
    """Select a row by one scalar metric for explicit CLI overrides."""

    def sort_key(row: dict[str, Any]) -> tuple[float, str]:
        value = _metric_value(row, metric, mode=mode)
        return (value if mode == "min" else -value, _row_label(row))

    best = min(rows, key=sort_key)
    return best, [f"selected by explicit scalar metric {metric!r} ({mode})"], metric, str(best.get(metric, ""))


def select_champions(
    rows: Sequence[dict[str, Any]], *, metric: str | None = None, mode: str = "min", group_by: str = "architecture"
) -> dict[str, Any]:
    """Select the best row per group plus the overall best."""

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    candidates = [row for row in rows if str(row.get("status", "")) in SUCCESS_STATUSES]
    used_fallback = not candidates
    if used_fallback:
        candidates = list(rows)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row.get(group_by, "")), []).append(row)

    champions = []
    decisions_by_group: dict[str, list[str]] = {}
    for group, group_rows in sorted(groups.items()):
        if metric is None:
            best, decisions, selected_metric, selected_value = _select_by_energy_ladder(group_rows)
        else:
            best, decisions, selected_metric, selected_value = _select_by_single_metric(
                group_rows, metric=metric, mode=mode
            )
        decisions_by_group[group] = decisions
        champions.append(
            {
                group_by: group,
                "run_id": best.get("run_id", ""),
                "normalization": best.get("normalization", ""),
                "lr": best.get("lr", ""),
                "channels": best.get("channels", ""),
                "seed": best.get("seed", ""),
                "metric": selected_metric,
                "metric_value": selected_value,
            }
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
    return {
        "champions": champions,
        "overall_champion": None if overall is None else overall.get("run_id", ""),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_metric_value,
        "overall_decisions": overall_decisions,
        "decisions_by_group": decisions_by_group,
        "used_status_fallback": used_fallback,
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
    group_by: str = "architecture",
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
    columns = [group_by, "run_id", "normalization", "lr", "channels", "seed", "metric", "metric_value"]
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
        "group_by": group_by,
        "energy_task_order": list(ENERGY_TASK_ORDER),
        "wall_time_metrics": list(WALL_TIME_METRICS),
        "n_candidates": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
        "overall_metric": selection["overall_metric"],
        "overall_metric_value": selection["overall_metric_value"],
        "overall_decisions": selection["overall_decisions"],
        "decisions_by_group": selection["decisions_by_group"],
        "used_status_fallback": selection["used_status_fallback"],
        "champions": champions,
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
    parser.add_argument("--group-by", default="architecture")
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
        f"(overall {report['overall_champion']}) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
