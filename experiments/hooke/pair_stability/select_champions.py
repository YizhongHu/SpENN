"""Select pair-stability champions from a collection attempt (PR8.8).

Reads a ``03_collect`` summary table and selects one champion per architecture
(plus an overall champion) by a configurable metric, writing a ``04_select``
attempt with ``champions.csv``, ``selection_report.json``, and an explicit
pointer to the collection attempt consumed.
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
# Lower reference energy error is better; this metric is emitted by the energy
# task's ReferenceEnergySummary. Override with --metric/--mode as needed.
DEFAULT_METRIC = "eval/energy/reference_abs_error"
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


def select_champions(
    rows: Sequence[dict[str, Any]], *, metric: str, mode: str, group_by: str = "architecture"
) -> dict[str, Any]:
    """Select the best row per group plus the overall best."""

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    candidates = [row for row in rows if str(row.get("status", "")) in SUCCESS_STATUSES]
    used_fallback = not candidates
    if used_fallback:
        candidates = list(rows)

    def sort_key(row: dict[str, Any]) -> tuple[float, str]:
        value = _metric_value(row, metric, mode=mode)
        return (value if mode == "min" else -value, str(row.get("run_id", "")))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row.get(group_by, "")), []).append(row)

    champions = []
    for group, group_rows in sorted(groups.items()):
        best = min(group_rows, key=sort_key)
        champions.append(
            {
                group_by: group,
                "run_id": best.get("run_id", ""),
                "normalization": best.get("normalization", ""),
                "lr": best.get("lr", ""),
                "channels": best.get("channels", ""),
                "seed": best.get("seed", ""),
                "metric": metric,
                "metric_value": best.get(metric, ""),
            }
        )

    overall = min(candidates, key=sort_key) if candidates else None
    return {
        "champions": champions,
        "overall_champion": None if overall is None else overall.get("run_id", ""),
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
    metric: str = DEFAULT_METRIC,
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
        "n_candidates": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
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
    parser.add_argument("--metric", default=DEFAULT_METRIC)
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
