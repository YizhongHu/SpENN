"""Collection and champion-selection replay for the V4 compatibility audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.toolkit.artifacts import duration_from_status
from experiments.toolkit.selection import (
    aggregate_candidates,
    champion_record,
    group_key as selection_group_key,
    group_label_from_key,
    normalize_champion_specs,
    reference_columns,
    reference_metrics as normalize_reference_metrics,
    select_by_spec,
)
import selector_verifier_v1
import selector_verifiers

COLLECTION_BASE_COLUMNS = (
    "run_id",
    "validation_attempt_id",
    "validation_attempt_dir",
    "status",
    "major_id",
    "minor_id",
    "config_id",
    "train_attempt_id",
    "checkpoint_path",
    "n_diagnostics",
)
TRAIN_WALL_TIME_METRIC = "train/runtime/wall_time_sec"
SELECTION_REFERENCE_STATISTICS = ("median", "mean", "stderr")
SELECTION_SUCCESS_STATUSES = {"completed", "success"}
SELECTION_DEFAULT_FALLBACK_METRIC = TRAIN_WALL_TIME_METRIC


def audit_screen_science_status(
    rows: Sequence[Mapping[str, str]],
    *,
    required_tasks: Sequence[str],
    errors: list[str],
) -> None:
    """Require successful suite and task status cells in every summary row."""

    if not required_tasks:
        errors.append("screen summary has no configured evaluation task contract")
        return
    for row in rows:
        run_id = row.get("run_id", "")
        if csv_bool(row.get("eval/status/suite_success")) is not True:
            errors.append(f"screen summary {run_id} suite_success is not true")
        if csv_bool(row.get("eval/status/suite_failed")) is not False:
            errors.append(f"screen summary {run_id} suite_failed is not false")
        for task in required_tasks:
            if csv_bool(
                row.get(f"eval/{task}/status/task_success")
            ) is not True:
                errors.append(
                    f"screen summary {run_id} {task} task_success is not true"
                )
            if csv_bool(
                row.get(f"eval/{task}/status/task_failed")
            ) is not False:
                errors.append(
                    f"screen summary {run_id} {task} task_failed is not false"
                )
        if any(
            key.endswith("/status/task_failed") and csv_bool(value) is True
            for key, value in row.items()
        ):
            errors.append(f"screen summary {run_id} contains task_failed=true")


def audit_selection_replay(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, str]],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, str]],
    errors: list[str],
) -> None:
    """Replay V3 champion selection solely through the generic toolkit."""

    try:
        replay = replay_selection(grid=grid, summary_rows=summary_rows)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot replay champion selection: {exc}")
        return
    expected_report_fields = {
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
    for key, expected in expected_report_fields.items():
        if selection_report.get(key) != expected:
            errors.append(f"selection replay differs for report field {key}")

    expected_columns = selection_champion_columns(replay)
    observed_columns = tuple(champion_rows[0]) if champion_rows else ()
    if observed_columns != expected_columns:
        errors.append("champions.csv header differs from selection replay")
    if len(champion_rows) != len(replay["champions"]):
        errors.append("champions.csv row count differs from selection replay")
        return
    for index, (actual, expected) in enumerate(
        zip(champion_rows, replay["champions"], strict=True)
    ):
        for key in expected_columns:
            if actual.get(key, "") != metric_csv_scalar(expected.get(key, "")):
                errors.append(
                    f"champions.csv row {index} differs from selection replay "
                    f"for {key}"
                )


def replay_selection(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the deterministic frozen V3 selector result."""

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
    references = normalize_reference_metrics(
        grid.get("champion_reference_metrics")
    )

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
        key = selection_group_key(candidate, major_axes)
        groups.setdefault(key, []).append(candidate)

    champions: list[dict[str, Any]] = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    for group_key, group_rows in sorted(groups.items()):
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
                    group_key=group_key,
                    config_keys=config_keys,
                    winner_kind=kind,
                    metric=metric,
                    metric_value=metric_value,
                    reference_metrics=references,
                    reference_statistics=SELECTION_REFERENCE_STATISTICS,
                )
            )
        decisions_by_group[
            group_label_from_key(major_axes, group_key)
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


def selection_champion_columns(
    replay: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the exact V3 ``champions.csv`` header for a replay."""

    group_keys = tuple(str(value) for value in replay["group_by"])
    config_keys = tuple(str(value) for value in replay["config_keys"])
    references = normalize_reference_metrics(replay["reference_metrics"])
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


def selection_contract(
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    selection_toolkit_sha256: str,
) -> dict[str, Any]:
    """Cross-check the mutable producer with immutable V1 semantics."""

    return selector_verifiers.build_contract(
        selector_verifier_v1.VERIFIER_ID,
        grid=grid,
        summary_rows=summary_rows,
        selection_report=selection_report,
        champion_rows=champion_rows,
        artifact_sha256=artifact_sha256,
        producer_replay=replay,
        producer_source_path="experiments/toolkit/selection.py",
        producer_source_sha256=selection_toolkit_sha256,
    )


def canonical_sha256(value: object) -> str:
    """Return SHA-256 of canonical JSON bytes."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_collection_metric_reconciliation(
    root: Path,
    *,
    rows: Sequence[Mapping[str, str]],
    grid: Mapping[str, Any],
    collection_report: Mapping[str, Any],
    validation_attempt: str,
    errors: list[str],
) -> None:
    """Replay the V3 collection projection from its declared source artifacts."""

    by_run: dict[str, Mapping[str, str]] = {}
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in by_run:
            errors.append("screen summary contains duplicate/empty run ids")
            continue
        by_run[run_id] = row
    major_axes = tuple(str(value) for value in grid.get("major_axes", ()))
    minor_axes = tuple(str(value) for value in grid.get("minor_axes", ()))
    seed_axis = str(grid.get("scan_seed_axis") or "")
    run_axes = (*major_axes, *minor_axes, seed_axis)
    if not seed_axis:
        errors.append("collection source grid has no scan seed axis")
    base_columns = (*COLLECTION_BASE_COLUMNS, *run_axes)
    required_train_metrics = _required_train_metrics(grid)
    if collection_report.get("required_train_metrics") != sorted(
        required_train_metrics
    ):
        errors.append(
            "collection required_train_metrics differ from source selector"
        )

    jobs = [
        job
        for job in grid.get("jobs", ())
        if isinstance(job, Mapping) and job.get("run_id")
    ]
    expected_order = [str(job["run_id"]) for job in jobs]
    observed_order = [str(row.get("run_id") or "") for row in rows]
    if observed_order != expected_order:
        errors.append("screen summary row order differs from grid manifest")

    expected_rows: dict[str, dict[str, Any]] = {}
    expected_metric_union: set[str] = set()
    for job in jobs:
        run_id = str(job["run_id"])
        validation_dir = (
            root / "02_validation" / run_id / validation_attempt
        )
        source = _read_json_for_audit(
            validation_dir / "source_train_attempt.json",
            errors,
        )
        expected = {column: "" for column in base_columns}
        choices = job.get("choices")
        if not isinstance(choices, Mapping):
            choices = {}
        expected.update(
            {
                "run_id": run_id,
                "validation_attempt_id": validation_attempt,
                "validation_attempt_dir": str(validation_dir),
                "status": _status_of_for_audit(validation_dir, errors),
                "major_id": job.get("major_id", ""),
                "minor_id": job.get("minor_id", ""),
                "config_id": job.get("config_id", ""),
                "train_attempt_id": source.get("train_attempt_id", ""),
                "checkpoint_path": source.get("checkpoint_path", ""),
                "n_diagnostics": _count_diagnostics_for_audit(
                    validation_dir,
                    errors,
                ),
            }
        )
        expected.update({axis: choices.get(axis, "") for axis in run_axes})
        expected.update(
            _required_train_metric_projection(
                source,
                required_metrics=required_train_metrics,
                errors=errors,
            )
        )
        expected.update(
            _read_metric_map_for_audit(
                validation_dir / "metrics.jsonl",
                errors,
            )
        )
        expected_rows[run_id] = expected
        expected_metric_union.update(set(expected) - set(base_columns))

    expected_metric_columns = sorted(expected_metric_union)
    if collection_report.get("metric_columns") != expected_metric_columns:
        errors.append("collection metric_columns differ from replayed projection")
    expected_header = (*base_columns, *expected_metric_columns)
    observed_header = tuple(rows[0]) if rows else ()
    if observed_header != expected_header:
        errors.append("screen summary header differs from replayed projection")
    if any(tuple(row) != observed_header for row in rows):
        errors.append("screen summary rows do not share one exact header")

    for run_id, row in by_run.items():
        expected = expected_rows.get(run_id)
        if expected is None:
            errors.append(f"screen summary run is absent from grid: {run_id}")
            continue
        for key in expected_header:
            expected_value = metric_csv_scalar(expected.get(key, ""))
            if row.get(key, "") != expected_value:
                errors.append(
                    "screen summary cell differs from replayed projection: "
                    f"{run_id} {key}"
                )


def _required_train_metrics(grid: Mapping[str, Any]) -> set[str]:
    metrics: set[str] = set()
    for spec in grid.get("champions", ()) or ():
        if not isinstance(spec, Mapping):
            continue
        for key in ("metric", "fallback_metric"):
            metric = str(spec.get(key) or "").strip()
            if metric.startswith("train/"):
                metrics.add(metric)
    for spec in grid.get("champion_reference_metrics", ()) or ():
        if not isinstance(spec, Mapping):
            continue
        metric = str(spec.get("metric") or "").strip()
        if metric.startswith("train/"):
            metrics.add(metric)
    return metrics


def _required_train_metric_projection(
    source: Mapping[str, Any],
    *,
    required_metrics: set[str],
    errors: list[str],
) -> dict[str, Any]:
    if not required_metrics:
        return {}
    train_attempt_value = source.get("train_attempt_dir")
    if not isinstance(train_attempt_value, str) or not train_attempt_value:
        checkpoint_value = source.get("checkpoint_path")
        checkpoint = (
            Path(checkpoint_value)
            if isinstance(checkpoint_value, str) and checkpoint_value
            else None
        )
        train_attempt = (
            checkpoint.parent
            if checkpoint is not None and checkpoint.name == "checkpoints"
            else None
        )
    else:
        train_attempt = Path(train_attempt_value)
    if train_attempt is None:
        return {}
    output: dict[str, Any] = {}
    pending = set(required_metrics)
    if TRAIN_WALL_TIME_METRIC in pending:
        status = _read_json_for_audit(train_attempt / "status.json", errors)
        duration = duration_from_status(status, clamp_negative=True)
        if duration is not None:
            output[TRAIN_WALL_TIME_METRIC] = duration
            pending.remove(TRAIN_WALL_TIME_METRIC)
    if pending:
        raw = _read_metric_map_for_audit(
            train_attempt / "metrics.jsonl",
            errors,
            prefix="train",
        )
        output.update({key: raw[key] for key in pending if key in raw})
    return output


def _status_of_for_audit(attempt_dir: Path, errors: list[str]) -> str:
    status = _read_json_for_audit(attempt_dir / "status.json", errors)
    return str(status.get("status", "unknown")) if status else "missing_status"


def _count_diagnostics_for_audit(
    attempt_dir: Path,
    errors: list[str],
) -> int:
    index_path = attempt_dir / "diagnostics" / "index.json"
    if index_path.is_file():
        try:
            value = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid diagnostics index {index_path}: {exc}")
            value = None
        if isinstance(value, Mapping) and isinstance(
            value.get("artifacts"),
            list,
        ):
            return len(value["artifacts"])
        if isinstance(value, list):
            return len(value)
    try:
        return sum(
            1
            for child in attempt_dir.iterdir()
            if child.is_dir()
            and child.name not in {"checkpoints", "diagnostics"}
        )
    except OSError as exc:
        errors.append(f"cannot scan diagnostics in {attempt_dir}: {exc}")
        return 0


def _read_metric_map_for_audit(
    path: Path,
    errors: list[str],
    *,
    prefix: str | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not path.is_file() or path.is_symlink():
        errors.append(f"missing metrics JSONL: {path}")
        return output
    line_number = 0
    try:
        for line_number, line in enumerate(
            path.read_text().splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("metric row is not an object")
            namespace = str(record.get("namespace") or "").strip("/")
            if "metrics" in record:
                metrics = record.get("metrics")
                if not isinstance(metrics, dict):
                    raise ValueError("metric row has invalid metric mapping")
                items = metrics.items()
            elif "metric" in record and "value" in record:
                items = ((record["metric"], record["value"]),)
            else:
                continue
            for name, value in items:
                key = f"{namespace}/{name}" if namespace else str(name)
                if prefix:
                    key = f"{prefix}/{key}" if key else prefix
                output[key] = value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid metrics JSONL {path}:{line_number}: {exc}")
    return output


def metric_csv_scalar(value: object) -> str:
    """Return the stable CSV string produced by the toolkit writer."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, sort_keys=True)


def csv_bool(value: object) -> bool | None:
    """Parse the toolkit's boolean CSV representation."""

    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _read_json_for_audit(
    path: Path,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid JSON: {path}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value
