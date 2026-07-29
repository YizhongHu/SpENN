"""Read-only ownership and completed-lineage audits for pair-stability V4-0."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_STUDY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STUDY_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit_receipts import (  # noqa: E402
    inventory_results_tree,
    inventory_source_tree,
)
from fanout_audit import (  # noqa: E402
    SCIENCE_METRIC_ANCHORS,
    STAGE_EXPECTATIONS,
    _expected_submitted_command,
    audit_gpu_test_fanout_profile,
    audit_fanout_stages,
)
from roots import (
    PURPOSE_EXPERIMENT,
    ROOT_SENTINEL,
    require_v4_root,
    root_metadata,
    validate_lineage_id,
    validate_root_links,
)
from routes import V3_STUDY_DIR  # noqa: E402
from science_audit import (  # noqa: E402
    SELECTION_REFERENCE_STATISTICS,
    TRAIN_WALL_TIME_METRIC,
    audit_collection_metric_reconciliation as _audit_collection_metric_reconciliation,
    audit_screen_science_status as _audit_screen_science_status,
    audit_selection_replay as _audit_selection_replay,
    metric_csv_scalar as _metric_csv_scalar,
    replay_selection,
    selection_contract,
    selection_champion_columns,
)
from smoke_plan import (  # noqa: E402
    EXPECTED_BLIND_SEED,
    SMOKE_PLAN_CONTRACT_PATH,
    audit_smoke_plan,
    smoke_plan_digest,
    smoke_plan_projection,
    verify_smoke_plan_contract,
)
from strict_data import (  # noqa: E402
    StrictDataError,
    iter_jsonl,
    load_json,
    load_yaml,
    validate_structured_tree,
)

EXPECTED_SCAN_COUNT = 64
EXPECTED_FINAL_COUNT = 8
EXPECTED_FINAL_COLLECT_TABLES = {
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
    "cost_by_run.csv",
    "cost_by_axis.csv",
    "cost_by_task.csv",
}


def audit_completed_lineage(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[str, ...]:
    """Return every reason one lineage cannot become V4-0 reference."""

    errors: list[str] = []
    try:
        root, expected_study = _resolve_lineage_root(results_root)
        normalized = _normalize_attempts(attempts)
    except (OSError, ValueError) as exc:
        return (str(exc),)
    try:
        validate_structured_tree(root)
    except StrictDataError as exc:
        return (f"invalid structured acceptance evidence: {exc}",)
    errors.extend(audit_gpu_test_fanout_profile())

    lineage_id = next(iter(normalized.values()))
    if (root / ROOT_SENTINEL).exists():
        lineage_id = str(root_metadata(root)["lineage_id"])
    if set(normalized.values()) != {lineage_id}:
        errors.append(
            "every V4-0 stage attempt must equal the guarded root lineage_id"
        )

    paths = _lineage_paths(root, normalized)
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {label}: {path.relative_to(root)}")

    grid = _read_json_for_audit(paths["grid_manifest"], errors)
    if grid:
        _expect(
            grid.get("study") == expected_study,
            f"grid study is not {expected_study}",
            errors,
        )
        _expect(grid.get("attempt_id") == normalized["grid"], "grid attempt mismatch", errors)
        _expect(int(grid.get("n_jobs", -1)) == EXPECTED_SCAN_COUNT, "grid n_jobs is not 64", errors)
        _expect(bool(grid.get("blinding", {}).get("enabled")), "grid is not blinded", errors)
        _expect(
            int(grid.get("blinding", {}).get("blind_seed", -1))
            == EXPECTED_BLIND_SEED,
            "grid blind seed is not 811",
            errors,
        )
        _expect(
            int(grid.get("final_replicates", -1)) == 1,
            "grid final_replicates is not 1",
            errors,
        )
        errors.extend(
            audit_smoke_plan(
                root,
                attempt=normalized["grid"],
                manifest=grid,
                expected_study=expected_study,
            )
        )

    collection = _read_json_for_audit(paths["collection_report"], errors)
    if collection:
        _expect(
            collection.get("study") == expected_study,
            f"collection study is not {expected_study}",
            errors,
        )
        _expect(
            collection.get("stage") == "03_collect",
            "collection stage mismatch",
            errors,
        )
        _expect(
            collection.get("attempt_id") == normalized["collection"],
            "collection attempt mismatch",
            errors,
        )
        _expect(
            int(collection.get("n_collected", -1)) == EXPECTED_SCAN_COUNT,
            "collection n_collected is not 64",
            errors,
        )
        _expect(
            int(collection.get("n_failures", -1)) == 0,
            "collection n_failures is not 0",
            errors,
        )
        _expect(
            collection.get("grid_attempt_id") == normalized["grid"],
            "collection source grid attempt mismatch",
            errors,
        )
        _audit_source_record(
            paths["collection_source_grid"],
            expected={
                "grid_attempt_id": normalized["grid"],
                "grid_attempt_dir": str(
                    root / "00_grid" / normalized["grid"]
                ),
                "manifest_path": str(paths["grid_manifest"]),
            },
            label="collection source-grid",
            errors=errors,
        )
        if grid:
            for key, expected in {
                "major_axes": grid.get("major_axes"),
                "minor_axes": grid.get("minor_axes"),
                "scan_seed_axis": grid.get("scan_seed_axis"),
                "axis_id_labels": grid.get("axis_id_labels"),
                "config_keys": [
                    *list(grid.get("major_axes") or []),
                    *list(grid.get("minor_axes") or []),
                ],
            }.items():
                if collection.get(key) != expected:
                    errors.append(f"collection {key} differs from grid")

    selection = _read_json_for_audit(paths["selection_report"], errors)
    if selection:
        for key, expected in {
            "study": expected_study,
            "stage": "04_select",
            "attempt_id": normalized["selection"],
            "collection_attempt_id": normalized["collection"],
        }.items():
            if selection.get(key) != expected:
                errors.append(
                    f"selection {key}={selection.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if int(selection.get("n_champions", -1)) != EXPECTED_FINAL_COUNT:
            errors.append("selection n_champions is not 8")
        if int(selection.get("n_candidates", -1)) != EXPECTED_SCAN_COUNT:
            errors.append("selection n_candidates is not 64")
        if int(selection.get("n_configs", -1)) != EXPECTED_SCAN_COUNT:
            errors.append("selection n_configs is not 64")
        if selection.get("used_status_fallback") is not False:
            errors.append("selection used status fallback")
        if grid:
            for key, expected in {
                "major_axes": grid.get("major_axes"),
                "minor_axes": grid.get("minor_axes"),
                "scan_seed_axis": grid.get("scan_seed_axis"),
                "axis_id_labels": grid.get("axis_id_labels"),
                "group_by": grid.get("major_axes"),
                "config_keys": [
                    *list(grid.get("major_axes") or []),
                    *list(grid.get("minor_axes") or []),
                ],
                "champion_specs": grid.get("champions"),
                "champion_kinds": [
                    str(row.get("name"))
                    for row in list(grid.get("champions") or [])
                    if isinstance(row, dict)
                ],
            }.items():
                if selection.get(key) != expected:
                    errors.append(f"selection {key} differs from grid")
        _audit_source_record(
            paths["selection_source_collection"],
            expected={
                "collection_attempt_id": normalized["collection"],
                "collection_attempt_dir": str(
                    root / "03_collect" / normalized["collection"]
                ),
            },
            label="selection source-collection",
            errors=errors,
        )

    final_grid = _read_json_for_audit(paths["final_grid_manifest"], errors)
    if final_grid:
        _expect(
            final_grid.get("study") == expected_study,
            f"final grid study is not {expected_study}",
            errors,
        )
        _expect(
            int(final_grid.get("replicates", -1)) == 1,
            "final grid replicates is not 1",
            errors,
        )
        _expect(
            int(final_grid.get("final_replicates", -1)) == 1,
            "final grid final_replicates is not 1",
            errors,
        )
        _expect(
            int(final_grid.get("n_jobs", -1)) == EXPECTED_FINAL_COUNT,
            "final grid n_jobs is not 8",
            errors,
        )
        for key, expected in {
            "stage": "05_final_grid",
            "attempt_id": normalized["final_grid"],
            "source_selection_attempt_id": normalized["selection"],
            "source_selection_attempt_dir": str(
                root / "04_select" / normalized["selection"]
            ),
            "train_config": str(
                root
                / "00_grid"
                / normalized["grid"]
                / "train_config.yaml"
            ),
            "eval_config": str(
                root
                / "00_grid"
                / normalized["grid"]
                / "validation_config.yaml"
            ),
            "results_root": str(root),
            "n_source_champions": EXPECTED_FINAL_COUNT,
        }.items():
            if final_grid.get(key) != expected:
                errors.append(
                    f"final grid {key}={final_grid.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if grid:
            for key in (
                "config_snapshots",
                "major_axes",
                "minor_axes",
                "axis_id_labels",
                "axis_overrides",
                "seed_overrides",
                "final_seed_sequences",
                "static_overrides",
            ):
                if final_grid.get(key) != grid.get(key):
                    errors.append(f"final grid {key} differs from grid")
            expected_kinds = sorted(
                str(row.get("name"))
                for row in list(grid.get("champions") or [])
                if isinstance(row, dict)
            )
            if final_grid.get("champion_kinds") != expected_kinds:
                errors.append("final grid champion_kinds differs from grid")
        _audit_source_record(
            paths["final_grid_source_selection"],
            expected={
                "selection_attempt_id": normalized["selection"],
                "selection_attempt_dir": str(
                    root / "04_select" / normalized["selection"]
                ),
                "champions_path": str(paths["champions"]),
            },
            label="final-grid source-selection",
            errors=errors,
        )
        try:
            if (
                paths["champions"].read_bytes()
                != paths["final_grid_source_champions"].read_bytes()
            ):
                errors.append(
                    "final-grid source_champions does not exactly copy champions"
                )
        except OSError as exc:
            errors.append(f"cannot compare source champions: {exc}")

    try:
        evaluation_tasks = {
            "02_validation": _configured_evaluation_tasks(
                root
                / "00_grid"
                / normalized["grid"]
                / "validation_config.yaml",
                suite="validation",
            ),
            "07_final_eval": _configured_evaluation_tasks(
                root
                / "00_grid"
                / normalized["grid"]
                / "validation_config.yaml",
                suite="final_eval",
            ),
        }
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"cannot resolve configured evaluation tasks: {exc}")
        evaluation_tasks = {}
    stage_run_ids, worker_commits, fanout_errors, _fanout_evidence = audit_fanout_stages(
        root,
        attempt=lineage_id,
        expected_study=expected_study,
        evaluation_tasks=evaluation_tasks,
    )
    errors.extend(fanout_errors)
    if len(worker_commits) != 1:
        errors.append(
            f"fan-out workers record {len(worker_commits)} distinct git commits"
        )

    scan_run_ids = {
        str(job.get("run_id"))
        for job in grid.get("jobs", [])
        if isinstance(job, dict) and job.get("run_id")
    }
    if len(scan_run_ids) != EXPECTED_SCAN_COUNT:
        errors.append(
            f"grid run-id population is {len(scan_run_ids)}, "
            f"expected {EXPECTED_SCAN_COUNT}"
        )
    for stage in ("01_train", "02_validation"):
        if stage_run_ids.get(stage) != scan_run_ids:
            errors.append(f"{stage} run-id population differs from 00_grid")
    _audit_task_lineage(
        paths["collection_lineage"],
        expected_row_ids=scan_run_ids,
        required_task_kinds=("train", "validation"),
        errors=errors,
    )
    _audit_collection_task_lineage_values(
        paths["collection_lineage"],
        attempt=normalized["validation"],
        scan_run_ids=scan_run_ids,
        errors=errors,
    )

    final_jobs = _read_csv(
        root / "05_final_grid" / normalized["final_grid"] / "final_jobs.csv",
        errors,
    )
    final_run_ids = {
        str(row.get("final_run_id"))
        for row in final_jobs
        if row.get("final_run_id")
    }
    if len(final_run_ids) != EXPECTED_FINAL_COUNT:
        errors.append(
            f"final-grid run-id population is {len(final_run_ids)}, "
            f"expected {EXPECTED_FINAL_COUNT}"
        )
    for stage in ("06_final_train", "07_final_eval"):
        if stage_run_ids.get(stage) != final_run_ids:
            errors.append(f"{stage} run-id population differs from 05_final_grid")
    champion_rows = _read_csv(paths["champions"], errors)
    if len(champion_rows) != EXPECTED_FINAL_COUNT:
        errors.append(
            f"champion row count is {len(champion_rows)}, expected 8"
        )
    _audit_task_lineage(
        paths["selection_lineage"],
        expected_count=EXPECTED_FINAL_COUNT,
        require_nonempty_tasks=True,
        errors=errors,
    )
    _audit_selection_rows(
        selection,
        champion_rows=champion_rows,
        lineage_path=paths["selection_lineage"],
        major_axes=tuple(grid.get("major_axes") or ()),
        scan_run_ids=scan_run_ids,
        attempt=normalized["validation"],
        errors=errors,
    )
    _audit_task_lineage(
        paths["final_grid_lineage"],
        expected_row_ids=final_run_ids,
        require_nonempty_tasks=True,
        errors=errors,
    )
    _audit_final_grid_rows(
        final_jobs,
        champion_rows=champion_rows,
        lineage_path=paths["final_grid_lineage"],
        final_seed_sequences=grid.get("final_seed_sequences", {}),
        source_selection_attempt=normalized["selection"],
        upstream_attempt=normalized["validation"],
        errors=errors,
    )
    _audit_final_grid_job_files(
        root / "05_final_grid" / normalized["final_grid"],
        final_jobs=final_jobs,
        champion_rows=champion_rows,
        errors=errors,
    )

    summary_rows = _read_csv(paths["summary"], errors)
    if len(summary_rows) != EXPECTED_SCAN_COUNT:
        errors.append(
            f"screen summary row count is {len(summary_rows)}, "
            f"expected {EXPECTED_SCAN_COUNT}"
        )
    if {row.get("run_id", "") for row in summary_rows} != scan_run_ids:
        errors.append("screen summary run-id population differs from 00_grid")
    if any(row.get("status") != "completed" for row in summary_rows):
        errors.append("screen summary contains non-completed rows")
    if any(row.get("train_attempt_id") != normalized["train"] for row in summary_rows):
        errors.append("screen summary contains wrong train attempt")
    for row in summary_rows:
        run_id = row.get("run_id", "")
        expected_validation_dir = (
            root / "02_validation" / run_id / normalized["validation"]
        )
        expected_checkpoint = (
            root
            / "01_train"
            / run_id
            / normalized["train"]
            / "checkpoints"
        )
        if row.get("validation_attempt_id") != normalized["validation"]:
            errors.append(
                f"screen summary {run_id} validation attempt mismatch"
            )
        if row.get("validation_attempt_dir") != str(expected_validation_dir):
            errors.append(
                f"screen summary {run_id} validation directory mismatch"
            )
        if row.get("checkpoint_path") != str(expected_checkpoint):
            errors.append(
                f"screen summary {run_id} checkpoint path mismatch"
            )
    _audit_screen_science_status(
        summary_rows,
        required_tasks=evaluation_tasks.get("02_validation", ()),
        errors=errors,
    )
    _audit_collection_metric_reconciliation(
        root,
        rows=summary_rows,
        grid=grid,
        collection_report=collection,
        validation_attempt=normalized["validation"],
        errors=errors,
    )
    _audit_selection_replay(
        grid=grid,
        summary_rows=summary_rows,
        selection_report=selection,
        champion_rows=champion_rows,
        errors=errors,
    )
    failure_rows = _read_csv(paths["failures"], errors)
    if _read_csv_header(paths["failures"], errors) != _read_csv_header(
        paths["summary"],
        errors,
    ):
        errors.append("collection failures.csv header differs from summary.csv")
    if collection.get("n_failures") == 0 and failure_rows:
        errors.append("collection failures.csv is not empty")
    if len(failure_rows) != int(collection.get("n_failures", -1)):
        errors.append("collection failures.csv count differs from report")
    _audit_collection_sources(
        root,
        collection_attempt=normalized["collection"],
        validation_attempt=normalized["validation"],
        scan_run_ids=scan_run_ids,
        errors=errors,
    )
    final_rows = _read_csv(paths["final_run_index"], errors)
    if len(final_rows) != EXPECTED_FINAL_COUNT:
        errors.append(
            f"final run-index row count is {len(final_rows)}, "
            f"expected {EXPECTED_FINAL_COUNT}"
        )
    if {row.get("final_run_id", "") for row in final_rows} != final_run_ids:
        errors.append("final run-index population differs from 05_final_grid")
    if any(
        row.get("final_eval_attempt_id") != normalized["final_eval"]
        or row.get("train_status") != "checkpoint_selected"
        or row.get("eval_status") != "completed"
        or int(row.get("n_eval_tasks_success", "-1"))
        != len(evaluation_tasks.get("07_final_eval", ()))
        or int(row.get("n_eval_tasks_failed", "-1")) != 0
        for row in final_rows
    ):
        errors.append("final run-index contains wrong/incomplete evaluation rows")
    _audit_final_collection_manifest(
        paths["final_collect_manifest"],
        final_run_ids=final_run_ids,
        expected_study=expected_study,
        final_grid_attempt=normalized["final_grid"],
        final_collect_attempt=normalized["final_collect"],
        final_eval_attempt=normalized["final_eval"],
        major_axes=tuple(grid.get("major_axes") or ()),
        minor_axes=tuple(grid.get("minor_axes") or ()),
        errors=errors,
    )
    failure_rows = _read_csv(paths["final_failure_modes"], errors)
    if any(row.get("severity") == "failed" for row in failure_rows):
        errors.append("final failure_modes contains failed science rows")
    _audit_final_report(
        root,
        report_path=paths["report_json"],
        report_markdown=paths["report_markdown"],
        report_attempt=normalized["report"],
        final_collect_attempt=normalized["final_collect"],
        expected_study=expected_study,
        final_collect_manifest=paths["final_collect_manifest"],
        errors=errors,
    )

    if expected_study == "pair_stability_v4":
        errors.extend(audit_identity(root, attempts=normalized))
        errors.extend(
            f"unsafe root link: {path}" for path in validate_root_links(root)
        )
    return tuple(_deduplicate(errors))


def reference_evidence(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> dict[str, Any]:
    """Return summaries produced by the same fan-out and selector audits."""

    root, _root_study = _resolve_lineage_root(results_root)
    # Failed-comparison provenance may call this public summarizer without the
    # full completed-lineage audit.  Parse all structured acceptance evidence
    # strictly before any downstream reader can normalize it.
    validate_structured_tree(root)
    normalized = _normalize_attempts(attempts)
    grid_dir = root / "00_grid" / normalized["grid"]
    grid = _read_json_for_audit(grid_dir / "manifest.json", [])
    expected_study = str(grid.get("study") or _root_study)
    evaluation_tasks = {
        "02_validation": _configured_evaluation_tasks(
            grid_dir / "validation_config.yaml",
            suite="validation",
        ),
        "07_final_eval": _configured_evaluation_tasks(
            grid_dir / "validation_config.yaml",
            suite="final_eval",
        ),
    }
    _run_ids, _commits, fanout_errors, fanout_evidence = (
        audit_fanout_stages(
            root,
            attempt=normalized["grid"],
            expected_study=expected_study,
            evaluation_tasks=evaluation_tasks,
        )
    )
    if fanout_errors:
        raise ValueError(
            "cannot summarize unaudited fan-out evidence: "
            + "; ".join(fanout_errors)
        )
    summary_path = (
        root
        / "03_collect"
        / normalized["collection"]
        / "summary.csv"
    )
    summary_errors: list[str] = []
    summary_rows = _read_csv(summary_path, summary_errors)
    if summary_errors:
        raise ValueError(
            "cannot summarize collection evidence: "
            + "; ".join(summary_errors)
        )
    selection_dir = root / "04_select" / normalized["selection"]
    selection_report_path = selection_dir / "selection_report.json"
    champions_path = selection_dir / "champions.csv"
    selection_report = _read_json_for_audit(
        selection_report_path,
        summary_errors,
    )
    champion_rows = _read_csv(champions_path, summary_errors)
    if summary_errors:
        raise ValueError(
            "cannot summarize selection evidence: "
            + "; ".join(summary_errors)
        )
    replay = replay_selection(grid=grid, summary_rows=summary_rows)
    selection_source = (
        _REPO_ROOT / "experiments" / "toolkit" / "selection.py"
    )
    selection_toolkit_sha256 = hashlib.sha256(
        selection_source.read_bytes()
    ).hexdigest()
    contract = selection_contract(
        grid=grid,
        summary_rows=summary_rows,
        replay=replay,
        selection_report=selection_report,
        champion_rows=champion_rows,
        artifact_sha256={
            "grid_manifest": hashlib.sha256(
                (grid_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "summary_csv": hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest(),
            "selection_report": hashlib.sha256(
                selection_report_path.read_bytes()
            ).hexdigest(),
            "champions_csv": hashlib.sha256(
                champions_path.read_bytes()
            ).hexdigest(),
        },
        selection_toolkit_sha256=selection_toolkit_sha256,
    )
    return {
        **fanout_evidence,
        "selection": contract,
    }


def audit_identity(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[str, ...]:
    """Return candidate artifacts that retain wrong scientific identity."""

    errors: list[str] = []
    try:
        root = require_v4_root(results_root)
        normalized = _normalize_attempts(attempts, require_all=False)
    except (OSError, ValueError) as exc:
        return (str(exc),)

    manifest_specs = (
        ("grid", "00_grid", "manifest.json"),
        ("final_grid", "05_final_grid", "manifest.json"),
    )
    for key, stage, filename in manifest_specs:
        attempt = normalized.get(key)
        if attempt is None:
            continue
        path = root / stage / attempt / filename
        if not path.is_file():
            errors.append(f"missing identity artifact: {path.relative_to(root)}")
            continue
        try:
            payload = load_json(path)
        except (OSError, StrictDataError):
            errors.append(f"invalid identity artifact: {path.relative_to(root)}")
            continue
        if payload.get("study") != "pair_stability_v4":
            errors.append(
                f"{path.relative_to(root)} study is "
                f"{payload.get('study')!r}, expected 'pair_stability_v4'"
            )
        if payload.get("attempt_id") != attempt:
            errors.append(f"{path.relative_to(root)} attempt_id mismatch")

    grid_attempt = normalized.get("grid")
    if grid_attempt is not None:
        grid_dir = root / "00_grid" / grid_attempt
        for filename in ("grid.yaml", "train_config.yaml", "validation_config.yaml"):
            path = grid_dir / filename
            if not path.is_file():
                errors.append(f"missing v4 config snapshot: {path.relative_to(root)}")
                continue
            text = path.read_text()
            if "pair_stability_v3" in text:
                errors.append(
                    f"{path.relative_to(root)} retains pair_stability_v3 identity"
                )
    return tuple(_deduplicate(errors))


def _lineage_paths(
    root: Path,
    attempts: Mapping[str, str],
) -> dict[str, Path]:
    return {
        "grid_manifest": root / "00_grid" / attempts["grid"] / "manifest.json",
        "train_plan": root
        / "01_train"
        / "stage_plans"
        / attempts["train"]
        / "stage_manifest.json",
        "validation_plan": root
        / "02_validation"
        / "stage_plans"
        / attempts["validation"]
        / "stage_manifest.json",
        "collection_report": root
        / "03_collect"
        / attempts["collection"]
        / "collection_report.json",
        "collection_source_grid": root
        / "03_collect"
        / attempts["collection"]
        / "source_grid_attempt.json",
        "collection_lineage": root
        / "03_collect"
        / attempts["collection"]
        / "task_lineage.jsonl",
        "summary": root / "03_collect" / attempts["collection"] / "summary.csv",
        "failures": root
        / "03_collect"
        / attempts["collection"]
        / "failures.csv",
        "selection_report": root
        / "04_select"
        / attempts["selection"]
        / "selection_report.json",
        "champions": root
        / "04_select"
        / attempts["selection"]
        / "champions.csv",
        "selection_source_collection": root
        / "04_select"
        / attempts["selection"]
        / "source_collection_attempt.json",
        "selection_lineage": root
        / "04_select"
        / attempts["selection"]
        / "task_lineage.jsonl",
        "final_grid_manifest": root
        / "05_final_grid"
        / attempts["final_grid"]
        / "manifest.json",
        "final_grid_source_selection": root
        / "05_final_grid"
        / attempts["final_grid"]
        / "source_selection_attempt.json",
        "final_grid_source_champions": root
        / "05_final_grid"
        / attempts["final_grid"]
        / "source_champions.csv",
        "final_grid_lineage": root
        / "05_final_grid"
        / attempts["final_grid"]
        / "task_lineage.jsonl",
        "final_train_plan": root
        / "06_final_train"
        / "stage_plans"
        / attempts["final_train"]
        / "stage_manifest.json",
        "final_eval_plan": root
        / "07_final_eval"
        / "stage_plans"
        / attempts["final_eval"]
        / "stage_manifest.json",
        "final_run_index": root
        / "08_final_collect"
        / attempts["final_collect"]
        / "run_index.csv",
        "final_collect_manifest": root
        / "08_final_collect"
        / attempts["final_collect"]
        / "manifest.yaml",
        "final_failure_modes": root
        / "08_final_collect"
        / attempts["final_collect"]
        / "failure_modes.csv",
        "report_json": root
        / "09_final_report"
        / attempts["report"]
        / "final_report.json",
        "report_markdown": root
        / "09_final_report"
        / attempts["report"]
        / "report.md",
    }


def _normalize_attempts(
    attempts: Mapping[str, str],
    *,
    require_all: bool = True,
) -> dict[str, str]:
    required = {
        "grid",
        "train",
        "validation",
        "collection",
        "selection",
        "final_grid",
        "final_train",
        "final_eval",
        "final_collect",
        "report",
    }
    normalized = {
        str(key): validate_lineage_id(str(value))
        for key, value in attempts.items()
    }
    unknown = set(normalized) - required
    missing = required - set(normalized) if require_all else set()
    if unknown or missing:
        raise ValueError(
            f"attempt map mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return normalized


def _audit_collection_sources(
    root: Path,
    *,
    collection_attempt: str,
    validation_attempt: str,
    scan_run_ids: set[str],
    errors: list[str],
) -> None:
    path = (
        root
        / "03_collect"
        / collection_attempt
        / "source_validation_attempts.json"
    )
    try:
        rows = load_json(path)
    except (OSError, StrictDataError) as exc:
        errors.append(f"invalid source validation attempts {path}: {exc}")
        return
    if not isinstance(rows, list):
        errors.append("source validation attempts is not a list")
        return
    if len(rows) != EXPECTED_SCAN_COUNT:
        errors.append("source validation attempt row count is not 64")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("source validation attempt row is not an object")
            continue
        run_id = str(row.get("run_id") or "")
        if run_id in seen:
            errors.append(
                f"duplicate collection source-validation row: {run_id}"
            )
        seen.add(run_id)
        if set(row) != {
            "run_id",
            "validation_attempt_id",
            "validation_attempt_dir",
        }:
            errors.append(
                f"collection source-validation fields differ for {run_id}"
            )
        expected_dir = root / "02_validation" / run_id / validation_attempt
        if row.get("validation_attempt_id") != validation_attempt:
            errors.append(f"collection resolved unexpected validation attempt for {run_id}")
        if row.get("validation_attempt_dir") != str(expected_dir):
            errors.append(f"collection resolved unexpected validation directory for {run_id}")
    if seen != scan_run_ids:
        errors.append("collection source-validation population differs from 00_grid")


def _audit_final_collection_manifest(
    path: Path,
    *,
    final_run_ids: set[str],
    expected_study: str,
    final_grid_attempt: str,
    final_collect_attempt: str,
    final_eval_attempt: str,
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    errors: list[str],
) -> None:
    try:
        payload = load_yaml(path)
    except (OSError, StrictDataError) as exc:
        errors.append(f"invalid final-collect manifest {path}: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("final-collect manifest is not an object")
        return
    for key, expected in {
        "study": expected_study,
        "stage": "08_final_collect",
        "attempt_id": final_collect_attempt,
        "final_grid_attempt_id": final_grid_attempt,
    }.items():
        if payload.get(key) != expected:
            errors.append(
                f"final-collect {key}={payload.get(key)!r}, "
                f"expected {expected!r}"
            )
    if payload.get("n_final_eval_attempts") != EXPECTED_FINAL_COUNT:
        errors.append("final-collect manifest does not record eight evaluations")
    if payload.get("expected_final_replicates") != 1:
        errors.append("final-collect manifest expected_final_replicates is not 1")
    if payload.get("major_axes") != list(major_axes):
        errors.append("final-collect manifest major_axes mismatch")
    if payload.get("minor_axes") != list(minor_axes):
        errors.append("final-collect manifest minor_axes mismatch")
    if payload.get("axis_columns") != [*major_axes, *minor_axes]:
        errors.append("final-collect manifest axis_columns mismatch")
    if payload.get("source_stages") != {
        "final_grid": "05_final_grid",
        "final_train": "06_final_train",
        "final_eval": "07_final_eval",
    }:
        errors.append("final-collect manifest source_stages mismatch")
    if {
        "basis",
        "update_normalization",
        "feature_normalization",
    }.issubset(set(major_axes)):
        expected_report_axes = ("basis_update", "feature_normalization")
    else:
        expected_report_axes = (
            major_axes[0] if major_axes else None,
            (
                major_axes[1]
                if len(major_axes) > 1
                else (minor_axes[0] if minor_axes else None)
            ),
        )
    if payload.get("report_row_key") != expected_report_axes[0]:
        errors.append("final-collect manifest report_row_key mismatch")
    if payload.get("report_col_key") != expected_report_axes[1]:
        errors.append("final-collect manifest report_col_key mismatch")
    if payload.get("final_eval_attempt_id") != final_eval_attempt:
        errors.append("final-collect manifest final_eval_attempt_id mismatch")
    attempts = payload.get("final_eval_attempts")
    if attempts != {run_id: final_eval_attempt for run_id in final_run_ids}:
        errors.append("final-collect manifest per-run evaluation attempts mismatch")
    attempt_ids = payload.get("final_eval_attempt_ids")
    if attempt_ids != [final_eval_attempt]:
        errors.append("final-collect manifest resolved attempt-id list mismatch")
    tables = payload.get("tables")
    if not isinstance(tables, dict) or not tables:
        errors.append("final-collect manifest tables are missing")
    else:
        if set(tables) != EXPECTED_FINAL_COLLECT_TABLES:
            errors.append("final-collect manifest table population mismatch")
        for name, expected_count in tables.items():
            table_path = path.parent / str(name)
            rows = _read_csv(table_path, errors)
            if len(rows) != int(expected_count):
                errors.append(
                    f"final-collect table {name} row count differs from manifest"
                )


def _audit_source_record(
    path: Path,
    *,
    expected: Mapping[str, str],
    label: str,
    errors: list[str],
) -> None:
    payload = _read_json_for_audit(path, errors)
    if not payload:
        errors.append(f"{label} record is empty")
        return
    if set(payload) != set(expected):
        errors.append(f"{label} fields mismatch")
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(
                f"{label} {key}={payload.get(key)!r}, expected {value!r}"
            )


def _audit_task_lineage(
    path: Path,
    *,
    expected_row_ids: set[str] | None = None,
    expected_count: int | None = None,
    required_task_kinds: Sequence[str] = (),
    require_nonempty_tasks: bool = False,
    errors: list[str],
) -> None:
    rows = _read_jsonl(path, errors)
    row_ids: list[str] = []
    for index, row in enumerate(rows):
        if set(row) != {"row_id", "task_ids"}:
            errors.append(f"task-lineage row {index} fields mismatch: {path}")
            continue
        row_id = str(row.get("row_id") or "")
        task_ids = row.get("task_ids")
        row_ids.append(row_id)
        if not isinstance(task_ids, dict):
            errors.append(f"task-lineage row {row_id} task_ids is not an object")
            continue
        if require_nonempty_tasks and not task_ids:
            errors.append(f"task-lineage row {row_id} has no upstream tasks")
        for kind in required_task_kinds:
            value = task_ids.get(kind)
            if not isinstance(value, str) or not value:
                errors.append(
                    f"task-lineage row {row_id} lacks {kind} task identity"
                )
    if len(set(row_ids)) != len(row_ids):
        errors.append(f"task-lineage contains duplicate row ids: {path}")
    if expected_row_ids is not None and set(row_ids) != expected_row_ids:
        errors.append(f"task-lineage population mismatch: {path}")
    if expected_count is not None and len(rows) != expected_count:
        errors.append(
            f"task-lineage row count is {len(rows)}, expected {expected_count}"
        )


def _audit_collection_task_lineage_values(
    path: Path,
    *,
    attempt: str,
    scan_run_ids: set[str],
    errors: list[str],
) -> None:
    for row in _read_jsonl(path, errors):
        run_id = str(row.get("row_id") or "")
        if run_id not in scan_run_ids:
            continue
        expected = {
            "validation": f"02_validation:{run_id}:{attempt}",
            "train": f"01_train:{run_id}:{attempt}",
        }
        if row.get("task_ids") != expected:
            errors.append(
                f"collection task-lineage values differ for {run_id}"
            )


def _audit_selection_rows(
    report: Mapping[str, Any],
    *,
    champion_rows: Sequence[Mapping[str, str]],
    lineage_path: Path,
    major_axes: Sequence[str],
    scan_run_ids: set[str],
    attempt: str,
    errors: list[str],
) -> None:
    report_rows = report.get("champions")
    if not isinstance(report_rows, list) or len(report_rows) != len(
        champion_rows
    ):
        errors.append("selection report champion population differs from CSV")
        return
    unique_keys: set[tuple[str, ...]] = set()
    for index, (csv_row, report_row) in enumerate(
        zip(champion_rows, report_rows, strict=True)
    ):
        if not isinstance(report_row, dict):
            errors.append(f"selection report champion {index} is not an object")
            continue
        key = (
            str(csv_row.get("winner_kind") or ""),
            *(str(csv_row.get(axis) or "") for axis in major_axes),
        )
        if key in unique_keys:
            errors.append("selection champions contain duplicate group/winner")
        unique_keys.add(key)
        for column, value in csv_row.items():
            if _csv_scalar(report_row.get(column)) != value:
                errors.append(
                    f"selection champion {index} column {column} differs"
                )
    lineage = _read_jsonl(lineage_path, errors)
    by_row = {str(row.get("row_id") or ""): row for row in lineage}
    if len(by_row) != len(lineage):
        errors.append("selection lineage contains duplicate row ids")
    for row in champion_rows:
        winner = str(row.get("winner_kind") or "")
        row_id = winner + ":" + "|".join(
            f"{axis}={row.get(axis, '')}" for axis in major_axes
        )
        run_ids = [
            value for value in str(row.get("run_ids") or "").split(";") if value
        ]
        if not run_ids or any(run_id not in scan_run_ids for run_id in run_ids):
            errors.append(f"selection champion {row_id} has unknown run ids")
            continue
        expected_tasks = {
            "validation": [
                f"02_validation:{run_id}:{attempt}" for run_id in run_ids
            ],
            "train": [
                f"01_train:{run_id}:{attempt}" for run_id in run_ids
            ],
        }
        lineage_row = by_row.get(row_id)
        if lineage_row is None or lineage_row.get("task_ids") != expected_tasks:
            errors.append(f"selection lineage differs for {row_id}")


def _csv_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _audit_final_grid_job_files(
    final_grid_dir: Path,
    *,
    final_jobs: Sequence[Mapping[str, str]],
    champion_rows: Sequence[Mapping[str, str]],
    errors: list[str],
) -> None:
    final_run_ids = {
        str(row.get("final_run_id") or "")
        for row in final_jobs
        if row.get("final_run_id")
    }
    jobs_dir = final_grid_dir / "jobs"
    actual = {
        path.name
        for path in jobs_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    } if jobs_dir.is_dir() and not jobs_dir.is_symlink() else set()
    expected = {f"{run_id}.json" for run_id in final_run_ids}
    if actual != expected:
        errors.append("final-grid per-run job file population mismatch")
        return
    rows_by_id = {
        str(row.get("final_run_id") or ""): row for row in final_jobs
    }
    for run_id in sorted(final_run_ids):
        payload = _read_json_for_audit(jobs_dir / f"{run_id}.json", errors)
        if payload.get("final_run_id") != run_id:
            errors.append(f"final-grid job payload run id mismatch: {run_id}")
            continue
        csv_row = rows_by_id[run_id]
        for column, value in csv_row.items():
            if _csv_scalar(payload.get(column)) != value:
                errors.append(
                    f"final-grid job payload {run_id} column {column} differs"
                )
        try:
            champion_index = int(csv_row["source_champion_row_index"])
            champion = champion_rows[champion_index]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        source_champion = payload.get("source_champion")
        if not isinstance(source_champion, dict):
            errors.append(
                f"final-grid job payload {run_id} source_champion is missing"
            )
        elif {
            str(key): _csv_scalar(value)
            for key, value in source_champion.items()
        } != dict(champion):
            errors.append(
                f"final-grid job payload {run_id} source_champion differs"
            )


def _audit_final_grid_rows(
    final_jobs: Sequence[Mapping[str, str]],
    *,
    champion_rows: Sequence[Mapping[str, str]],
    lineage_path: Path,
    final_seed_sequences: object,
    source_selection_attempt: str,
    upstream_attempt: str,
    errors: list[str],
) -> None:
    """Audit exact champion expansion, seed policy, and inherited task lineage."""

    if len(final_jobs) != EXPECTED_FINAL_COUNT:
        errors.append("final-grid job row count is not 8")
    if not isinstance(final_seed_sequences, dict):
        errors.append("final-grid seed sequence contract is not an object")
        sequences: Mapping[str, object] = {}
    else:
        sequences = final_seed_sequences
    lineage_rows = _read_jsonl(lineage_path, errors)
    lineage_by_id = {
        str(row.get("row_id") or ""): row for row in lineage_rows
    }
    champion_indices: set[int] = set()
    final_run_ids: set[str] = set()
    for row_number, row in enumerate(final_jobs):
        run_id = str(row.get("final_run_id") or "")
        if not run_id or run_id in final_run_ids:
            errors.append(f"final-grid row {row_number} has duplicate/empty run id")
        final_run_ids.add(run_id)
        try:
            champion_index = int(row.get("source_champion_row_index", ""))
            champion = champion_rows[champion_index]
        except (IndexError, TypeError, ValueError):
            errors.append(
                f"final-grid row {run_id or row_number} has invalid champion index"
            )
            continue
        if champion_index < 0:
            errors.append(f"final-grid row {run_id} has negative champion index")
            continue
        champion_indices.add(champion_index)
        try:
            replicate_index = int(row.get("replicate_index", ""))
        except ValueError:
            errors.append(f"final-grid row {run_id} replicate index is invalid")
            continue
        if replicate_index != 0:
            errors.append(f"final-grid row {run_id} replicate index is not 0")
        expected_id = f"champion-{champion_index:04d}"
        if row.get("source_champion_id") != expected_id:
            errors.append(f"final-grid row {run_id} source champion id differs")
        if row.get("source_selection_attempt_id") != source_selection_attempt:
            errors.append(
                f"final-grid row {run_id} source selection attempt differs"
            )
        direct_mappings = {
            "source_scan_run_id": champion.get("config_id", ""),
            "source_scan_run_ids": champion.get("run_ids", ""),
            "source_scan_seeds": champion.get("seeds", ""),
            "winner_kind": champion.get("winner_kind", ""),
            "metric": champion.get("metric", ""),
            "metric_value": champion.get("metric_value", ""),
        }
        for column, expected in direct_mappings.items():
            if row.get(column, "") != expected:
                errors.append(
                    f"final-grid row {run_id} {column} differs from champion"
                )
        for column, expected in champion.items():
            if column in row and row[column] != expected:
                errors.append(
                    f"final-grid row {run_id} axis/value {column} differs"
                )
        config_id = str(champion.get("config_id") or "")
        winner_kind = str(champion.get("winner_kind") or "")
        if config_id:
            expected_run_id = (
                f"{config_id}_winner-{winner_kind}_rep-{replicate_index}"
            )
            if run_id != expected_run_id:
                errors.append(
                    f"final-grid row {run_id} does not use canonical final run id"
                )
        for seed_name, raw_spec in sequences.items():
            if not isinstance(raw_spec, dict):
                errors.append(f"final seed sequence {seed_name} is invalid")
                continue
            try:
                expected_seed = int(raw_spec["start"]) + (
                    replicate_index * int(raw_spec["step"])
                )
                observed_seed = int(row.get(str(seed_name), ""))
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"final-grid row {run_id} seed {seed_name} is invalid"
                )
                continue
            if observed_seed != expected_seed:
                errors.append(
                    f"final-grid row {run_id} seed {seed_name} differs"
                )
        source_run_ids = [
            value
            for value in str(champion.get("run_ids") or "").split(";")
            if value
        ]
        expected_tasks = {
            "validation": [
                f"02_validation:{source_id}:{upstream_attempt}"
                for source_id in source_run_ids
            ],
            "train": [
                f"01_train:{source_id}:{upstream_attempt}"
                for source_id in source_run_ids
            ],
        }
        lineage_row = lineage_by_id.get(run_id)
        if lineage_row is None or lineage_row.get("task_ids") != expected_tasks:
            errors.append(f"final-grid lineage differs for {run_id}")
    expected_indices = set(range(len(champion_rows)))
    if champion_indices != expected_indices:
        errors.append("final-grid champion index population mismatch")


def _audit_final_report(
    root: Path,
    *,
    report_path: Path,
    report_markdown: Path,
    report_attempt: str,
    final_collect_attempt: str,
    expected_study: str,
    final_collect_manifest: Path,
    errors: list[str],
) -> None:
    report = _read_json_for_audit(report_path, errors)
    if not report:
        errors.append("final report record is empty")
        return
    for key, expected in {
        "study": expected_study,
        "stage": "09_final_report",
        "attempt_id": report_attempt,
        "final_collect_attempt_id": final_collect_attempt,
        "final_collect_dir": str(
            root / "08_final_collect" / final_collect_attempt
        ),
    }.items():
        if report.get(key) != expected:
            errors.append(
                f"final report {key}={report.get(key)!r}, expected {expected!r}"
            )
    tables = report.get("tables")
    if not isinstance(tables, dict) or not tables:
        errors.append("final report table contract is empty")
    else:
        for name, expected_count in tables.items():
            path = report_path.parent / "tables" / str(name)
            rows = _read_csv(path, errors)
            if len(rows) != int(expected_count):
                errors.append(
                    f"final report table {name} row count differs from report"
                )
    try:
        collect_manifest = load_yaml(final_collect_manifest)
    except (OSError, StrictDataError) as exc:
        errors.append(f"cannot read final-collect report contract: {exc}")
        collect_manifest = {}
    if isinstance(collect_manifest, dict):
        expected_axes = {
            "row": collect_manifest.get("report_row_key"),
            "column": collect_manifest.get("report_col_key"),
        }
        if report.get("report_axes") != expected_axes:
            errors.append("final report axes differ from final-collect manifest")
        collect_tables = collect_manifest.get("tables")
        if not isinstance(collect_tables, dict):
            errors.append("final-collect table contract is unavailable to report")
        elif isinstance(tables, dict):
            for name, expected_count in collect_tables.items():
                if tables.get(name) != expected_count:
                    errors.append(
                        f"final report omits/changes final-collect table {name}"
                    )
                    continue
                source = final_collect_manifest.parent / str(name)
                copied = report_path.parent / "tables" / str(name)
                try:
                    if source.read_bytes() != copied.read_bytes():
                        errors.append(
                            f"final report table {name} is not an exact copy"
                        )
                except OSError as exc:
                    errors.append(
                        f"cannot compare final report table {name}: {exc}"
                    )
    figures = report.get("figures")
    if not isinstance(figures, list) or not figures:
        errors.append("final report figure contract is empty")
    else:
        for name in figures:
            relative = Path(str(name))
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"unsafe final report figure path: {name}")
                continue
            path = report_path.parent / "figures" / relative
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                errors.append(f"final report figure is missing/empty: {name}")
    if (
        not report_markdown.is_file()
        or report_markdown.is_symlink()
        or not report_markdown.read_text().strip()
    ):
        errors.append("final report Markdown is missing or empty")


def _configured_evaluation_tasks(
    config_path: Path,
    *,
    suite: str,
) -> tuple[str, ...]:
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("validation config is not an object")
    suites = config.get("evaluation_suites")
    specifications = config.get("evaluation_tasks")
    if not isinstance(suites, dict) or not isinstance(specifications, dict):
        raise ValueError("validation config evaluation contracts are missing")
    suite_spec = suites.get(suite)
    if not isinstance(suite_spec, dict) or not isinstance(
        suite_spec.get("tasks"),
        list,
    ):
        raise ValueError(f"evaluation suite {suite!r} is invalid")
    names: list[str] = []
    for reference in suite_spec["tasks"]:
        text = str(reference)
        prefix = "${evaluation_tasks."
        if not text.startswith(prefix) or not text.endswith("}"):
            raise ValueError(f"evaluation task reference is not explicit: {text}")
        key = text[len(prefix) : -1]
        task = specifications.get(key)
        if not isinstance(task, dict):
            raise ValueError(f"evaluation task {key!r} is missing")
        name = str(task.get("name") or "")
        if not name or "/" in name:
            raise ValueError(f"evaluation task {key!r} has invalid name")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"evaluation suite {suite!r} has duplicate task names")
    return tuple(names)


def _resolve_lineage_root(path: Path) -> tuple[Path, str]:
    requested = Path(path)
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("lineage audit root must be absolute without traversal")
    if requested.is_symlink():
        raise ValueError("lineage audit root may not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("lineage audit root must be a directory")
    sentinel = root / ROOT_SENTINEL
    if sentinel.exists() or sentinel.is_symlink():
        return (
            require_v4_root(root, purpose=PURPOSE_EXPERIMENT),
            "pair_stability_v4",
        )
    live_v3 = V3_STUDY_DIR / "results"
    if live_v3.exists():
        canonical_live_v3 = live_v3.resolve()
        if root == canonical_live_v3 or canonical_live_v3 in root.parents:
            raise ValueError(
                "live pair_stability_v3 results cannot become a reference"
            )
    return root, "pair_stability_v3"


def _read_json_for_audit(
    path: Path,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = load_json(path)
    except (OSError, StrictDataError):
        errors.append(f"invalid JSON: {path}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing JSONL: {path}")
        return []
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        for line_number, value in enumerate(iter_jsonl(path), start=1):
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (OSError, StrictDataError, ValueError) as exc:
        errors.append(f"invalid JSONL {path}:{line_number}: {exc}")
    return rows


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing CSV: {path}")
        return []
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or not header or any(not name for name in header):
                errors.append(f"invalid CSV header: {path}")
                return []
            if len(set(header)) != len(header):
                errors.append(f"duplicate CSV header fields: {path}")
                return []
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    errors.append(
                        f"CSV row width mismatch {path}:{line_number}"
                    )
                    continue
                rows.append(dict(zip(header, values, strict=True)))
            return rows
    except (OSError, csv.Error) as exc:
        errors.append(f"invalid CSV {path}: {exc}")
        return []


def _read_csv_header(path: Path, errors: list[str]) -> tuple[str, ...]:
    if not path.is_file():
        errors.append(f"missing CSV: {path}")
        return ()
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, csv.Error) as exc:
        errors.append(f"invalid CSV {path}: {exc}")
        return ()
    if header is None or not header or any(not name for name in header):
        errors.append(f"invalid CSV header: {path}")
        return ()
    if len(set(header)) != len(header):
        errors.append(f"duplicate CSV header fields: {path}")
    return tuple(header)


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))
