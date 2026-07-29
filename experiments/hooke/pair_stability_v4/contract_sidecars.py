"""Project V4-1A contract sidecars from a completed V4-0 lineage.

The toolkit package owns generic record codecs and bundle publication.  This
module owns the pair-stability-specific reading of already completed V4 stage
artifacts.  It is deliberately a post-route adapter: it neither schedules
work nor resolves mutable ``latest`` pointers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import audit  # noqa: E402
import control_audit  # noqa: E402
from experiments.toolkit import ExecutionRecord, StagePlan  # noqa: E402
from experiments.toolkit.contracts import (  # noqa: E402
    ContractBundleV1,
    ContractError,
    ExecutionProfileV1,
    MetricKeyV1,
    ProducerAttemptV1,
    ProducerV1,
    RunV1,
    SeedAssignmentV1,
    SourceDescriptorV1,
    StageResultV1,
    TrialV1,
    bundle_manifest_sha256,
    canonical_sha256,
    publish_bundle,
    read_bundle,
)
from roots import PURPOSE_EXPERIMENT, require_beneath_root, require_v4_root  # noqa: E402
from routes import ROLE_TO_STAGE, load_routes  # noqa: E402
from strict_data import StrictDataError, iter_jsonl, load_json  # noqa: E402


CONTRACT_STUDY = "pair_stability_v4"
CONTRACT_SCOPE_SCHEMA = "pair-stability-v4/contracts/v1"
VERIFIER_VERSION = "pair-stability-v4/contract-sidecars-verifier/v1"
VERIFICATION_RECEIPT_SCHEMA_VERSION = "pair-stability-v4/contract-verification/v1"
_ATTEMPT_KEYS = (
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
)
_TRAINING_ROLES = {
    "screen_train": ("01_train", "scan"),
    "confirm_train": ("06_final_train", "confirm"),
}
_FANOUT_ROLES = frozenset(
    {"screen_train", "screen_eval", "confirm_train", "confirm_eval"}
)
_DIRECT_STAGE_MARKERS = {
    "screen_plan": ("00_grid", "manifest.json"),
    "screen_collect": ("03_collect", "collection_report.json"),
    "select": ("04_select", "selection_report.json"),
    "confirm_plan": ("05_final_grid", "manifest.json"),
    "confirm_collect": ("08_final_collect", "manifest.yaml"),
    "report": ("09_final_report", "final_report.json"),
}


@dataclass(frozen=True)
class _FanoutEvidence:
    """One validated stage-plan population used by the adapter only."""

    role: str
    stage: str
    plan: StagePlan
    records_by_task_id: Mapping[str, ExecutionRecord]
    plan_source_keys: tuple[str, ...]
    result_source_keys: Mapping[str, tuple[str, ...]]
    terminal_population_sha256: str
    execution_profile_id: str


class _SourceLedger:
    """Allow-list source artifacts before any row may cite them."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._by_relative_path: dict[str, SourceDescriptorV1] = {}

    def add(
        self,
        path: Path,
        *,
        logical_role: str,
        artifact_kind: str,
        schema: str,
    ) -> str:
        candidate = _regular_beneath_root(path, self._root)
        relative = candidate.relative_to(self._root).as_posix()
        descriptor = SourceDescriptorV1(
            source_key=(
                "source-"
                + canonical_sha256(
                    {
                        "path": relative,
                        "logical_role": logical_role,
                        "artifact_kind": artifact_kind,
                        "schema": schema,
                    }
                )[:32]
            ),
            root_relative_path=relative,
            logical_role=logical_role,
            artifact_kind=artifact_kind,
            schema=schema,
            sha256=_sha256_file(candidate),
        )
        existing = self._by_relative_path.get(relative)
        if existing is not None and existing != descriptor:
            raise ValueError(
                "one source artifact was assigned incompatible contract metadata: "
                f"{relative}"
            )
        self._by_relative_path[relative] = descriptor
        return descriptor.source_key

    def descriptors(self) -> tuple[SourceDescriptorV1, ...]:
        return tuple(
            sorted(self._by_relative_path.values(), key=lambda item: item.source_key)
        )


def contract_bundle_directory(root: Path, *, lineage_id: str) -> Path:
    """Return the create-only V4-only bundle location for one lineage."""

    root = _root(root, lineage_id)
    return require_beneath_root(root / "_v4" / "contracts" / lineage_id, root)


def contract_verification_receipt_path(root: Path, *, lineage_id: str) -> Path:
    """Return the create-only V4-only verification receipt location."""

    root = _root(root, lineage_id)
    return require_beneath_root(
        root / "_v4" / "contract_verifications" / f"{lineage_id}.json",
        root,
    )


def finalize_contract_sidecars(root: Path, *, lineage_id: str) -> Path:
    """Publish and freshly verify one completed lineage's contract sidecars.

    This function is intentionally only callable before ``controller-result``.
    The shell controller records any raised failure as a named finalization
    error, then writes its immutable terminal result last.
    """

    root = _root(root, lineage_id)
    result_path = root / "_v4" / "stack" / lineage_id / "controller-result.json"
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("V4-1A sidecars must finalize before controller-result.json")
    _require_preclose_and_lineage(root, lineage_id)
    bundle = project_contract_bundle(root, lineage_id=lineage_id)
    destination = contract_bundle_directory(root, lineage_id=lineage_id)
    publish_bundle(destination, bundle)
    fresh = _fresh_verify(root, lineage_id=lineage_id)
    receipt = contract_verification_receipt_path(root, lineage_id=lineage_id)
    _write_new_json(
        receipt,
        {
            "schema_version": VERIFICATION_RECEIPT_SCHEMA_VERSION,
            "lineage_id": lineage_id,
            "bundle_scope_id": bundle.bundle_scope_id,
            "manifest_sha256": bundle_manifest_sha256(destination),
            "preclose_control_sha256": fresh["preclose_control_sha256"],
            "verifier_version": VERIFIER_VERSION,
            "status": "verified",
            "verified_at": _timestamp(),
        },
    )
    return receipt


def verify_contract_sidecars(
    root: Path,
    *,
    lineage_id: str,
    require_receipt: bool = True,
) -> dict[str, str]:
    """Re-open bundle, re-hash its sources, and validate current provenance."""

    root = _root(root, lineage_id)
    _require_preclose_and_lineage(root, lineage_id)
    destination = contract_bundle_directory(root, lineage_id=lineage_id)
    bundle = read_bundle(destination, source_root=root)
    preclose = control_audit.preclose_control_provenance(
        root,
        attempts=_attempts(lineage_id),
    )
    result = {
        "bundle_scope_id": bundle.bundle_scope_id,
        "manifest_sha256": bundle_manifest_sha256(destination),
        "preclose_control_sha256": _required_text(
            preclose.get("verification_sha256"),
            "pre-close control verification digest",
        ),
        "verifier_version": VERIFIER_VERSION,
    }
    if require_receipt:
        _validate_verification_receipt(
            contract_verification_receipt_path(root, lineage_id=lineage_id),
            lineage_id=lineage_id,
            result=result,
        )
    return result


def project_contract_bundle(root: Path, *, lineage_id: str) -> ContractBundleV1:
    """Build one closed semantic bundle from explicit completed-lineage facts."""

    root = _root(root, lineage_id)
    _require_preclose_and_lineage(root, lineage_id)
    ledger = _SourceLedger(root)

    grid_path = root / "00_grid" / lineage_id / "manifest.json"
    grid_source = ledger.add(
        grid_path,
        logical_role="screen_plan",
        artifact_kind="grid-manifest",
        schema="pair-stability-grid/v1",
    )
    grid = _json_object(grid_path, "grid manifest")
    _require_equal(grid.get("study"), CONTRACT_STUDY, "grid study")
    _require_equal(grid.get("attempt_id"), lineage_id, "grid attempt")
    scope = "scope-" + canonical_sha256(
        {
            "schema": CONTRACT_SCOPE_SCHEMA,
            "study": CONTRACT_STUDY,
            "lineage_id": lineage_id,
            "grid_manifest_sha256": _sha256_file(grid_path),
        }
    )
    trials, scan_seeds, scan_runs = _project_scan_graph(
        grid,
        bundle_scope_id=scope,
        grid_source=grid_source,
    )
    final_trials, confirm_seeds, confirm_runs = _project_confirm_graph(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        grid_source=grid_source,
        grid=grid,
        trials_by_key={trial.trial_key: trial for trial in trials},
        ledger=ledger,
    )
    if final_trials:
        raise AssertionError("confirmation projection must reuse scan TrialV1 records")

    controller_profile_id, profile_candidates = _controller_profile_candidate(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        ledger=ledger,
    )
    fanout = _project_fanout_evidence(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        ledger=ledger,
        profile_candidates=profile_candidates,
    )
    producers, producer_attempts = _project_training_producers(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        scan_runs=scan_runs,
        confirm_runs=confirm_runs,
        fanout=fanout,
        ledger=ledger,
    )
    profiles = _materialize_profiles(scope, profile_candidates)
    stages = _project_stage_results(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        controller_profile_id=controller_profile_id,
        fanout=fanout,
        ledger=ledger,
    )
    metrics = _project_metric_keys(
        root,
        lineage_id=lineage_id,
        bundle_scope_id=scope,
        stages_by_role={stage.logical_role: stage for stage in stages},
        fanout=fanout,
        ledger=ledger,
    )
    expected_roles = set(ROLE_TO_STAGE)
    actual_roles = {stage.logical_role for stage in stages}
    if actual_roles != expected_roles:
        raise ValueError(
            "contract stage population mismatch; "
            f"missing={sorted(expected_roles - actual_roles)}, "
            f"extra={sorted(actual_roles - expected_roles)}"
        )
    return ContractBundleV1(
        study=CONTRACT_STUDY,
        bundle_scope_id=scope,
        sources=ledger.descriptors(),
        trials=tuple(sorted(trials, key=lambda item: item.id)),
        seed_assignments=tuple(
            sorted((*scan_seeds, *confirm_seeds), key=lambda item: item.id)
        ),
        runs=tuple(sorted((*scan_runs, *confirm_runs), key=lambda item: item.id)),
        producers=tuple(sorted(producers, key=lambda item: item.id)),
        producer_attempts=tuple(sorted(producer_attempts, key=lambda item: item.id)),
        execution_profiles=tuple(sorted(profiles, key=lambda item: item.id)),
        metric_keys=tuple(sorted(metrics, key=lambda item: item.id)),
        stage_results=tuple(sorted(stages, key=lambda item: item.id)),
    )


def _project_scan_graph(
    grid: Mapping[str, object],
    *,
    bundle_scope_id: str,
    grid_source: str,
) -> tuple[tuple[TrialV1, ...], tuple[SeedAssignmentV1, ...], tuple[RunV1, ...]]:
    jobs = _object_rows(grid.get("jobs"), "grid jobs")
    major_axes = _text_sequence(grid.get("major_axes"), "grid major_axes")
    minor_axes = _text_sequence(grid.get("minor_axes"), "grid minor_axes")
    axes = (*major_axes, *minor_axes)
    if len(axes) != len(set(axes)):
        raise ValueError("grid configuration axes are not unique")
    trials_by_key: dict[str, TrialV1] = {}
    seeds_by_id: dict[str, SeedAssignmentV1] = {}
    runs: list[RunV1] = []
    run_keys: set[str] = set()
    for index, job in enumerate(jobs):
        trial_key = _required_identifier(job.get("config_id"), f"grid job {index} config_id")
        run_key = _required_identifier(job.get("run_id"), f"grid job {index} run_id")
        if run_key in run_keys:
            raise ValueError(f"grid contains duplicate run_id: {run_key}")
        run_keys.add(run_key)
        choices = _object(job.get("choices"), f"grid job {run_key} choices")
        blinded_choices = {
            axis: _required_mapping_value(choices, axis, f"grid job {run_key} choices")
            for axis in axes
        }
        trial = TrialV1(
            bundle_scope_id=bundle_scope_id,
            trial_key=trial_key,
            blinded_choices=blinded_choices,
            source_keys=(grid_source,),
        )
        existing = trials_by_key.get(trial_key)
        if existing is not None and existing.id != trial.id:
            raise ValueError(f"grid config_id has inconsistent blinded choices: {trial_key}")
        trials_by_key[trial_key] = trial
        seed = SeedAssignmentV1(
            bundle_scope_id=bundle_scope_id,
            assignment_kind="scan",
            values=_object(job.get("seed_values"), f"grid job {run_key} seed_values"),
            source_keys=(grid_source,),
        )
        seeds_by_id[seed.id] = seed
        runs.append(
            RunV1(
                bundle_scope_id=bundle_scope_id,
                trial_id=trial.id,
                seed_assignment_id=seed.id,
                lane="scan",
                run_key=run_key,
                source_champion_key=None,
                source_keys=(grid_source,),
            )
        )
    return (
        tuple(sorted(trials_by_key.values(), key=lambda item: item.id)),
        tuple(sorted(seeds_by_id.values(), key=lambda item: item.id)),
        tuple(sorted(runs, key=lambda item: item.id)),
    )


def _project_confirm_graph(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    grid_source: str,
    grid: Mapping[str, object],
    trials_by_key: Mapping[str, TrialV1],
    ledger: _SourceLedger,
) -> tuple[tuple[TrialV1, ...], tuple[SeedAssignmentV1, ...], tuple[RunV1, ...]]:
    final_root = root / "05_final_grid" / lineage_id
    final_manifest_path = final_root / "manifest.json"
    final_manifest_source = ledger.add(
        final_manifest_path,
        logical_role="confirm_plan",
        artifact_kind="final-grid-manifest",
        schema="pair-stability-final-grid/v1",
    )
    final_manifest = _json_object(final_manifest_path, "final grid manifest")
    _require_equal(final_manifest.get("study"), CONTRACT_STUDY, "final grid study")
    _require_equal(final_manifest.get("attempt_id"), lineage_id, "final grid attempt")
    final_csv_path = final_root / "final_jobs.csv"
    final_csv_source = ledger.add(
        final_csv_path,
        logical_role="confirm_plan",
        artifact_kind="final-grid-jobs",
        schema="pair-stability-final-jobs/csv-v1",
    )
    final_job_ids = _final_job_ids(final_csv_path)
    expected_n_jobs = _positive_int(final_manifest.get("n_jobs"), "final grid n_jobs")
    if len(final_job_ids) != expected_n_jobs:
        raise ValueError("final job CSV population differs from final-grid manifest")
    axes = (*_text_sequence(grid.get("major_axes"), "grid major_axes"), *_text_sequence(grid.get("minor_axes"), "grid minor_axes"))
    seeds_by_id: dict[str, SeedAssignmentV1] = {}
    runs: list[RunV1] = []
    seen_runs: set[str] = set()
    for run_key in final_job_ids:
        job_path = final_root / "jobs" / f"{run_key}.json"
        job_source = ledger.add(
            job_path,
            logical_role="confirm_plan",
            artifact_kind="final-grid-job",
            schema="pair-stability-final-job/v1",
        )
        job = _json_object(job_path, f"final job {run_key}")
        _require_equal(job.get("final_run_id"), run_key, "final job run_id")
        trial_key = _required_identifier(
            job.get("source_scan_run_id"),
            f"final job {run_key} source_scan_run_id",
        )
        trial = trials_by_key.get(trial_key)
        if trial is None:
            raise ValueError(
                "final source_scan_run_id must name an existing scan config_id, "
                f"not a scan RunV1: {trial_key}"
            )
        champion = _object(job.get("source_champion"), f"final job {run_key} source_champion")
        for axis, value in trial.blinded_choices.items():
            # Selection provenance reaches final planning through a CSV row,
            # so its explicit scalar spelling is authoritative here.  The
            # completed-lineage audit separately proves that this champion
            # relation is the exact grid-derived one.
            if champion.get(axis) != _csv_provenance_scalar(value):
                raise ValueError(
                    f"final job {run_key} champion {axis} does not match source config"
                )
        seed_values: dict[str, object] = {
            "replicate_index": _required_mapping_value(
                job, "replicate_index", f"final job {run_key}"
            )
        }
        for key, value in job.items():
            if key.endswith("_seed"):
                seed_values[key] = value
        if len(seed_values) == 1:
            raise ValueError(f"final job {run_key} has no named final seed values")
        seed = SeedAssignmentV1(
            bundle_scope_id=bundle_scope_id,
            assignment_kind="confirm",
            values=seed_values,
            source_keys=tuple(sorted((final_csv_source, job_source))),
        )
        seeds_by_id[seed.id] = seed
        if run_key in seen_runs:
            raise ValueError(f"final grid contains duplicate final_run_id: {run_key}")
        seen_runs.add(run_key)
        runs.append(
            RunV1(
                bundle_scope_id=bundle_scope_id,
                trial_id=trial.id,
                seed_assignment_id=seed.id,
                lane="confirm",
                run_key=run_key,
                source_champion_key=job_source,
                source_keys=tuple(sorted((grid_source, final_manifest_source, final_csv_source, job_source))),
            )
        )
    return (), tuple(sorted(seeds_by_id.values(), key=lambda item: item.id)), tuple(
        sorted(runs, key=lambda item: item.id)
    )


def _controller_profile_candidate(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    ledger: _SourceLedger,
) -> tuple[str, dict[str, dict[str, object]]]:
    path = root / "_v4" / "stack" / lineage_id / "controller-request.json"
    source = ledger.add(
        path,
        logical_role="controller",
        artifact_kind="controller-request",
        schema=control_audit.CONTROLLER_SCHEMA_VERSION,
    )
    payload = _json_object(path, "controller request")
    controller = _object(payload.get("controller"), "controller request controller")
    requested = {
        key: _required_mapping_value(controller, key, "controller request controller")
        for key in ("partition", "walltime", "cpus", "mem_per_cpu_gb")
    }
    candidate = ExecutionProfileV1(
        bundle_scope_id=bundle_scope_id,
        profile_kind="controller",
        requested=requested,
        source_keys=(source,),
    )
    return candidate.id, {
        candidate.id: {
            "profile_kind": candidate.profile_kind,
            "requested": requested,
            "source_keys": {source},
        }
    }


def _project_fanout_evidence(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    ledger: _SourceLedger,
    profile_candidates: dict[str, dict[str, object]],
) -> dict[str, _FanoutEvidence]:
    routes = load_routes()
    output: dict[str, _FanoutEvidence] = {}
    for role in sorted(_FANOUT_ROLES):
        route = routes[role]
        stage = ROLE_TO_STAGE[role]
        if route.kind != "fanout":
            raise ValueError(f"{role} is not a fan-out route")
        plan_dir = root / stage / "stage_plans" / lineage_id
        manifest_path = plan_dir / "stage_manifest.json"
        tasks_path = plan_dir / "tasks.jsonl"
        execution_path = plan_dir / "execution_records.jsonl"
        plan_sources = tuple(
            sorted(
                (
                    ledger.add(
                        manifest_path,
                        logical_role=role,
                        artifact_kind="stage-plan-manifest",
                        schema="experiment-toolkit/v1",
                    ),
                    ledger.add(
                        tasks_path,
                        logical_role=role,
                        artifact_kind="stage-plan-tasks",
                        schema="experiment-toolkit/v1",
                    ),
                    ledger.add(
                        execution_path,
                        logical_role=role,
                        artifact_kind="execution-records",
                        schema="experiment-toolkit/v1",
                    ),
                )
            )
        )
        try:
            plan = StagePlan.read(plan_dir)
            raw_records = tuple(iter_jsonl(execution_path))
            records = tuple(ExecutionRecord.from_dict(_object(row, "execution row")) for row in raw_records)
        except (OSError, StrictDataError, ValueError) as exc:
            raise ValueError(f"{role} stage-plan evidence is invalid: {exc}") from exc
        if plan.study != CONTRACT_STUDY or plan.stage != stage or plan.attempt_id != lineage_id:
            raise ValueError(f"{role} stage plan identity differs from V4 lineage")
        if len(plan.tasks) != len(records) or not plan.tasks:
            raise ValueError(f"{role} task/execution population is incomplete")
        task_ids = [task.task_id for task in plan.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{role} contains duplicate task identities")
        records_by_task = {record.task_id: record for record in records}
        if len(records_by_task) != len(records) or set(records_by_task) != set(task_ids):
            raise ValueError(f"{role} task/execution bindings are not one-to-one")
        first = plan.tasks[0]
        requested = _fanout_requested(first)
        candidate = ExecutionProfileV1(
            bundle_scope_id=bundle_scope_id,
            profile_kind="fanout",
            requested=requested,
            source_keys=plan_sources,
        )
        existing = profile_candidates.setdefault(
            candidate.id,
            {
                "profile_kind": candidate.profile_kind,
                "requested": requested,
                "source_keys": set(),
            },
        )
        if existing["profile_kind"] != candidate.profile_kind or existing["requested"] != requested:
            raise AssertionError("semantic execution profile id collision")
        source_set = existing["source_keys"]
        assert isinstance(source_set, set)
        source_set.update(plan_sources)
        for task in plan.tasks:
            if _fanout_requested(task) != requested:
                raise ValueError(f"{role} mixes requested execution profiles")
        result_source_keys: dict[str, tuple[str, ...]] = {}
        population: list[dict[str, object]] = []
        for task in sorted(plan.tasks, key=lambda item: item.task_id):
            record = records_by_task[task.task_id]
            _validate_task_execution_binding(
                task,
                record,
                root=root,
                stage=stage,
                lineage_id=lineage_id,
            )
            result_dir = _expected_result_directory(root, stage, task.run_id, lineage_id)
            _validate_direct_attempt_population(
                root,
                stage=stage,
                run_id=task.run_id,
                lineage_id=lineage_id,
            )
            status_source = ledger.add(
                result_dir / "status.json",
                logical_role=role,
                artifact_kind="terminal-status",
                schema="spenn-run-status/v1",
            )
            launcher_source = ledger.add(
                result_dir / "launcher_status.json",
                logical_role=role,
                artifact_kind="launcher-status",
                schema="experiment-launcher-status/v1",
            )
            keys = tuple(sorted((*plan_sources, status_source, launcher_source)))
            result_source_keys[task.task_id] = keys
            population.append(
                {
                    "task_id": task.task_id,
                    "execution_task_id": record.task_id,
                    "status_source": status_source,
                    "launcher_source": launcher_source,
                }
            )
        output[role] = _FanoutEvidence(
            role=role,
            stage=stage,
            plan=plan,
            records_by_task_id=records_by_task,
            plan_source_keys=plan_sources,
            result_source_keys=result_source_keys,
            terminal_population_sha256=canonical_sha256(
                {"role": role, "stage": stage, "terminal_population": population}
            ),
            execution_profile_id=candidate.id,
        )
    return output


def _project_training_producers(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    scan_runs: Sequence[RunV1],
    confirm_runs: Sequence[RunV1],
    fanout: Mapping[str, _FanoutEvidence],
    ledger: _SourceLedger,
) -> tuple[tuple[ProducerV1, ...], tuple[ProducerAttemptV1, ...]]:
    producers: list[ProducerV1] = []
    attempts: list[ProducerAttemptV1] = []
    for role, (stage, lane) in _TRAINING_ROLES.items():
        evidence = fanout[role]
        run_rows = scan_runs if lane == "scan" else confirm_runs
        runs_by_key = {row.run_key: row for row in run_rows}
        tasks_by_run = {task.run_id: task for task in evidence.plan.tasks}
        if len(tasks_by_run) != len(evidence.plan.tasks) or set(tasks_by_run) != set(runs_by_key):
            raise ValueError(f"{role} task population does not exactly match {lane} RunV1")
        for run_key, run in sorted(runs_by_key.items()):
            task = tasks_by_run[run_key]
            record = evidence.records_by_task_id[task.task_id]
            result_dir = _expected_result_directory(root, stage, run_key, lineage_id)
            relation_sources = _training_relation_sources(
                root,
                result_dir=result_dir,
                role=role,
                run_key=run_key,
                lineage_id=lineage_id,
                ledger=ledger,
            )
            claim_sources = _claim_sources(record, root=root, role=role, ledger=ledger)
            keys = tuple(
                sorted(
                    (*evidence.result_source_keys[task.task_id], *relation_sources, *claim_sources)
                )
            )
            producer = ProducerV1(
                bundle_scope_id=bundle_scope_id,
                run_id=run.id,
                role=role,
                source_keys=keys,
            )
            producers.append(producer)
            attempts.append(
                ProducerAttemptV1(
                    bundle_scope_id=bundle_scope_id,
                    producer_id=producer.id,
                    source_task_id=task.task_id,
                    source_execution_task_id=record.task_id,
                    source_keys=keys,
                )
            )
    return tuple(producers), tuple(attempts)


def _training_relation_sources(
    root: Path,
    *,
    result_dir: Path,
    role: str,
    run_key: str,
    lineage_id: str,
    ledger: _SourceLedger,
) -> tuple[str, ...]:
    if role == "screen_train":
        path = result_dir / "source_grid_attempt.json"
        payload = _json_object(path, "screen train source grid")
        expected = {
            "run_id": run_key,
            "grid_attempt_id": lineage_id,
            "grid_attempt_dir": str(root / "00_grid" / lineage_id),
            "manifest_path": str(root / "00_grid" / lineage_id / "manifest.json"),
        }
        if payload != expected:
            raise ValueError(f"screen train source-grid relation differs for {run_key}")
        return (
            ledger.add(
                path,
                logical_role=role,
                artifact_kind="source-grid-attempt",
                schema="pair-stability-source-grid/v1",
            ),
        )
    final_job_path = root / "05_final_grid" / lineage_id / "jobs" / f"{run_key}.json"
    final_job = _json_object(final_job_path, "confirmation final job")
    source_path = result_dir / "source_final_job.json"
    source_payload = _json_object(source_path, "confirmation train source final job")
    if source_payload != final_job:
        raise ValueError(f"confirmation train source final job differs for {run_key}")
    return tuple(
        sorted(
            (
                ledger.add(
                    source_path,
                    logical_role=role,
                    artifact_kind="source-final-job",
                    schema="pair-stability-final-job/v1",
                ),
                ledger.add(
                    result_dir / "source_final_grid_attempt.json",
                    logical_role=role,
                    artifact_kind="source-final-grid-attempt",
                    schema="pair-stability-source-final-grid/v1",
                ),
            )
        )
    )


def _claim_sources(
    record: ExecutionRecord,
    *,
    root: Path,
    role: str,
    ledger: _SourceLedger,
) -> tuple[str, ...]:
    if record.claim_path is None:
        return ()
    path = _regular_beneath_root(Path(record.claim_path), root)
    payload = _json_object(path, "execution claim")
    if payload.get("reclaimed") is True:
        raise ValueError("V4-1A does not recognize reclaimed producer attempts")
    return (
        ledger.add(
            path,
            logical_role=role,
            artifact_kind="task-claim",
            schema="experiment-task-claim/v1",
        ),
    )


def _materialize_profiles(
    bundle_scope_id: str,
    candidates: Mapping[str, Mapping[str, object]],
) -> tuple[ExecutionProfileV1, ...]:
    profiles: list[ExecutionProfileV1] = []
    for candidate_id, data in candidates.items():
        sources = data.get("source_keys")
        if not isinstance(sources, set) or not sources:
            raise ValueError("execution profile has no explicit source evidence")
        profile = ExecutionProfileV1(
            bundle_scope_id=bundle_scope_id,
            profile_kind=_required_text(data.get("profile_kind"), "profile kind"),
            requested=_object(data.get("requested"), "profile requested values"),
            source_keys=tuple(sorted(_required_identifier(item, "profile source") for item in sources)),
        )
        if profile.id != candidate_id:
            raise AssertionError("execution profile semantic identity changed during source aggregation")
        profiles.append(profile)
    return tuple(sorted(profiles, key=lambda item: item.id))


def _project_stage_results(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    controller_profile_id: str,
    fanout: Mapping[str, _FanoutEvidence],
    ledger: _SourceLedger,
) -> tuple[StageResultV1, ...]:
    rows: list[StageResultV1] = []
    for role, stage in sorted(ROLE_TO_STAGE.items()):
        if role in fanout:
            evidence = fanout[role]
            rows.append(
                StageResultV1(
                    bundle_scope_id=bundle_scope_id,
                    logical_role=role,
                    physical_stage=stage,
                    execution_profile_id=evidence.execution_profile_id,
                    terminal_population_sha256=evidence.terminal_population_sha256,
                    source_keys=tuple(
                        sorted(
                            {
                                source
                                for keys in evidence.result_source_keys.values()
                                for source in keys
                            }
                        )
                    ),
                )
            )
            continue
        marker_stage, marker_name = _DIRECT_STAGE_MARKERS[role]
        if marker_stage != stage:
            raise AssertionError("direct stage marker route differs")
        marker_path = root / stage / lineage_id / marker_name
        marker_kind, marker_schema = {
            "screen_plan": ("grid-manifest", "pair-stability-grid/v1"),
            "confirm_plan": (
                "final-grid-manifest",
                "pair-stability-final-grid/v1",
            ),
        }.get(role, ("terminal-stage-marker", "pair-stability-stage-result/v1"))
        marker_source = ledger.add(
            marker_path,
            logical_role=role,
            artifact_kind=marker_kind,
            schema=marker_schema,
        )
        rows.append(
            StageResultV1(
                bundle_scope_id=bundle_scope_id,
                logical_role=role,
                physical_stage=stage,
                execution_profile_id=controller_profile_id,
                terminal_population_sha256=canonical_sha256(
                    {"role": role, "stage": stage, "marker_source": marker_source, "marker_sha256": _sha256_file(marker_path)}
                ),
                source_keys=(marker_source,),
            )
        )
    return tuple(rows)


def _project_metric_keys(
    root: Path,
    *,
    lineage_id: str,
    bundle_scope_id: str,
    stages_by_role: Mapping[str, StageResultV1],
    fanout: Mapping[str, _FanoutEvidence],
    ledger: _SourceLedger,
) -> tuple[MetricKeyV1, ...]:
    observed: dict[tuple[str, str, str], tuple[str, set[str]]] = {}
    for role in ("screen_eval", "confirm_eval"):
        evidence = fanout[role]
        stage_id = stages_by_role[role].id
        for task in evidence.plan.tasks:
            result_dir = _expected_result_directory(root, evidence.stage, task.run_id, lineage_id)
            path = result_dir / "metrics.jsonl"
            source = ledger.add(
                path,
                logical_role=role,
                artifact_kind="literal-metrics",
                schema="spenn-metrics-jsonl/v1",
            )
            for row in iter_jsonl(path):
                payload = _object(row, f"{role} metric row")
                if set(payload) != {"namespace", "step", "metrics"}:
                    raise ValueError(f"{role} metric row does not use literal metrics shape")
                namespace = _required_text(payload.get("namespace"), "metric namespace")
                metrics = _object(payload.get("metrics"), "metric values")
                for key, value in metrics.items():
                    metric_key = _required_text(key, "metric key")
                    representation = _scalar_representation(value)
                    identity = (stage_id, namespace, metric_key)
                    existing = observed.get(identity)
                    if existing is not None and existing[0] != representation:
                        raise ValueError(
                            "literal metric scalar representation differs across "
                            f"completed {role} tasks: {namespace}/{metric_key}"
                        )
                    source_set = set() if existing is None else existing[1]
                    source_set.add(source)
                    observed[identity] = (representation, source_set)
    if not observed:
        raise ValueError("completed evaluation stages exposed no literal metric keys")
    return tuple(
        MetricKeyV1(
            bundle_scope_id=bundle_scope_id,
            stage_result_id=stage_id,
            namespace=namespace,
            key=key,
            scalar_representation=representation,
            source_keys=tuple(sorted(source_keys)),
        )
        for (stage_id, namespace, key), (representation, source_keys) in sorted(observed.items())
    )


def _fanout_requested(task: object) -> dict[str, object]:
    resources = getattr(task, "resources", None)
    if resources is None:
        raise ValueError("fan-out task has no typed ResourceSpec")
    return {
        "profile": resources.profile,
        "device": resources.device,
        "partition": resources.partition,
        "threads": resources.threads,
        "mem_gb": resources.mem_gb,
        "gpus": resources.gpus,
        "timeout_min": resources.timeout_min,
        "uv_environment": resources.uv_environment,
        "uv_extras": list(resources.uv_extras),
    }


def _validate_task_execution_binding(
    task: object,
    record: ExecutionRecord,
    *,
    root: Path,
    stage: str,
    lineage_id: str,
) -> None:
    task_id = _required_text(getattr(task, "task_id", None), "task id")
    run_id = _required_identifier(getattr(task, "run_id", None), "task run_id")
    if getattr(task, "stage", None) != stage or getattr(task, "attempt_id", None) != lineage_id:
        raise ValueError("task stage/attempt is foreign to the V4-1A lineage")
    if record.task_id != task_id or record.run_id != run_id:
        raise ValueError("execution record does not bind the planned task identity")
    if record.stage != stage or record.attempt_id != lineage_id:
        raise ValueError("execution record stage/attempt is foreign to the V4-1A lineage")
    if getattr(task, "resume", None):
        raise ValueError("V4-1A does not recognize nonempty typed TaskSpec.resume")
    expected = _expected_result_directory(root, stage, run_id, lineage_id)
    actual = _regular_or_directory_beneath_root(Path(getattr(task, "result_dir", "")), root)
    if actual != expected:
        raise ValueError("task result directory differs from canonical lineage attempt")
    if record.status_path is None:
        raise ValueError("execution record lacks explicit terminal launcher-status path")
    status_path = _regular_beneath_root(Path(record.status_path), root)
    if status_path != expected / "launcher_status.json":
        raise ValueError("execution record status path is not canonical launcher status")


def _validate_direct_attempt_population(
    root: Path,
    *,
    stage: str,
    run_id: str,
    lineage_id: str,
) -> None:
    parent = require_beneath_root(root / stage / run_id, root)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("training result parent is missing or unsafe")
    allowed = {lineage_id, "latest", "latest.json"}
    actual = {entry.name for entry in parent.iterdir()}
    if actual - allowed:
        raise ValueError(
            "V4-1A training result parent has unsupported attempt evidence: "
            f"{sorted(actual - allowed)}"
        )
    expected = parent / lineage_id
    if not expected.is_dir() or expected.is_symlink():
        raise ValueError("canonical training attempt directory is missing or unsafe")
    for pointer_name in ("latest", "latest.json"):
        pointer = parent / pointer_name
        if pointer.exists() or pointer.is_symlink():
            # Pointers are tolerated as legacy filesystem state but are never
            # parsed, selected, or cited as semantic V4-1A evidence.
            if pointer_name == "latest.json" and (pointer.is_symlink() or not pointer.is_file()):
                raise ValueError("legacy latest.json pointer is unsafe")


def _expected_result_directory(root: Path, stage: str, run_id: str, lineage_id: str) -> Path:
    return require_beneath_root(root / stage / run_id / lineage_id, root)


def _regular_beneath_root(path: Path, root: Path) -> Path:
    candidate = require_beneath_root(path, root)
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"required source artifact is unavailable: {candidate}") from exc
    if candidate.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"required source artifact is not a regular file: {candidate}")
    return candidate


def _regular_or_directory_beneath_root(path: Path, root: Path) -> Path:
    candidate = require_beneath_root(path, root)
    if candidate.is_symlink() or not candidate.exists() or not (candidate.is_file() or candidate.is_dir()):
        raise ValueError(f"artifact path is missing or unsafe: {candidate}")
    return candidate


def _final_job_ids(path: Path) -> tuple[str, ...]:
    _require_regular_file(path)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read final job CSV: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("final job CSV is empty")
    ids = tuple(_required_identifier(row.get("final_run_id"), "final job final_run_id") for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("final job CSV contains duplicate final_run_id")
    return tuple(sorted(ids))


def _require_preclose_and_lineage(root: Path, lineage_id: str) -> None:
    attempts = _attempts(lineage_id)
    control_errors = control_audit.audit_preclose_control(root, attempts=attempts)
    if control_errors:
        raise ValueError("pre-close control audit failed: " + "; ".join(control_errors))
    lineage_errors = audit.audit_completed_lineage(root, attempts=attempts)
    if lineage_errors:
        raise ValueError("completed-lineage audit failed: " + "; ".join(lineage_errors))


def _attempts(lineage_id: str) -> dict[str, str]:
    return {key: lineage_id for key in _ATTEMPT_KEYS}


def _root(root: Path, lineage_id: str) -> Path:
    return require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )


def _json_object(path: Path, label: str) -> dict[str, object]:
    _require_regular_file(path)
    try:
        payload = load_json(path)
    except (OSError, StrictDataError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return payload


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"required structured artifact is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"required structured artifact is not a regular file: {path}")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _object_rows(value: object, label: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is not a nonempty row list")
    return tuple(_object(item, label) for item in value)


def _text_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} is not a nonempty text list")
    return tuple(value)


def _required_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(character in value for character in "/\\"):
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not nonempty text")
    return value


def _required_mapping_value(mapping: Mapping[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ValueError(f"{label} is missing {key}")
    return mapping[key]


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} is not a positive integer")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from expected V4 lineage identity")


def _scalar_representation(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metric scalar is non-finite")
        return "float"
    if isinstance(value, str):
        return "str"
    raise ValueError("metric value is not a literal scalar")


def _csv_provenance_scalar(value: object) -> str:
    """Return the V3 CSV spelling used by final champion provenance."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError("grid configuration axis is not a scalar CSV provenance value")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_verify(root: Path, *, lineage_id: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "verify",
            "--results-root",
            str(root),
            "--lineage-id",
            lineage_id,
            "--no-receipt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "fresh V4-1A sidecar verification failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("fresh V4-1A verifier returned invalid JSON") from exc
    if not isinstance(payload, dict) or not all(isinstance(value, str) for value in payload.values()):
        raise ValueError("fresh V4-1A verifier returned invalid result")
    return {str(key): str(value) for key, value in payload.items()}


def _validate_verification_receipt(
    path: Path,
    *,
    lineage_id: str,
    result: Mapping[str, str],
) -> None:
    payload = _json_object(path, "contract verification receipt")
    expected = {
        "schema_version": VERIFICATION_RECEIPT_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "bundle_scope_id": result["bundle_scope_id"],
        "manifest_sha256": result["manifest_sha256"],
        "preclose_control_sha256": result["preclose_control_sha256"],
        "verifier_version": VERIFIER_VERSION,
        "status": "verified",
    }
    if set(payload) != {*expected, "verified_at"}:
        raise ValueError("contract verification receipt fields differ from v1")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"contract verification receipt {key} differs")
    if not isinstance(payload.get("verified_at"), str) or not payload["verified_at"]:
        raise ValueError("contract verification receipt timestamp is invalid")


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="microseconds")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the finalizer or fresh verifier without a controller import path."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "verify"):
        item = commands.add_parser(command)
        item.add_argument("--results-root", required=True)
        item.add_argument("--lineage-id", required=True)
        if command == "verify":
            item.add_argument("--no-receipt", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "finalize":
        print(finalize_contract_sidecars(Path(args.results_root), lineage_id=args.lineage_id))
        return 0
    result = verify_contract_sidecars(
        Path(args.results_root),
        lineage_id=args.lineage_id,
        require_receipt=not args.no_receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
