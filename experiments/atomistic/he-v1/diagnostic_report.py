"""Publish deterministic figures and Markdown for ``he-v1-diagnostic-v1``.

This stage is deliberately downstream-only.  It verifies the content-addressed
plan, the successful 42-row collection receipt, and every artifact's size and
SHA-256 before parsing any scientific value.  Large CSV records are then read
through a read-only :mod:`mmap`; they are never copied wholesale into memory.

Estimator semantics remain explicit throughout the output.  The primary energy
is the full retained-trajectory mean with its correlation-aware MCSE.  Snapshot
IID summaries, long-chain/sensitivity diagnostics, conditioned statistics,
executed versus ideal cusp behavior, and fixed-configuration versus
re-equilibrated factor responses are never pooled or silently substituted.

This module imports no :mod:`tpen`, as required by ``experiments/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import mmap
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import diagnostic_collect as collect_stage  # noqa: E402
import diagnostic_plan as plan_stage  # noqa: E402
import diagnostic_plot as plot_stage  # noqa: E402
import layout  # noqa: E402

REPORT_SCHEMA = "he-v1-diagnostic-report/v1"
CONDITIONED_SCHEMA = "conditioned_local_energy/v1"
TRAJECTORY_SCHEMA = "trajectory_records/v1"
REPORT_FILENAME = "report.md"
REPORT_MANIFEST_FILENAME = "report_manifest.json"
EXPECTED_CHECKPOINTS = ("step_025000", "step_050000")
EXPECTED_ROW_COUNT = 42
EXPECTED_FACTOR_ARMS = (
    "baseline",
    "b_ee_minus_10pct",
    "b_ee_plus_10pct",
    "c_en_minus_10pct",
    "c_en_plus_10pct",
    "d_en_minus_10pct",
    "d_en_plus_10pct",
)
REQUIRED_CONDITION_QUANTITIES = (
    "minimum_electron_nuclear_radius",
    "electron_electron_distance",
    "maximum_electron_nuclear_radius",
    "hyperradius",
    "cos_theta12",
    "logabs",
)
TABLE_NAMES = (
    "energy_mcse",
    "distribution_ccdf",
    "conditioned_variance",
    "cusp_curvature",
    "singular_cancellation",
    "tails",
    "symmetry_equivariance",
    "sampler_health_timing",
    "factor_response",
)
PROVENANCE_KEYS = (
    "plan_sha256",
    "evaluation_git_sha",
    "checkpoint_label",
    "checkpoint_model_sha256",
    "source_row_ids",
    "source_artifact_sha256",
)


class DiagnosticReportError(RuntimeError):
    """A reporting input failed its explicit provenance or semantic contract."""


@dataclass(frozen=True)
class VerifiedArtifact:
    """One collected artifact whose current bytes match the collection receipt."""

    row_id: str
    task: str
    namespace: str
    name: str
    kind: str
    path: Path
    provenance_path: str
    sha256: str
    byte_count: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedRow:
    """One collected row joined exactly to its immutable planned row."""

    planned: Mapping[str, Any]
    collected: Mapping[str, Any]
    plan_attempt_id: str
    artifacts: tuple[VerifiedArtifact, ...]
    metrics_by_namespace: Mapping[str, Mapping[str, Any]]

    @property
    def row_id(self) -> str:
        return str(self.planned["row_id"])

    @property
    def checkpoint_label(self) -> str:
        return str(self.planned["checkpoint_label"])

    def artifact(self, *, task: str, name: str) -> VerifiedArtifact:
        matches = [
            artifact
            for artifact in self.artifacts
            if artifact.task == task and artifact.name == name
        ]
        if len(matches) != 1:
            raise DiagnosticReportError(
                f"row {self.row_id} requires exactly one {task}/{name} artifact; "
                f"found {len(matches)}"
            )
        return matches[0]

    def task_metrics(self, task: str) -> Mapping[str, Any]:
        namespaces = {
            artifact.namespace for artifact in self.artifacts if artifact.task == task
        }
        if len(namespaces) != 1:
            raise DiagnosticReportError(
                f"row {self.row_id} task {task!r} has ambiguous artifact namespaces: "
                f"{sorted(namespaces)}"
            )
        namespace = next(iter(namespaces))
        metrics = self.metrics_by_namespace.get(namespace)
        if metrics is None:
            raise DiagnosticReportError(
                f"row {self.row_id} has no collected metrics for task namespace {namespace!r}"
            )
        return metrics


@dataclass(frozen=True)
class VerifiedStudy:
    """The complete manifest-verified source bundle for one report."""

    results_root: Path
    manifest: Mapping[str, Any]
    collected: Mapping[str, Any]
    rows: tuple[VerifiedRow, ...]
    collected_sha256: str

    @property
    def plan_sha256(self) -> str:
        return str(self.manifest["plan_sha256"])

    @property
    def evaluation_git_sha(self) -> str:
        return str(self.manifest["evaluation_git_sha"])


def read_verified_study(
    results_root: str | Path,
    *,
    plan_attempt_id: str,
    collect_attempt_id: str,
) -> VerifiedStudy:
    """Verify the plan, collection receipt, rows, and artifact bytes."""

    root = Path(results_root).resolve()
    manifest = plan_stage.read_manifest(root, plan_attempt_id)
    collected_path = layout.collect_attempt_dir(root, collect_attempt_id) / "collected.json"
    collected_bytes = collected_path.read_bytes()
    collected_sha256 = hashlib.sha256(collected_bytes).hexdigest()
    collected = _json_mapping_from_bytes(collected_bytes, source=collected_path)
    _validate_collection(
        manifest,
        collected,
        plan_attempt_id=plan_attempt_id,
        collect_attempt_id=collect_attempt_id,
    )

    planned_rows = list(manifest["rows"])
    collected_rows = list(collected["rows"])
    verified_rows = tuple(
        _verify_row(
            planned,
            observed,
            results_root=root,
            plan_attempt_id=plan_attempt_id,
        )
        for planned, observed in zip(planned_rows, collected_rows, strict=True)
    )
    return VerifiedStudy(
        results_root=root,
        manifest=manifest,
        collected=collected,
        rows=verified_rows,
        collected_sha256=collected_sha256,
    )


def _validate_collection(
    manifest: Mapping[str, Any],
    collected: Mapping[str, Any],
    *,
    plan_attempt_id: str,
    collect_attempt_id: str,
) -> None:
    expected = {
        "schema": collect_stage.COLLECT_SCHEMA,
        "study": manifest["study"],
        "scale": manifest["scale"],
        "plan_attempt_id": plan_attempt_id,
        "plan_sha256": manifest["plan_sha256"],
        "collect_attempt_id": collect_attempt_id,
        "evaluation_git_sha": manifest["evaluation_git_sha"],
        "checkpoint_reporting": "both_without_selection",
        "selection_policy": "none",
        "production_run_mutation_authorized": False,
        "status": "success",
        "n_planned_rows": EXPECTED_ROW_COUNT,
        "n_collected_rows": EXPECTED_ROW_COUNT,
    }
    mismatches = [
        f"{key}: collected={collected.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if collected.get(key) != value
    ]
    if collected.get("errors") != []:
        mismatches.append(f"errors must be empty, got {collected.get('errors')!r}")
    planned = manifest.get("rows")
    observed = collected.get("rows")
    if not isinstance(planned, Sequence) or isinstance(planned, (str, bytes)):
        mismatches.append("manifest rows are not a sequence")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        mismatches.append("collected rows are not a sequence")
    if isinstance(planned, Sequence) and not isinstance(planned, (str, bytes)):
        if len(planned) != EXPECTED_ROW_COUNT:
            mismatches.append(f"manifest requires {EXPECTED_ROW_COUNT} rows, found {len(planned)}")
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        if len(observed) != EXPECTED_ROW_COUNT:
            mismatches.append(f"collection requires {EXPECTED_ROW_COUNT} rows, found {len(observed)}")
    checkpoint_labels = tuple(item.get("checkpoint_label") for item in collected.get("checkpoints", ()))
    if checkpoint_labels != EXPECTED_CHECKPOINTS:
        mismatches.append(
            f"checkpoint order must be {EXPECTED_CHECKPOINTS!r}, got {checkpoint_labels!r}"
        )
    if manifest.get("production_grid_sha256_before") != collected.get(
        "production_grid_sha256_before"
    ) or collected.get("production_grid_sha256_before") != collected.get(
        "production_grid_sha256_after"
    ):
        mismatches.append("production grid hash changed across diagnostic evaluation")
    if mismatches:
        raise DiagnosticReportError("collection receipt verification failed: " + "; ".join(mismatches))


def _verify_row(
    planned: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    results_root: Path,
    plan_attempt_id: str,
) -> VerifiedRow:
    keys = (
        "row_id",
        "checkpoint_label",
        "checkpoint_model_sha256",
        "profile",
        "protocol",
        "comparison_kind",
        "factor_arm",
        "seed",
        "n_walkers",
        "n_draws",
        "burn_in",
        "stride",
        "task_names",
    )
    mismatches = [
        f"{key}: collected={observed.get(key)!r}, planned={planned.get(key)!r}"
        for key in keys
        if observed.get(key) != planned.get(key)
    ]
    config_sha = observed.get("config_sha256")
    if not _is_sha256(config_sha):
        mismatches.append(f"config_sha256 is invalid: {config_sha!r}")
    if observed.get("artifact_count") != len(observed.get("artifacts", ())):
        mismatches.append("artifact_count disagrees with artifacts")
    if mismatches:
        raise DiagnosticReportError(
            f"row {planned.get('row_id')!r} differs from its plan: " + "; ".join(mismatches)
        )

    row_id = str(planned["row_id"])
    run_dir = layout.row_dir(
        results_root,
        layout.STAGE_EVAL,
        row_id,
        plan_attempt_id,
    ) / row_id
    artifacts = tuple(
        _verify_artifact(
            row_id,
            record,
            run_dir=run_dir,
            results_root=results_root,
        )
        for record in observed["artifacts"]
    )
    planned_tasks = set(planned["task_names"])
    artifact_tasks = {artifact.task for artifact in artifacts}
    if artifact_tasks != planned_tasks:
        raise DiagnosticReportError(
            f"row {row_id} artifact task coverage changed: "
            f"observed={sorted(artifact_tasks)!r}, planned={sorted(planned_tasks)!r}"
        )
    metric_records = observed.get("metrics")
    if not isinstance(metric_records, Sequence) or isinstance(metric_records, (str, bytes)):
        raise DiagnosticReportError(f"row {row_id} metrics are not a sequence")
    metrics_by_namespace = _merge_metrics(row_id, metric_records)
    if "eval/perf" not in metrics_by_namespace or "runtime" not in metrics_by_namespace:
        raise DiagnosticReportError(f"row {row_id} lacks eval/perf or runtime metrics")
    return VerifiedRow(
        planned=dict(planned),
        collected=dict(observed),
        plan_attempt_id=plan_attempt_id,
        artifacts=artifacts,
        metrics_by_namespace=metrics_by_namespace,
    )


def _verify_artifact(
    row_id: str,
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    results_root: Path,
) -> VerifiedArtifact:
    required = ("task", "namespace", "name", "kind", "path", "sha256", "bytes", "metadata")
    missing = [key for key in required if key not in record]
    if missing:
        raise DiagnosticReportError(f"row {row_id} artifact lacks fields {missing}")
    path = Path(str(record["path"]))
    resolved_results_root = results_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (resolved_results_root / path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise DiagnosticReportError(
            f"row {row_id} artifact escapes its run directory: {resolved}"
        ) from exc
    try:
        provenance_path = resolved.relative_to(resolved_results_root).as_posix()
    except ValueError as exc:
        raise DiagnosticReportError(
            f"row {row_id} artifact escapes its results root: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise DiagnosticReportError(f"row {row_id} artifact is missing: {resolved}")
    expected_bytes = int(record["bytes"])
    actual_bytes = resolved.stat().st_size
    if actual_bytes != expected_bytes:
        raise DiagnosticReportError(
            f"row {row_id} artifact byte mismatch at {resolved}: "
            f"collected={expected_bytes}, actual={actual_bytes}"
        )
    expected_sha = str(record["sha256"])
    actual_sha = _file_sha256(resolved)
    if expected_sha != actual_sha:
        raise DiagnosticReportError(
            f"row {row_id} artifact hash mismatch at {resolved}: "
            f"collected={expected_sha}, actual={actual_sha}"
        )
    metadata = record["metadata"]
    if not isinstance(metadata, Mapping):
        raise DiagnosticReportError(f"row {row_id} artifact metadata is not a mapping")
    return VerifiedArtifact(
        row_id=row_id,
        task=str(record["task"]),
        namespace=str(record["namespace"]),
        name=str(record["name"]),
        kind=str(record["kind"]),
        path=resolved,
        provenance_path=provenance_path,
        sha256=actual_sha,
        byte_count=actual_bytes,
        metadata=dict(metadata),
    )


def _merge_metrics(
    row_id: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        namespace = record.get("namespace")
        metrics = record.get("metrics")
        if not isinstance(namespace, str) or not isinstance(metrics, Mapping):
            raise DiagnosticReportError(f"row {row_id} malformed metric record {index}")
        target = merged.setdefault(namespace, {})
        for key, value in metrics.items():
            name = str(key)
            if name in target and target[name] != value:
                raise DiagnosticReportError(
                    f"row {row_id} metric {namespace}.{name} has conflicting values"
                )
            target[name] = value
    if not merged:
        raise DiagnosticReportError(f"row {row_id} has no metric records")
    return merged


@contextmanager
def mmap_csv_rows(
    artifact: VerifiedArtifact,
    *,
    required_fields: Sequence[str],
) -> Iterator[tuple[tuple[str, ...], Iterator[dict[str, str]]]]:
    """Yield a verified CSV through a read-only memory map.

    The map's size and SHA-256 are checked again at open and close so a writer
    cannot change bytes between verification and consumption unnoticed.
    """

    if artifact.byte_count <= 0:
        raise DiagnosticReportError(f"cannot memory-map empty artifact {artifact.path}")
    with artifact.path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            if len(mapped) != artifact.byte_count:
                raise DiagnosticReportError(f"artifact size changed before mmap: {artifact.path}")
            if hashlib.sha256(mapped).hexdigest() != artifact.sha256:
                raise DiagnosticReportError(f"artifact hash changed before mmap: {artifact.path}")
            mapped.seek(0)

            def decoded_lines() -> Iterator[str]:
                while encoded := mapped.readline():
                    yield encoded.decode("utf-8")

            reader = csv.DictReader(decoded_lines())
            fieldnames = tuple(reader.fieldnames or ())
            missing = sorted(set(required_fields) - set(fieldnames))
            if missing:
                raise DiagnosticReportError(
                    f"artifact {artifact.path} lacks required CSV fields {missing}"
                )
            yield fieldnames, reader
            if len(mapped) != artifact.byte_count or hashlib.sha256(mapped).hexdigest() != artifact.sha256:
                raise DiagnosticReportError(f"artifact identity changed during mmap: {artifact.path}")
        finally:
            mapped.close()


def _read_json_artifact(artifact: VerifiedArtifact) -> dict[str, Any]:
    return _json_mapping_from_bytes(
        _read_verified_artifact_bytes(artifact), source=artifact.path
    )


def _read_jsonl_one(artifact: VerifiedArtifact) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in _read_verified_artifact_bytes(artifact).decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise DiagnosticReportError(
            f"artifact {artifact.path} must contain exactly one JSONL mapping"
        )
    return records[0]


def _json_mapping_from_bytes(payload: bytes, *, source: Path) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticReportError(f"required JSON artifact is not a mapping: {source}")
    return value


def _read_verified_artifact_bytes(artifact: VerifiedArtifact) -> bytes:
    payload = artifact.path.read_bytes()
    if len(payload) != artifact.byte_count:
        raise DiagnosticReportError(f"artifact size changed before read: {artifact.path}")
    if hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise DiagnosticReportError(f"artifact hash changed before read: {artifact.path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DiagnosticReportError(f"{name} must be finite, got {value!r}")
    return number


def _bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise DiagnosticReportError(f"expected CSV boolean true|false, got {value!r}")
    return normalized == "true"


def _provenance(
    study: VerifiedStudy,
    *,
    checkpoint_label: str,
    checkpoint_model_sha256: str,
    source_row_ids: Sequence[str],
    artifact_sha256: Sequence[str],
) -> dict[str, str]:
    return {
        "plan_sha256": study.plan_sha256,
        "evaluation_git_sha": study.evaluation_git_sha,
        "checkpoint_label": checkpoint_label,
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "source_row_ids": ";".join(source_row_ids),
        "source_artifact_sha256": ";".join(artifact_sha256),
    }


def build_plot_tables(study: VerifiedStudy) -> dict[str, list[dict[str, Any]]]:
    """Build the nine provenance-bearing plot-data tables."""

    energy = _energy_rows(study)
    distribution, conditioned = _distribution_and_conditioned_rows(study)
    cusp, cancellation, tails = _atlas_rows(study)
    symmetry = _symmetry_rows(study)
    health = _health_timing_rows(study, energy)
    factor = _factor_rows(study)
    tables = {
        "energy_mcse": energy,
        "distribution_ccdf": distribution,
        "conditioned_variance": conditioned,
        "cusp_curvature": cusp,
        "singular_cancellation": cancellation,
        "tails": tails,
        "symmetry_equivariance": symmetry,
        "sampler_health_timing": health,
        "factor_response": factor,
    }
    missing = sorted(set(TABLE_NAMES) - set(tables))
    if missing or any(not tables[name] for name in TABLE_NAMES):
        empty = [name for name in TABLE_NAMES if not tables.get(name)]
        raise DiagnosticReportError(
            f"plot-data construction is incomplete: missing={missing}, empty={empty}"
        )
    return tables


def _energy_rows(study: VerifiedStudy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in study.rows:
        if row.planned["profile"] != "retained_energy":
            continue
        receipt, artifact = _trajectory_receipt(row, task="retained_energy")
        metrics = row.task_metrics("retained_energy")
        snapshot_mean = _finite_float(metrics.get("local_energy_mean"), "local_energy_mean")
        snapshot_stderr = _finite_float(
            metrics.get("local_energy_stderr"), "local_energy_stderr"
        )
        status = str(receipt["status"])
        statistics = receipt.get("statistics")
        record: dict[str, Any] = {
            **_provenance(
                study,
                checkpoint_label=row.checkpoint_label,
                checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                source_row_ids=[row.row_id],
                artifact_sha256=[artifact.sha256],
            ),
            "row_id": row.row_id,
            "protocol": row.planned["protocol"],
            "comparison_kind": row.planned["comparison_kind"],
            "estimator_role": (
                "primary"
                if row.planned["comparison_kind"] == "primary_headline"
                else "diagnostic"
            ),
            "status": status,
            "estimator_id": receipt["estimator_id"],
            "estimator_version": receipt["estimator_version"],
            "tau_convention": receipt["tau_convention"],
            "trajectory_mean_ha": "unavailable",
            "mcse_ha": "unavailable",
            "same_trajectory_iid_stderr_ha": "unavailable",
            "mcse_inflation": "unavailable",
            "tau_int": "unavailable",
            "ess": "unavailable",
            "snapshot_iid_mean_ha": snapshot_mean,
            "snapshot_iid_stderr_ha": snapshot_stderr,
            "snapshot_estimator_label": "terminal-snapshot IID diagnostic; not an MCSE",
            "unavailable_reason": receipt.get("reason") or "not_applicable",
        }
        if status == "available":
            if not isinstance(statistics, Mapping):
                raise DiagnosticReportError(f"row {row.row_id} available receipt lacks statistics")
            record.update(
                {
                    "trajectory_mean_ha": _finite_float(
                        statistics.get("mean"), "trajectory statistics mean"
                    ),
                    "mcse_ha": _finite_float(statistics.get("mcse"), "trajectory MCSE"),
                    "same_trajectory_iid_stderr_ha": _finite_float(
                        metrics.get("local_energy_stderr_iid"),
                        "same-trajectory IID stderr",
                    ),
                    "mcse_inflation": _finite_float(
                        metrics.get("local_energy_mcse_inflation"), "MCSE inflation"
                    ),
                    "tau_int": _finite_float(statistics.get("tau_int"), "tau_int"),
                    "ess": _finite_float(statistics.get("ess"), "ESS"),
                    "unavailable_reason": "not_applicable",
                }
            )
        rows.append(record)
    if len(rows) != 24:
        raise DiagnosticReportError(f"energy table requires 24 retained-energy rows, found {len(rows)}")
    return rows


def _trajectory_receipt(
    row: VerifiedRow,
    *,
    task: str,
) -> tuple[dict[str, Any], VerifiedArtifact]:
    artifact = row.artifact(task=task, name="local_energy_trajectory_statistics")
    if artifact.kind != "trajectory_statistics_sidecar":
        raise DiagnosticReportError(f"row {row.row_id} trajectory artifact kind changed")
    receipt = _read_jsonl_one(artifact)
    expected_identity = {
        "stage": row.planned["stage"],
        "run_id": row.row_id,
        "attempt_id": row.plan_attempt_id,
        "checkpoint_sha256": row.planned["checkpoint_model_sha256"],
        "config_sha256": row.collected["config_sha256"],
        "observable": "local_energy",
        "evaluator_id": "he-v1-diagnostic-v1",
    }
    _validate_trajectory_receipt(row, receipt, expected_identity=expected_identity)
    metrics = row.task_metrics(task)
    available_metric = metrics.get("local_energy_trajectory_statistics_available")
    if not isinstance(available_metric, bool) or available_metric != (
        receipt["status"] == "available"
    ):
        raise DiagnosticReportError(
            f"row {row.row_id} trajectory availability metric disagrees with receipt"
        )
    _reconcile_trajectory_metrics(row, receipt, metrics)
    return receipt, artifact


def _reconcile_trajectory_metrics(
    row: VerifiedRow,
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    statistic_keys = {
        "mean": "local_energy_trajectory_mean",
        "variance": "local_energy_trajectory_variance",
        "mcse": "local_energy_mcse",
        "ess": "local_energy_ess",
        "tau_int": "local_energy_tau_int",
    }
    statistics = receipt.get("statistics")
    if receipt["status"] != "available":
        leaked = sorted(metric for metric in statistic_keys.values() if metric in metrics)
        if leaked:
            raise DiagnosticReportError(
                f"row {row.row_id} unavailable trajectory publishes numeric metrics {leaked}"
            )
        return
    if not isinstance(statistics, Mapping):
        raise DiagnosticReportError(f"row {row.row_id} available trajectory lacks statistics")
    mismatches: list[str] = []
    for receipt_key, metric_key in statistic_keys.items():
        receipt_value = _finite_float(statistics.get(receipt_key), receipt_key)
        metric_value = _finite_float(metrics.get(metric_key), metric_key)
        if not math.isclose(receipt_value, metric_value, rel_tol=1.0e-12, abs_tol=1.0e-15):
            mismatches.append(
                f"{metric_key}={metric_value!r}, receipt.{receipt_key}={receipt_value!r}"
            )
    total_draws = int(receipt["shape"]["total_draws"])
    variance = _finite_float(statistics["variance"], "trajectory variance")
    stderr_iid = math.sqrt(variance / total_draws)
    observed_stderr = _finite_float(
        metrics.get("local_energy_stderr_iid"), "local_energy_stderr_iid"
    )
    if not math.isclose(stderr_iid, observed_stderr, rel_tol=1.0e-12, abs_tol=1.0e-15):
        mismatches.append(
            f"local_energy_stderr_iid={observed_stderr!r}, derived={stderr_iid!r}"
        )
    if stderr_iid > 0.0:
        inflation = _finite_float(statistics["mcse"], "trajectory MCSE") / stderr_iid
        observed_inflation = _finite_float(
            metrics.get("local_energy_mcse_inflation"),
            "local_energy_mcse_inflation",
        )
        if not math.isclose(
            inflation,
            observed_inflation,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            mismatches.append(
                f"local_energy_mcse_inflation={observed_inflation!r}, "
                f"derived={inflation!r}"
            )
    if mismatches:
        raise DiagnosticReportError(
            f"row {row.row_id} trajectory receipt/metric mismatch: " + "; ".join(mismatches)
        )


def _validate_trajectory_receipt(
    row: VerifiedRow,
    receipt: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> None:
    statuses = {"available", "absent", "unresolved"}
    status = receipt.get("status")
    if status not in statuses:
        raise DiagnosticReportError(f"row {row.row_id} invalid trajectory status {status!r}")
    identity_keys = (
        "stage",
        "run_id",
        "attempt_id",
        "checkpoint_sha256",
        "config_sha256",
        "observable",
        "evaluator_id",
    )
    mismatches = [
        f"{key}: receipt={receipt.get(key)!r}, expected={expected_identity.get(key)!r}"
        for key in identity_keys
        if receipt.get(key) != expected_identity.get(key)
    ]
    if receipt.get("estimator_id") != "pooled_geyer_ips" or receipt.get(
        "estimator_version"
    ) != "1":
        mismatches.append("trajectory estimator identity is not pooled_geyer_ips/v1")
    statistics = receipt.get("statistics")
    reason = receipt.get("reason")
    if status == "available":
        if not isinstance(statistics, Mapping) or reason is not None:
            mismatches.append("available trajectory receipt requires statistics and no reason")
    elif statistics is not None or not str(reason or "").strip():
        mismatches.append(f"{status} trajectory receipt requires reason and no statistics")
    shape = receipt.get("shape")
    if status != "absent":
        expected_shape = {
            "walker_count": row.planned["n_walkers"],
            "draw_count": row.planned["n_draws"],
            "total_draws": row.planned["record_capacity"],
            "draw_stride": row.planned["stride"],
            "burn_in_draws": row.planned["burn_in"],
        }
        if not isinstance(shape, Mapping) or any(
            shape.get(key) != value for key, value in expected_shape.items()
        ):
            mismatches.append(f"trajectory shape disagrees with planned row: {shape!r}")
    if mismatches:
        raise DiagnosticReportError(
            f"row {row.row_id} trajectory receipt verification failed: " + "; ".join(mismatches)
        )


def _distribution_and_conditioned_rows(
    study: VerifiedStudy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [
        row
        for row in study.rows
        if row.planned["comparison_kind"] == "primary_headline"
    ]
    if len(primary) != 8:
        raise DiagnosticReportError(f"primary distribution requires 8 rows, found {len(primary)}")
    histogram_edges = np.linspace(-8.0, 8.0, 161)
    distributions: list[dict[str, Any]] = []
    conditioned_rows: list[dict[str, Any]] = []
    for checkpoint in EXPECTED_CHECKPOINTS:
        selected = [row for row in primary if row.checkpoint_label == checkpoint]
        counts = np.zeros(histogram_edges.size - 1, dtype=np.int64)
        total_rows = 0
        finite_rows = 0
        nonfinite_rows = 0
        underflow = 0
        overflow = 0
        artifacts: list[VerifiedArtifact] = []
        conditioned_payloads: list[tuple[VerifiedRow, dict[str, Any], VerifiedArtifact]] = []
        for row in selected:
            sampled = row.artifact(task="retained_energy", name="sampled_eval_table")
            if (
                sampled.metadata.get("rows") != row.planned["record_capacity"]
                or sampled.metadata.get("truncated") is not False
                or sampled.metadata.get("selection") != "complete_draw_walker_grid"
            ):
                raise DiagnosticReportError(f"row {row.row_id} sampled record is not complete")
            artifacts.append(sampled)
            row_count = 0
            with mmap_csv_rows(
                sampled,
                required_fields=(
                    "sample_index",
                    "draw_index",
                    "walker_index",
                    "local_energy",
                    "finite",
                ),
            ) as (_fieldnames, records):
                for record in records:
                    sample_index = int(record["sample_index"])
                    if sample_index != row_count:
                        raise DiagnosticReportError(
                            f"row {row.row_id} sampled record is not contiguous at {row_count}"
                        )
                    if int(record["draw_index"]) != row_count // int(row.planned["n_walkers"]):
                        raise DiagnosticReportError(f"row {row.row_id} draw-major grid changed")
                    if int(record["walker_index"]) != row_count % int(row.planned["n_walkers"]):
                        raise DiagnosticReportError(f"row {row.row_id} walker grid changed")
                    energy = float(record["local_energy"])
                    finite = _bool_text(record["finite"])
                    if finite != math.isfinite(energy):
                        raise DiagnosticReportError(
                            f"row {row.row_id} finite flag disagrees with local energy"
                        )
                    row_count += 1
                    total_rows += 1
                    if not finite:
                        nonfinite_rows += 1
                        continue
                    finite_rows += 1
                    delta = energy - plot_stage.REFERENCE_ENERGY_HA
                    transformed = math.copysign(math.log10(1.0 + abs(delta)), delta)
                    if transformed < histogram_edges[0]:
                        underflow += 1
                        index = 0
                    elif transformed >= histogram_edges[-1]:
                        overflow += 1
                        index = counts.size - 1
                    else:
                        index = int(np.searchsorted(histogram_edges, transformed, side="right") - 1)
                    counts[index] += 1
            if row_count != row.planned["record_capacity"]:
                raise DiagnosticReportError(
                    f"row {row.row_id} sampled record count {row_count} != planned "
                    f"{row.planned['record_capacity']}"
                )
            conditioned_artifact = row.artifact(
                task="retained_energy", name="conditioned_local_energy"
            )
            payload = _read_json_artifact(conditioned_artifact)
            _validate_conditioned_payload(row, payload, sampled=sampled)
            artifacts.append(conditioned_artifact)
            conditioned_payloads.append((row, payload, conditioned_artifact))
        if finite_rows <= 0:
            raise DiagnosticReportError(f"checkpoint {checkpoint} has no finite retained energies")
        provenance = _provenance(
            study,
            checkpoint_label=checkpoint,
            checkpoint_model_sha256=str(selected[0].planned["checkpoint_model_sha256"]),
            source_row_ids=[row.row_id for row in selected],
            artifact_sha256=[artifact.sha256 for artifact in artifacts],
        )
        for index, count in enumerate(counts.tolist()):
            distributions.append(
                {
                    **provenance,
                    "view": "histogram",
                    "bin_left_transformed": float(histogram_edges[index]),
                    "bin_right_transformed": float(histogram_edges[index + 1]),
                    "count": count,
                    "probability": count / finite_rows,
                    "finite_record_count": finite_rows,
                    "nonfinite_record_count": nonfinite_rows,
                    "underflow_count": underflow,
                    "overflow_count": overflow,
                    "record_reader": "verified_read_only_mmap",
                }
            )
        ccdf_totals: dict[float, tuple[int, int]] = {}
        for _row, payload, _artifact in conditioned_payloads:
            finite_count = int(payload["global"]["finite_local_energy_count"])
            for record in payload["rare_events"]["absolute_deviation_ccdf"]:
                threshold = _finite_float(record["threshold"], "CCDF threshold")
                count = int(record["count"])
                old_count, old_total = ccdf_totals.get(threshold, (0, 0))
                ccdf_totals[threshold] = (old_count + count, old_total + finite_count)
        for threshold, (count, denominator) in sorted(ccdf_totals.items()):
            distributions.append(
                {
                    **provenance,
                    "view": "ccdf",
                    "threshold_ha": threshold,
                    "count": count,
                    "finite_record_count": denominator,
                    "probability": count / denominator,
                    "estimator_role": "diagnostic_centered_ccdf_not_headline_energy",
                }
            )
        conditioned_rows.extend(
            _aggregate_conditioned(study, checkpoint, conditioned_payloads)
        )
    return distributions, conditioned_rows


def _validate_conditioned_payload(
    row: VerifiedRow,
    payload: Mapping[str, Any],
    *,
    sampled: VerifiedArtifact,
) -> None:
    source = payload.get("source")
    estimator = payload.get("estimator")
    partitions = payload.get("range_conditioned")
    mismatches: list[str] = []
    if payload.get("schema") != CONDITIONED_SCHEMA:
        mismatches.append(f"unexpected schema {payload.get('schema')!r}")
    if not isinstance(source, Mapping):
        mismatches.append("source identity missing")
    else:
        expected = {
            "trajectory_record_schema": TRAJECTORY_SCHEMA,
            "csv_sha256": sampled.sha256,
            "byte_count": sampled.byte_count,
            "row_count": row.planned["record_capacity"],
            "draw_count": row.planned["n_draws"],
            "walker_count": row.planned["n_walkers"],
            "two_pass_identity_confirmed": True,
        }
        for key, value in expected.items():
            if source.get(key) != value:
                mismatches.append(f"source.{key}={source.get(key)!r}, expected={value!r}")
        for pass_name in ("statistics_pass", "rare_events_pass"):
            pass_receipt = source.get(pass_name)
            if not isinstance(pass_receipt, Mapping) or any(
                pass_receipt.get(key) != value
                for key, value in {
                    "csv_sha256": sampled.sha256,
                    "byte_count": sampled.byte_count,
                    "row_count": row.planned["record_capacity"],
                }.items()
            ):
                mismatches.append(f"source.{pass_name} does not bind the sampled artifact")
    if not isinstance(estimator, Mapping) or estimator.get("headline_estimator") is not False:
        mismatches.append("conditioned estimator must be explicitly non-headline")
    if not isinstance(partitions, Mapping) or set(partitions) != set(
        REQUIRED_CONDITION_QUANTITIES
    ):
        mismatches.append(
            f"conditioned quantities changed: {tuple(partitions) if isinstance(partitions, Mapping) else None}"
        )
    else:
        global_record = payload.get("global")
        configuration = payload.get("configuration")
        rare_events = payload.get("rare_events")
        if not isinstance(global_record, Mapping):
            raise DiagnosticReportError(f"row {row.row_id} conditioned global record is missing")
        if not isinstance(configuration, Mapping):
            raise DiagnosticReportError(
                f"row {row.row_id} conditioned configuration is missing"
            )
        if not isinstance(rare_events, Mapping):
            raise DiagnosticReportError(f"row {row.row_id} conditioned rare events are missing")
        global_count = int(global_record.get("finite_local_energy_count", -1))
        global_nonfinite = int(global_record.get("nonfinite_local_energy_count", -1))
        if global_count <= 0 or global_count + global_nonfinite != row.planned["record_capacity"]:
            mismatches.append(
                "global finite/nonfinite counts do not reconcile to the complete record"
            )
        global_variance = _finite_float(
            global_record["second_moment_about_mean"], "conditioned global variance"
        )
        range_edges = configuration.get("range_edges")
        if not isinstance(range_edges, Mapping) or set(range_edges) != set(
            REQUIRED_CONDITION_QUANTITIES
        ):
            mismatches.append("predeclared range-edge quantities changed")
            range_edges = {}
        for quantity in REQUIRED_CONDITION_QUANTITIES:
            partition = partitions[quantity]
            if not isinstance(partition, Mapping):
                mismatches.append(f"{quantity} partition is not a mapping")
                continue
            if partition.get("structural_bins") != ["underflow", "overflow", "nonfinite"]:
                mismatches.append(f"{quantity} structural bins changed")
                continue
            raw_edges = range_edges.get(quantity)
            if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
                mismatches.append(f"{quantity} predeclared edges are missing")
                continue
            edges = [_finite_float(value, f"{quantity} edge") for value in raw_edges]
            if not edges or any(left >= right for left, right in zip(edges, edges[1:])):
                mismatches.append(f"{quantity} predeclared edges are not strictly increasing")
                continue
            if partition.get("predeclared_edges") != raw_edges:
                mismatches.append(f"{quantity} partition edges differ from configuration")
            bins = partition.get("bins")
            if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
                mismatches.append(f"{quantity} bins missing")
                continue
            expected_bins = [
                ("underflow", "underflow", "-inf", edges[0]),
                *[
                    (f"range_{index:03d}", "range", lower, upper)
                    for index, (lower, upper) in enumerate(zip(edges, edges[1:]))
                ],
                ("overflow", "overflow", edges[-1], "inf"),
                ("nonfinite", "nonfinite", None, None),
            ]
            if len(bins) != len(expected_bins):
                mismatches.append(
                    f"{quantity} bin count {len(bins)} != predeclared {len(expected_bins)}"
                )
                continue
            finite_count_sum = 0
            probability_values: list[float] = []
            contribution_values: list[float] = []
            for record, expected_bin in zip(bins, expected_bins, strict=True):
                if not isinstance(record, Mapping):
                    mismatches.append(f"{quantity} bin is not a mapping")
                    continue
                expected_id, expected_kind, expected_lower, expected_upper = expected_bin
                if (
                    record.get("id"),
                    record.get("kind"),
                    record.get("lower"),
                    record.get("upper"),
                ) != expected_bin:
                    mismatches.append(
                        f"{quantity} bin {record.get('id')!r} differs from "
                        f"{(expected_id, expected_kind, expected_lower, expected_upper)!r}"
                    )
                observables = record.get("observables")
                attribution = record.get("variance_attribution")
                local_energy = (
                    observables.get("local_energy")
                    if isinstance(observables, Mapping)
                    else None
                )
                if not isinstance(local_energy, Mapping) or not isinstance(
                    attribution, Mapping
                ):
                    mismatches.append(f"{quantity}.{expected_id} lacks reconciliation data")
                    continue
                finite_count = int(local_energy.get("finite_count", -1))
                probability = _finite_float(
                    attribution.get("probability"), f"{quantity} probability"
                )
                contribution = _finite_float(
                    attribution.get("second_moment_contribution"),
                    f"{quantity} variance contribution",
                )
                if finite_count < 0:
                    mismatches.append(f"{quantity}.{expected_id} has a negative finite count")
                expected_probability = finite_count / global_count if global_count > 0 else 0.0
                if not math.isclose(
                    probability,
                    expected_probability,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    mismatches.append(
                        f"{quantity}.{expected_id} probability {probability} "
                        f"!= count fraction {expected_probability}"
                    )
                finite_count_sum += finite_count
                probability_values.append(probability)
                contribution_values.append(contribution)
            probabilities = math.fsum(probability_values)
            contributions = math.fsum(contribution_values)
            reconciliation = partition.get("reconciliation")
            if not isinstance(reconciliation, Mapping):
                mismatches.append(f"{quantity} reconciliation is missing")
                continue
            expected_reconciliation = {
                "finite_count_sum": finite_count_sum,
                "global_finite_count": global_count,
            }
            for key, expected_value in expected_reconciliation.items():
                if reconciliation.get(key) != expected_value:
                    mismatches.append(
                        f"{quantity}.{key}={reconciliation.get(key)!r}, "
                        f"computed={expected_value!r}"
                    )
            if finite_count_sum != global_count:
                mismatches.append(f"{quantity} finite bin counts do not cover the global count")
            for key, computed, expected_value in (
                ("probability_sum", probabilities, 1.0),
                ("second_moment_contribution_sum", contributions, global_variance),
                ("global_second_moment", global_variance, global_variance),
            ):
                recorded = _finite_float(reconciliation.get(key), f"{quantity}.{key}")
                if not math.isclose(recorded, computed, rel_tol=1.0e-10, abs_tol=1.0e-12):
                    mismatches.append(f"{quantity}.{key} does not match its bins")
                if not math.isclose(recorded, expected_value, rel_tol=1.0e-10, abs_tol=1.0e-12):
                    mismatches.append(f"{quantity}.{key} does not match the global value")
            if not math.isclose(contributions, global_variance, rel_tol=1.0e-9, abs_tol=1.0e-12):
                mismatches.append(
                    f"{quantity} variance attribution {contributions} != {global_variance}"
                )
        raw_thresholds = configuration.get("deviation_ccdf_thresholds")
        ccdf_records = rare_events.get("absolute_deviation_ccdf")
        if (
            not isinstance(raw_thresholds, Sequence)
            or isinstance(raw_thresholds, (str, bytes))
            or not isinstance(ccdf_records, Sequence)
            or isinstance(ccdf_records, (str, bytes))
            or len(raw_thresholds) != len(ccdf_records)
        ):
            mismatches.append("absolute-deviation CCDF grid is incomplete")
        else:
            thresholds = [_finite_float(value, "CCDF threshold") for value in raw_thresholds]
            if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
                mismatches.append("CCDF thresholds are not strictly increasing")
            previous_count = global_count
            for threshold, record in zip(thresholds, ccdf_records, strict=True):
                if not isinstance(record, Mapping):
                    mismatches.append("CCDF record is not a mapping")
                    continue
                count = int(record.get("count", -1))
                probability = _finite_float(
                    record.get("probability_over_finite_local_energy"),
                    "CCDF probability",
                )
                if not math.isclose(
                    _finite_float(record.get("threshold"), "CCDF record threshold"),
                    threshold,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    mismatches.append("CCDF record threshold differs from configuration")
                if count < 0 or count > previous_count:
                    mismatches.append("CCDF counts are not bounded and nonincreasing")
                if not math.isclose(
                    probability,
                    count / global_count,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    mismatches.append("CCDF probability does not match its count")
                previous_count = count
    if mismatches:
        raise DiagnosticReportError(
            f"row {row.row_id} conditioned artifact verification failed: " + "; ".join(mismatches)
        )


def _aggregate_conditioned(
    study: VerifiedStudy,
    checkpoint: str,
    payloads: Sequence[tuple[VerifiedRow, Mapping[str, Any], VerifiedArtifact]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance = _provenance(
        study,
        checkpoint_label=checkpoint,
        checkpoint_model_sha256=str(payloads[0][0].planned["checkpoint_model_sha256"]),
        source_row_ids=[item[0].row_id for item in payloads],
        artifact_sha256=[item[2].sha256 for item in payloads],
    )
    for quantity in REQUIRED_CONDITION_QUANTITIES:
        bins_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for _row, payload, _artifact in payloads:
            for record in payload["range_conditioned"][quantity]["bins"]:
                bins_by_id[str(record["id"])].append(record)
        for bin_id, records in bins_by_id.items():
            first = records[0]
            attribution = [record["variance_attribution"] for record in records]
            lower = first.get("lower")
            upper = first.get("upper")
            rows.append(
                {
                    **provenance,
                    "quantity": quantity,
                    "bin_id": bin_id,
                    "bin_kind": "finite" if first.get("kind") == "range" else first.get("kind"),
                    "bin_label": _bin_label(lower, upper),
                    "lower": lower,
                    "upper": upper,
                    "probability": float(
                        np.mean(
                            [
                                _finite_float(value["probability"], "condition probability")
                                for value in attribution
                            ]
                        )
                    ),
                    "second_moment_contribution_ha2": float(
                        np.mean(
                            [
                                _finite_float(
                                    value["second_moment_contribution"],
                                    "variance contribution",
                                )
                                for value in attribution
                            ]
                        )
                    ),
                    "estimator_role": "diagnostic_variance_attribution_not_headline",
                }
            )
    return rows


def _bin_label(lower: Any, upper: Any) -> str:
    if lower is None and upper is None:
        return "nonfinite"
    if lower == "-inf":
        return f"< {float(upper):g}"
    if upper == "inf":
        return f"≥ {float(lower):g}"
    return f"{float(lower):g}–{float(upper):g}"


def _atlas_rows(
    study: VerifiedStudy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostic_rows = [
        row for row in study.rows if row.planned["profile"] == "checkpoint_diagnostics"
    ]
    if len(diagnostic_rows) != 2:
        raise DiagnosticReportError(
            f"atlas reporting requires two checkpoint diagnostic rows, found {len(diagnostic_rows)}"
        )
    cusp_rows: list[dict[str, Any]] = []
    cancellation_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        base = {
            "checkpoint_label": row.checkpoint_label,
            "checkpoint_model_sha256": str(row.planned["checkpoint_model_sha256"]),
        }
        radial = row.artifact(
            task="he_radial_profiles", name="electron_nucleus_radial_profile"
        )
        radial_metrics = row.task_metrics("he_radial_profiles")
        cusp_available = radial_metrics.get("cusp_available") is True
        cusp_rows.append(
            {
                **_provenance(
                    study,
                    source_row_ids=[row.row_id],
                    artifact_sha256=[radial.sha256],
                    **base,
                ),
                "view": "targeted_cusp_summary",
                "available": cusp_available,
                "radius_bohr": "not_applicable",
                "first_derivative": (
                    _finite_float(
                        radial_metrics.get("cusp_one_sided_slope_mean"),
                        "targeted cusp slope",
                    )
                    if cusp_available
                    else "unavailable"
                ),
                "ideal_cusp_law": (
                    _finite_float(
                        radial_metrics.get("cusp_expected_slope"),
                        "targeted ideal cusp slope",
                    )
                    if cusp_available
                    else -2.0
                ),
                "unavailable_reason": "not_applicable" if cusp_available else "cusp fit unavailable",
                "estimator_role": "targeted_one_sided_cusp_estimator",
            }
        )

        en = row.artifact(task="he_en_numerical_atlas", name="helium_atlas")
        ee = row.artifact(
            task="he_ee_ideal_vs_executed_numerical_atlas", name="helium_atlas"
        )
        cusp_rows.extend(
            _atlas_series_rows(
                study,
                row,
                en,
                view="electron_nucleus",
                series="executed_electron_nucleus_factor",
                derivative="first_derivative",
            )
        )
        cusp_rows.extend(
            _atlas_series_rows(
                study,
                row,
                ee,
                view="electron_electron",
                series="executed_smoothed_ee_factor",
                derivative="first_derivative",
            )
        )
        cusp_rows.extend(
            _atlas_series_rows(
                study,
                row,
                en,
                view="curvature",
                series="executed_full_logabs",
                derivative="second_derivative",
            )
        )
        for view, derivative, ideal_value, artifact in (
            ("electron_nucleus", "first_derivative", -2.0, en),
            ("electron_electron", "first_derivative", 0.5, ee),
        ):
            cusp_rows.append(
                {
                    **_provenance(
                        study,
                        checkpoint_label=row.checkpoint_label,
                        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                        source_row_ids=[row.row_id],
                        artifact_sha256=[artifact.sha256],
                    ),
                    "view": view,
                    "series": "analytic_ideal_cusp_law",
                    "available": True,
                    "radius_bohr": "all_positive_radii",
                    derivative: ideal_value,
                    "reference_kind": "analytic_exact",
                    "target_semantics": "universal_kato_first_derivative_target",
                    "unavailable_reason": "not_applicable",
                }
            )
        cusp_rows.append(
            {
                **_provenance(
                    study,
                    checkpoint_label=row.checkpoint_label,
                    checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                    source_row_ids=[row.row_id],
                    artifact_sha256=[en.sha256],
                ),
                "view": "curvature",
                "series": "no_analytic_curvature_target",
                "available": False,
                "radius_bohr": "not_applicable",
                "second_derivative": "not_applicable",
                "reference_kind": "not_applicable",
                "target_semantics": "no_universal_kato_curvature_target",
                "unavailable_reason": "no universal Kato curvature target exists",
            }
        )
        cancellation_rows.extend(_cancellation_series_rows(study, row, ee))

        for task, view in (
            ("he_one_electron_tail_atlas", "one_electron"),
            ("he_center_of_mass_tail_atlas", "center_of_mass"),
        ):
            artifact = row.artifact(task=task, name="helium_atlas")
            metrics = row.task_metrics(task)
            prefix = (
                "executed_full_logabs_one_electron_tail"
                if view == "one_electron"
                else "executed_full_logabs_center_of_mass_tail"
            )
            available = metrics.get(f"{prefix}_available") is True
            tail_rows.extend(
                _tail_series_rows(
                    study,
                    row,
                    artifact,
                    view=view,
                    available=available,
                    outer_slope=(
                        _finite_float(metrics.get(f"{prefix}_slope"), "outer tail slope")
                        if available
                        else None
                    ),
                )
            )
    return cusp_rows, cancellation_rows, tail_rows


def _atlas_series_rows(
    study: VerifiedStudy,
    row: VerifiedRow,
    artifact: VerifiedArtifact,
    *,
    view: str,
    series: str,
    derivative: str,
) -> list[dict[str, Any]]:
    value_field = f"{series}_{derivative}"
    finite_field = f"{value_field}_finite"
    grouped: dict[float, list[float]] = defaultdict(list)
    with mmap_csv_rows(
        artifact,
        required_fields=(
            "realized_physical_coordinate",
            "is_exact_zero_sentinel",
            value_field,
            finite_field,
        ),
    ) as (_fieldnames, records):
        for record in records:
            if _bool_text(record["is_exact_zero_sentinel"]):
                continue
            radius = float(record["realized_physical_coordinate"])
            if not math.isfinite(radius) or radius <= 0.0:
                continue
            if not _bool_text(record[finite_field]):
                continue
            value = _finite_float(record[value_field], value_field)
            grouped[radius].append(value)
    provenance = _provenance(
        study,
        checkpoint_label=row.checkpoint_label,
        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
        source_row_ids=[row.row_id],
        artifact_sha256=[artifact.sha256],
    )
    if not grouped:
        return [
            {
                **provenance,
                "view": view,
                "series": series,
                "available": False,
                "radius_bohr": "unavailable",
                derivative: "unavailable",
                "unavailable_reason": "no finite executed atlas derivatives",
            }
        ]
    return [
        {
            **provenance,
            "view": view,
            "series": series,
            "available": True,
            "radius_bohr": radius,
            derivative: float(np.mean(values)),
            "direction_count": len(values),
            "unavailable_reason": "not_applicable",
            "law_semantics": (
                "executed_smoothed_factor"
                if view == "electron_electron"
                else "executed_factor"
            ),
            "target_semantics": (
                "no_universal_kato_curvature_target"
                if view == "curvature"
                else "universal_kato_first_derivative_target"
            ),
        }
        for radius, values in sorted(grouped.items())
    ]


def _cancellation_series_rows(
    study: VerifiedStudy,
    row: VerifiedRow,
    artifact: VerifiedArtifact,
) -> list[dict[str, Any]]:
    fields = (
        "realized_physical_coordinate",
        "is_exact_zero_sentinel",
        "executed_hamiltonian_cancellation_abs_sum",
        "executed_hamiltonian_cancellation_abs_sum_finite",
        "executed_hamiltonian_cancellation_residual",
        "executed_hamiltonian_cancellation_residual_finite",
        "executed_hamiltonian_cancellation_ratio",
        "executed_hamiltonian_cancellation_ratio_finite",
    )
    grouped: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"abs_sum": [], "residual": [], "ratio": []}
    )
    with mmap_csv_rows(artifact, required_fields=fields) as (_fieldnames, records):
        for record in records:
            if _bool_text(record["is_exact_zero_sentinel"]):
                continue
            radius = float(record["realized_physical_coordinate"])
            if not math.isfinite(radius) or radius <= 0.0:
                continue
            for short, column in (
                ("abs_sum", "executed_hamiltonian_cancellation_abs_sum"),
                ("residual", "executed_hamiltonian_cancellation_residual"),
                ("ratio", "executed_hamiltonian_cancellation_ratio"),
            ):
                if _bool_text(record[f"{column}_finite"]):
                    grouped[radius][short].append(_finite_float(record[column], column))
    provenance = _provenance(
        study,
        checkpoint_label=row.checkpoint_label,
        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
        source_row_ids=[row.row_id],
        artifact_sha256=[artifact.sha256],
    )
    output = []
    for radius, values in sorted(grouped.items()):
        available = all(values[key] for key in values)
        output.append(
            {
                **provenance,
                "geometry": "electron_electron_coalescence",
                "available": available,
                "radius_bohr": radius,
                "cancellation_abs_sum_ha": (
                    float(np.mean(values["abs_sum"])) if available else "unavailable"
                ),
                "cancellation_residual_ha": (
                    float(np.mean(values["residual"])) if available else "unavailable"
                ),
                "cancellation_ratio": (
                    float(np.mean(values["ratio"])) if available else "unavailable"
                ),
                "unavailable_reason": (
                    "not_applicable" if available else "nonfinite executed cancellation fields"
                ),
            }
        )
    if not output:
        output.append(
            {
                **provenance,
                "geometry": "electron_electron_coalescence",
                "available": False,
                "radius_bohr": "unavailable",
                "cancellation_abs_sum_ha": "unavailable",
                "cancellation_residual_ha": "unavailable",
                "cancellation_ratio": "unavailable",
                "unavailable_reason": "no positive-radius atlas rows",
            }
        )
    return output


def _tail_series_rows(
    study: VerifiedStudy,
    row: VerifiedRow,
    artifact: VerifiedArtifact,
    *,
    view: str,
    available: bool,
    outer_slope: float | None,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    fields = (
        "realized_physical_coordinate",
        "executed_full_logabs_value",
        "executed_full_logabs_value_finite",
    )
    with mmap_csv_rows(artifact, required_fields=fields) as (_fieldnames, records):
        for record in records:
            radius = float(record["realized_physical_coordinate"])
            if not math.isfinite(radius) or radius <= 0.0:
                continue
            if _bool_text(record["executed_full_logabs_value_finite"]):
                grouped[radius].append(
                    _finite_float(record["executed_full_logabs_value"], "tail logabs")
                )
    provenance = _provenance(
        study,
        checkpoint_label=row.checkpoint_label,
        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
        source_row_ids=[row.row_id],
        artifact_sha256=[artifact.sha256],
    )
    if not available or not grouped or outer_slope is None:
        return [
            {
                **provenance,
                "view": view,
                "available": False,
                "radius_bohr": "unavailable",
                "executed_logabs": "unavailable",
                "outer_slope_bohr_inv": "unavailable",
                "unavailable_reason": "outer tail summary unavailable",
            }
        ]
    return [
        {
            **provenance,
            "view": view,
            "available": True,
            "radius_bohr": radius,
            "executed_logabs": float(np.mean(values)),
            "outer_slope_bohr_inv": outer_slope,
            "unavailable_reason": "not_applicable",
        }
        for radius, values in sorted(grouped.items())
    ]


def _symmetry_rows(study: VerifiedStudy) -> list[dict[str, Any]]:
    specs = (
        (
            "full_model_antisymmetry",
            "logabs_max_abs_error",
            "full-model |Δlog| max",
            "full-label antisymmetry; triplet fraction 1 is healthy and is not used",
        ),
        (
            "full_model_antisymmetry",
            "sign_failure_count",
            "full-model sign failures",
            "full-label antisymmetry invariant",
        ),
        (
            "spatial_exchange_symmetry",
            "triplet_fraction_mean_under_psi_orig_sq",
            "spatial triplet fraction",
            "singlet purity is interpretable only for spatial coordinate exchange",
        ),
        (
            "rotation_consistency",
            "local_energy_max_abs_error",
            "rotation |ΔE| max",
            "rotation consistency",
        ),
        (
            "trace_equivariance",
            "max_abs_error",
            "trace equivariance max",
            "typed trace equivariance comparison",
        ),
        (
            "trace_equivariance",
            "comparison_error_count",
            "trace comparison errors",
            "typed trace comparison count",
        ),
        (
            "feature_trace",
            "feature_nonfinite_count",
            "feature nonfinite count",
            "feature-trace health diagnostic",
        ),
        (
            "readout_trace",
            "readout_nonfinite_count",
            "readout nonfinite count",
            "readout-trace health diagnostic",
        ),
    )
    diagnostics = [
        row for row in study.rows if row.planned["profile"] == "checkpoint_diagnostics"
    ]
    output: list[dict[str, Any]] = []
    for row in diagnostics:
        for task, key, label, semantics in specs:
            metrics = row.task_metrics(task)
            artifacts = [artifact for artifact in row.artifacts if artifact.task == task]
            if not artifacts:
                raise DiagnosticReportError(f"row {row.row_id} task {task} has no record artifact")
            value = metrics.get(key)
            available = isinstance(value, (int, float)) and not isinstance(value, bool)
            output.append(
                {
                    **_provenance(
                        study,
                        checkpoint_label=row.checkpoint_label,
                        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                        source_row_ids=[row.row_id],
                        artifact_sha256=[artifact.sha256 for artifact in artifacts],
                    ),
                    "task": task,
                    "metric_key": key,
                    "metric_label": label,
                    "available": available,
                    "value": _finite_float(value, f"{task}.{key}") if available else "unavailable",
                    "semantics": semantics,
                    "unavailable_reason": "not_applicable" if available else "metric not emitted",
                }
            )
    return output


def _health_timing_rows(
    study: VerifiedStudy,
    energy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    energy_by_id = {str(row["row_id"]): row for row in energy_rows}
    primary = [
        row
        for row in study.rows
        if row.planned["comparison_kind"] == "primary_headline"
    ]
    output: list[dict[str, Any]] = []
    for row in primary:
        metrics = row.task_metrics("retained_energy")
        health_artifact = row.artifact(
            task="retained_energy", name="sampler_trajectory_diagnostics"
        )
        health_payload = _read_json_artifact(health_artifact)
        if health_payload.get("schema") != "sampler_trajectory_diagnostics/v1":
            raise DiagnosticReportError(f"row {row.row_id} sampler-health schema changed")
        if (
            health_payload.get("n_walkers") != row.planned["n_walkers"]
            or health_payload.get("draw_stride") != row.planned["stride"]
            or len(health_payload.get("retained_draws", ())) != row.planned["n_draws"]
        ):
            raise DiagnosticReportError(f"row {row.row_id} sampler-health shape changed")
        acceptance = metrics.get("sampler_trajectory_retained_draw_acceptance_rate_mean")
        perf = row.metrics_by_namespace["eval/perf"].get("wall_time_sec")
        runtime = row.metrics_by_namespace["runtime"]
        energy = energy_by_id[row.row_id]
        statistics_artifact = row.artifact(
            task="retained_energy", name="local_energy_trajectory_statistics"
        )
        acceptance_available = isinstance(acceptance, (int, float)) and not isinstance(
            acceptance, bool
        )
        wall_available = isinstance(perf, (int, float)) and not isinstance(perf, bool)
        ess_available = energy["status"] == "available"
        acceptance_value = (
            _finite_float(acceptance, "retained acceptance")
            if acceptance_available
            else None
        )
        if acceptance_value is not None and not 0.0 <= acceptance_value <= 1.0:
            raise DiagnosticReportError(f"row {row.row_id} acceptance rate is outside [0, 1]")
        wall = _finite_float(perf, "evaluation wall time") if wall_available else None
        if wall is not None and wall < 0.0:
            raise DiagnosticReportError(f"row {row.row_id} evaluation wall time is negative")
        ess = _finite_float(energy["ess"], "ESS") if ess_available else None
        peak = runtime.get("peak_memory_mb")
        cuda_peak = runtime.get("cuda_max_memory_allocated_mb")
        peak_available = isinstance(peak, (int, float)) and not isinstance(peak, bool)
        cuda_peak_available = isinstance(cuda_peak, (int, float)) and not isinstance(
            cuda_peak, bool
        )
        unavailable_fields = []
        if not acceptance_available:
            unavailable_fields.append("acceptance_rate: metric not emitted")
        if not wall_available:
            unavailable_fields.append("wall_time_sec: metric not emitted")
        if not ess_available:
            unavailable_fields.append(
                f"ess: {energy.get('unavailable_reason') or energy['status']}"
            )
        if wall == 0.0:
            unavailable_fields.append("ess_per_second: wall time is zero")
        if not peak_available:
            unavailable_fields.append("peak_rss_mb: metric not emitted")
        if not cuda_peak_available:
            unavailable_fields.append("cuda_max_memory_allocated_mb: metric not emitted")
        output.append(
            {
                **_provenance(
                    study,
                    checkpoint_label=row.checkpoint_label,
                    checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                    source_row_ids=[row.row_id],
                    artifact_sha256=[
                        health_artifact.sha256,
                        statistics_artifact.sha256,
                    ],
                ),
                "row_id": row.row_id,
                "estimator_role": "primary",
                "acceptance_rate_available": acceptance_available,
                "acceptance_rate": (
                    acceptance_value if acceptance_value is not None else "unavailable"
                ),
                "wall_time_sec_available": wall_available,
                "wall_time_sec": wall if wall is not None else "unavailable",
                "ess_available": ess_available,
                "ess": ess if ess is not None else "unavailable",
                "ess_per_second_available": ess is not None and wall is not None and wall > 0.0,
                "ess_per_second": (
                    ess / wall
                    if ess is not None and wall is not None and wall > 0.0
                    else "unavailable"
                ),
                "peak_rss_mb_available": peak_available,
                "peak_rss_mb": (
                    _finite_float(peak, "peak RSS") if peak_available else "unavailable"
                ),
                "cuda_max_memory_allocated_mb_available": cuda_peak_available,
                "cuda_max_memory_allocated_mb": (
                    _finite_float(cuda_peak, "peak CUDA allocation")
                    if cuda_peak_available
                    else "unavailable"
                ),
                "unavailable_reason": (
                    "not_applicable" if not unavailable_fields else "; ".join(unavailable_fields)
                ),
            }
        )
    return output


def _factor_rows(study: VerifiedStudy) -> list[dict[str, Any]]:
    common_rows = [
        row for row in study.rows if row.planned["profile"] == "common_factor_response"
    ]
    reequilibrated = [
        row for row in study.rows if row.planned["profile"] == "reequilibrated_energy"
    ]
    if len(common_rows) != 2 or len(reequilibrated) != 14:
        raise DiagnosticReportError(
            "factor response requires two common rows and fourteen re-equilibrated rows"
        )
    output: list[dict[str, Any]] = []
    for row in common_rows:
        artifact = row.artifact(
            task="common_factor_response", name="factor_response_common_configuration"
        )
        metadata = artifact.metadata
        if (
            metadata.get("comparison_kind") != "common_configuration"
            or metadata.get("baseline_label") != "baseline"
            or metadata.get("selection") != "complete_common_configuration_grid"
            or metadata.get("model_state_restored") is not True
            or metadata.get("arm_count") != len(EXPECTED_FACTOR_ARMS)
            or metadata.get("rows")
            != int(row.planned["record_capacity"]) * len(EXPECTED_FACTOR_ARMS)
        ):
            raise DiagnosticReportError(f"row {row.row_id} common factor metadata changed")
        aggregates: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "finite": 0,
                "delta_energy": 0.0,
                "delta_energy_square": 0.0,
                "delta_logabs": 0.0,
                "parameter_scales": None,
            }
        )
        with mmap_csv_rows(
            artifact,
            required_fields=(
                "arm",
                "sample_index",
                "comparison_kind",
                "parameter_scales",
                "local_energy",
                "delta_local_energy_from_baseline",
                "delta_logabs_from_baseline",
                "finite",
            ),
        ) as (_fieldnames, records):
            for record in records:
                arm = record["arm"]
                if arm not in EXPECTED_FACTOR_ARMS:
                    raise DiagnosticReportError(f"row {row.row_id} unknown common factor arm {arm}")
                if record["comparison_kind"] != "common_configuration":
                    raise DiagnosticReportError(f"row {row.row_id} common factor row mislabeled")
                bucket = aggregates[arm]
                if int(record["sample_index"]) != bucket["count"]:
                    raise DiagnosticReportError(
                        f"row {row.row_id} factor arm {arm} sample grid is not contiguous"
                    )
                scales = json.loads(record["parameter_scales"])
                if bucket["parameter_scales"] is None:
                    bucket["parameter_scales"] = scales
                elif bucket["parameter_scales"] != scales:
                    raise DiagnosticReportError(f"row {row.row_id} factor scales changed within arm")
                bucket["count"] += 1
                finite = _bool_text(record["finite"])
                if finite:
                    bucket["finite"] += 1
                    delta_energy = _finite_float(
                        record["delta_local_energy_from_baseline"], "paired factor delta energy"
                    )
                    bucket["delta_energy"] += delta_energy
                    bucket["delta_energy_square"] += delta_energy * delta_energy
                    bucket["delta_logabs"] += _finite_float(
                        record["delta_logabs_from_baseline"], "paired factor delta logabs"
                    )
        if tuple(aggregates) != EXPECTED_FACTOR_ARMS:
            raise DiagnosticReportError(
                f"row {row.row_id} common factor arm order changed: {tuple(aggregates)!r}"
            )
        for arm in EXPECTED_FACTOR_ARMS:
            bucket = aggregates[arm]
            if bucket["count"] != row.planned["record_capacity"]:
                raise DiagnosticReportError(f"row {row.row_id} common factor arm {arm} incomplete")
            available = bucket["finite"] > 1
            if available:
                paired_mean = bucket["delta_energy"] / bucket["finite"]
                paired_variance = max(
                    0.0,
                    (
                        bucket["delta_energy_square"]
                        - bucket["finite"] * paired_mean * paired_mean
                    )
                    / (bucket["finite"] - 1),
                )
                paired_iid_stderr = math.sqrt(paired_variance / bucket["finite"])
            else:
                paired_mean = None
                paired_iid_stderr = None
            parameter, scale = _factor_coordinate(arm)
            output.append(
                {
                    **_provenance(
                        study,
                        checkpoint_label=row.checkpoint_label,
                        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                        source_row_ids=[row.row_id],
                        artifact_sha256=[artifact.sha256],
                    ),
                    "arm_label": arm,
                    "factor_parameter": parameter,
                    "factor_scale": scale,
                    "comparison_basis": "fixed_configuration_paired",
                    "status": "available" if available else "unresolved",
                    "delta_energy_ha": (
                        paired_mean if paired_mean is not None else "unavailable"
                    ),
                    "delta_logabs": (
                        bucket["delta_logabs"] / bucket["finite"]
                        if available
                        else "unavailable"
                    ),
                    "arm_mcse_ha": "not_available_for_fixed_configuration_record",
                    "baseline_mcse_ha": "not_available_for_fixed_configuration_record",
                    "delta_mcse_ha": "not_available_for_fixed_configuration_record",
                    "paired_iid_stderr_ha": (
                        paired_iid_stderr
                        if paired_iid_stderr is not None
                        else "unavailable"
                    ),
                    "delta_uncertainty_ha": (
                        paired_iid_stderr
                        if paired_iid_stderr is not None
                        else "unavailable"
                    ),
                    "uncertainty_definition": (
                        "paired IID stderr across fixed configurations; not a correlation-aware MCSE"
                    ),
                    "finite_configuration_count": bucket["finite"],
                    "unavailable_reason": (
                        "not_applicable"
                        if available
                        else "fewer than two finite paired factor rows"
                    ),
                }
            )

    for checkpoint in EXPECTED_CHECKPOINTS:
        selected = [row for row in reequilibrated if row.checkpoint_label == checkpoint]
        by_arm = {str(row.planned["factor_arm"]["label"]): row for row in selected}
        if tuple(by_arm) != EXPECTED_FACTOR_ARMS:
            raise DiagnosticReportError(
                f"checkpoint {checkpoint} re-equilibrated factor order changed: {tuple(by_arm)!r}"
            )
        receipts: dict[str, tuple[Mapping[str, Any], VerifiedArtifact]] = {}
        for arm, row in by_arm.items():
            receipts[arm] = _trajectory_receipt(row, task="reequilibrated_energy")
        baseline = receipts["baseline"][0]
        baseline_available = baseline["status"] == "available"
        baseline_mean = (
            _finite_float(baseline["statistics"]["mean"], "re-equilibrated baseline mean")
            if baseline_available
            else None
        )
        baseline_mcse = (
            _finite_float(baseline["statistics"]["mcse"], "re-equilibrated baseline MCSE")
            if baseline_available
            else None
        )
        for arm in EXPECTED_FACTOR_ARMS:
            row = by_arm[arm]
            receipt, artifact = receipts[arm]
            available = receipt["status"] == "available" and baseline_mean is not None
            statistics = receipt.get("statistics")
            mean = (
                _finite_float(statistics["mean"], "re-equilibrated factor mean")
                if receipt["status"] == "available"
                else None
            )
            arm_mcse = (
                _finite_float(statistics["mcse"], "re-equilibrated arm MCSE")
                if receipt["status"] == "available"
                else None
            )
            delta_mcse = (
                0.0
                if available and arm == "baseline"
                else (
                    math.hypot(arm_mcse, baseline_mcse)
                    if available and arm_mcse is not None and baseline_mcse is not None
                    else None
                )
            )
            parameter, scale = _factor_coordinate(arm)
            output.append(
                {
                    **_provenance(
                        study,
                        checkpoint_label=checkpoint,
                        checkpoint_model_sha256=str(row.planned["checkpoint_model_sha256"]),
                        source_row_ids=[row.row_id, by_arm["baseline"].row_id],
                        artifact_sha256=[artifact.sha256, receipts["baseline"][1].sha256],
                    ),
                    "arm_label": arm,
                    "factor_parameter": parameter,
                    "factor_scale": scale,
                    "comparison_basis": "re_equilibrated_independent",
                    "status": (
                        "available"
                        if available
                        else (
                            receipt["status"]
                            if receipt["status"] != "available"
                            else "unresolved"
                        )
                    ),
                    "delta_energy_ha": (
                        mean - baseline_mean if available and mean is not None else "unavailable"
                    ),
                    "delta_logabs": "not_applicable_independent_chains",
                    "arm_mcse_ha": arm_mcse if arm_mcse is not None else "unavailable",
                    "baseline_mcse_ha": (
                        baseline_mcse if baseline_mcse is not None else "unavailable"
                    ),
                    "delta_mcse_ha": (
                        delta_mcse if delta_mcse is not None else "unavailable"
                    ),
                    "paired_iid_stderr_ha": "not_applicable_independent_chains",
                    "delta_uncertainty_ha": (
                        delta_mcse if delta_mcse is not None else "unavailable"
                    ),
                    "uncertainty_definition": (
                        "quadrature of independent correlation-aware arm and baseline MCSEs"
                        if arm != "baseline"
                        else "identical baseline estimator; delta is exactly zero"
                    ),
                    "finite_configuration_count": (
                        receipt["shape"]["total_draws"]
                        if receipt["status"] != "absent"
                        else "unavailable"
                    ),
                    "unavailable_reason": (
                        "not_applicable"
                        if available
                        else str(receipt.get("reason") or baseline.get("reason"))
                    ),
                }
            )
    return output


def _factor_coordinate(label: str) -> tuple[str, float]:
    if label == "baseline":
        return "baseline", 1.0
    if label.startswith("b_ee_"):
        parameter = "b_ee"
    elif label.startswith("c_en_"):
        parameter = "c_electron_nucleus"
    elif label.startswith("d_en_"):
        parameter = "d_electron_nucleus"
    else:
        raise DiagnosticReportError(f"unknown frozen factor arm {label!r}")
    return parameter, 0.9 if "minus" in label else 1.1


def build_report(
    *,
    results_root: str | Path,
    plan_attempt_id: str,
    collect_attempt_id: str,
    report_attempt_id: str,
) -> dict[str, Any]:
    """Verify all inputs, then write one immutable publication attempt."""

    study = read_verified_study(
        results_root,
        plan_attempt_id=plan_attempt_id,
        collect_attempt_id=collect_attempt_id,
    )
    tables = build_plot_tables(study)
    output_dir = layout.report_attempt_dir(study.results_root, report_attempt_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    table_dir = output_dir / "plot-data"
    figure_dir = output_dir / "figures"
    table_dir.mkdir()
    figure_dir.mkdir()

    table_paths = {
        name: _write_plot_table(table_dir / f"{name}.csv", tables[name])
        for name in TABLE_NAMES
    }
    figure_paths = plot_stage.render_all(figure_dir, tables)
    markdown = render_markdown(
        study,
        report_attempt_id=report_attempt_id,
        tables=tables,
    )
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(markdown, encoding="utf-8")

    source_artifacts = [
        {
            "row_id": row.row_id,
            "task": artifact.task,
            "namespace": artifact.namespace,
            "name": artifact.name,
            "kind": artifact.kind,
            "path": artifact.provenance_path,
            "sha256": artifact.sha256,
            "bytes": artifact.byte_count,
        }
        for row in study.rows
        for artifact in row.artifacts
    ]
    mmap_artifacts = {
        ("retained_energy", "sampled_eval_table"),
        ("common_factor_response", "factor_response_common_configuration"),
        ("he_en_numerical_atlas", "helium_atlas"),
        ("he_ee_ideal_vs_executed_numerical_atlas", "helium_atlas"),
        ("he_one_electron_tail_atlas", "helium_atlas"),
        ("he_center_of_mass_tail_atlas", "helium_atlas"),
    }
    manifest = {
        "schema": REPORT_SCHEMA,
        "study": study.manifest["study"],
        "scale": study.manifest["scale"],
        "report_attempt_id": report_attempt_id,
        "plan_attempt_id": plan_attempt_id,
        "plan_sha256": study.plan_sha256,
        "collect_attempt_id": collect_attempt_id,
        "collected_sha256": study.collected_sha256,
        "evaluation_git_sha": study.evaluation_git_sha,
        "source_created_at": study.collected["created_at"],
        "checkpoint_reporting": "both_without_selection",
        "selection_policy": "none",
        "manual_result_correction": False,
        "record_reader": "stdlib_mmap_read_only_after_sha256_and_size_verification",
        "memory_mapped_record_artifacts": [
            {
                "row_id": item["row_id"],
                "task": item["task"],
                "name": item["name"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for item in source_artifacts
            if (item["task"], item["name"]) in mmap_artifacts
        ],
        "source_artifacts": source_artifacts,
        "plot_data": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _file_sha256(path),
                "rows": len(tables[name]),
            }
            for name, path in table_paths.items()
        ],
        "figures": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
                "dpi": 300 if path.suffix == ".png" else "vector",
            }
            for name in TABLE_NAMES
            for path in figure_paths[name]
        ],
        "markdown": {
            "path": REPORT_FILENAME,
            "sha256": _file_sha256(report_path),
        },
    }
    manifest_path = output_dir / REPORT_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    layout.write_latest(layout.stage_dir(study.results_root, layout.STAGE_REPORT), report_attempt_id)
    return manifest


def _write_plot_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise DiagnosticReportError(f"refusing to write empty plot-data table {path.name}")
    for index, row in enumerate(rows):
        missing = sorted(set(PROVENANCE_KEYS) - set(row))
        if missing:
            raise DiagnosticReportError(
                f"plot-data row {path.name}:{index} lacks provenance fields {missing}"
            )
    all_fields = {str(key) for row in rows for key in row}
    fields = [*PROVENANCE_KEYS, *sorted(all_fields - set(PROVENANCE_KEYS))]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row[field]) if field in row else "not_applicable"
                    for field in fields
                }
            )
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticReportError(f"plot-data tables cannot contain nonfinite value {value!r}")
    return value


def render_markdown(
    study: VerifiedStudy,
    *,
    report_attempt_id: str,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Return the complete report with relative figure and table links."""

    energy = tables["energy_mcse"]
    primary = [row for row in energy if row["estimator_role"] == "primary"]
    unavailable = _unavailable_rows(tables)
    lines = [
        "# He-v1 diagnostic-v1 report",
        "",
        "This report is a deterministic post-hoc rendering. It does not retrain, mutate",
        "the production grid or run, select a checkpoint, or manually correct a result.",
        "",
        "## Provenance",
        "",
        f"- report attempt: `{report_attempt_id}`",
        f"- plan attempt: `{study.collected['plan_attempt_id']}`",
        f"- plan SHA-256: `{study.plan_sha256}`",
        f"- collection attempt: `{study.collected['collect_attempt_id']}`",
        f"- collected JSON SHA-256: `{study.collected_sha256}`",
        f"- evaluator Git SHA: `{study.evaluation_git_sha}`",
        f"- scale: `{study.manifest['scale']}`",
        "- checkpoint policy: both 25k and 50k are reported without selection",
        "- input policy: only manifest-verified summaries and SHA-verified, read-only",
        "  memory-mapped record CSVs were consumed",
        "- output policy: SVG, PDF, and 300-DPI PNG; every plot-data row carries",
        "  the plan, evaluator, checkpoint, source-row, and artifact hashes",
        "",
        "## Estimator and physics semantics",
        "",
        "- **Primary energy:** the complete 256×4096 retained-trajectory mean; error",
        "  bars are the correlation-aware MCSE from `pooled_geyer_ips/v1`.",
        "- **Diagnostic energy:** long-chain, burn-in, and stride arms stay labeled",
        "  diagnostic. The terminal-snapshot IID stderr and same-trajectory IID stderr",
        "  are tabulated, never substituted for the MCSE.",
        "- **Conditioned values:** variance attribution and CCDFs are diagnostic",
        "  centered statistics, explicitly not headline energy estimators.",
        "- **Cusp and curvature:** executed factor derivatives are shown beside the",
        "  ideal cusp laws. Direct second derivatives are descriptive; no universal",
        "  Kato curvature target is asserted.",
        "- **Factor response:** paired fixed-configuration deltas and independently",
        "  re-equilibrated energy estimates are separate comparison bases. Paired bars",
        "  are IID-only standard errors over fixed configurations, not MCSEs; independent",
        "  delta bars combine arm and baseline correlation-aware MCSEs in quadrature.",
        "- **Symmetry:** singlet purity is interpreted only for spatial coordinate",
        "  exchange. A full-label antisymmetric wavefunction has triplet fraction 1",
        "  by construction, so that number is not used as contamination.",
        "",
        "## Primary energy estimates",
        "",
        "| checkpoint | row | status | trajectory mean (Ha) | MCSE (Ha) | ESS |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["checkpoint_label"]),
                    f"`{row['row_id']}`",
                    str(row["status"]),
                    _markdown_value(row["trajectory_mean_ha"]),
                    _markdown_value(row["mcse_ha"]),
                    _markdown_value(row["ess"]),
                )
            )
            + " |"
        )
    lines += ["", "## Figures", ""]
    captions = {
        "energy_mcse": "Primary and diagnostic trajectory estimators; only MCSE is used as an energy error bar.",
        "distribution_ccdf": "Complete retained-record distribution and diagnostic absolute-deviation CCDF.",
        "conditioned_variance": "Predeclared geometric-bin contributions to local-energy variance; non-headline diagnostic.",
        "cusp_curvature": "Executed electron–nucleus/electron–electron cusp response, ideal laws, and target-free direct curvature.",
        "singular_cancellation": "Executed Hamiltonian singular-term magnitude and cancellation residual.",
        "tails": "Executed one-electron and centre-of-mass outer log-amplitude tails.",
        "symmetry_equivariance": "Worst-case typed symmetry, rotation, trace, feature, and readout checks.",
        "sampler_health_timing": "Sampler acceptance, evaluation wall time, and ESS throughput with availability preserved.",
        "factor_response": "Paired fixed-configuration response and independent re-equilibrated response in separate panels.",
    }
    for name in TABLE_NAMES:
        title = name.replace("_", " ").title()
        lines += [
            f"### {title}",
            "",
            f"![{title}](figures/{name}.svg)",
            "",
            captions[name],
            f"Sources and exact provenance: [plot data](plot-data/{name}.csv). "
            f"Alternate formats: [PDF](figures/{name}.pdf), "
            f"[300-DPI PNG](figures/{name}.png).",
            "",
        ]
    lines += [
        "## Unavailable, absent, or unresolved metrics",
        "",
        "`unavailable` is never rendered as zero. `absent` means no trajectory",
        "existed for the typed identity; `unresolved` means a trajectory existed but",
        "its estimator did not resolve. Successful reports may therefore contain",
        "explicitly unavailable diagnostics without becoming a fabricated number.",
        "",
    ]
    if unavailable:
        lines += [
            "| table | checkpoint | source rows | status | reason |",
            "|---|---|---|---|---|",
        ]
        for record in unavailable:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(record["table"]),
                        str(record.get("checkpoint_label", "not_applicable")),
                        f"`{record.get('source_row_ids', 'not_applicable')}`",
                        str(record["status"]),
                        str(record["reason"]).replace("|", "\\|"),
                    )
                )
                + " |"
            )
    else:
        lines.append("All requested metrics were available.")
    lines += [
        "",
        "## Artifact inventory",
        "",
        f"The machine-readable [`{REPORT_MANIFEST_FILENAME}`]({REPORT_MANIFEST_FILENAME})",
        "records every source artifact's path, byte count, and SHA-256 plus every",
        "generated table and figure hash. Generated scientific result data belong in",
        "the runtime report attempt, not in Git.",
        "",
    ]
    return "\n".join(lines)


def _unavailable_rows(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for table_name, rows in tables.items():
        for row in rows:
            status: str | None = None
            if row.get("status") in {"absent", "unresolved"}:
                status = str(row["status"])
            elif row.get("available") is False:
                status = "unavailable"
            elif any(value == "unavailable" for value in row.values()):
                status = "unavailable"
            if status is None:
                continue
            reason = str(row.get("unavailable_reason") or "metric not emitted")
            key = (
                table_name,
                str(row.get("checkpoint_label")),
                str(row.get("source_row_ids")),
                reason,
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "table": table_name,
                    "checkpoint_label": str(row.get("checkpoint_label", "not_applicable")),
                    "source_row_ids": str(row.get("source_row_ids", "not_applicable")),
                    "status": status,
                    "reason": reason,
                }
            )
    return output


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the immutable diagnostic-report invocation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    parser.add_argument("--collect-attempt-id", required=True)
    parser.add_argument("--report-attempt-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build one report attempt and print its durable location."""

    args = parse_args(argv)
    manifest = build_report(
        results_root=args.results_root,
        plan_attempt_id=args.plan_attempt_id,
        collect_attempt_id=args.collect_attempt_id,
        report_attempt_id=args.report_attempt_id,
    )
    report_dir = layout.report_attempt_dir(args.results_root, args.report_attempt_id)
    print(
        f"he-v1 diagnostic report wrote {report_dir} "
        f"({len(manifest['figures'])} figure files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagnosticReportError",
    "REPORT_SCHEMA",
    "build_plot_tables",
    "build_report",
    "mmap_csv_rows",
    "read_verified_study",
    "render_markdown",
]
