"""Portable, deterministic V3 smoke-reference storage for V4-0 parity."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import yaml

from audit import audit_completed_lineage, reference_evidence
from fanout_audit import (
    SCIENCE_METRIC_ANCHORS,
    STAGE_EXPECTATIONS,
    WORKER_RUNTIME_SCHEMA_VERSION,
    WORKER_RUNTIME_VOLATILE_POINTERS,
)
from roots import ROOT_SENTINEL, validate_lineage_id
from routes import (
    REPO_ROOT,
    STUDY_DIR,
    config_source_receipt,
    legacy_source_receipt,
    runtime_source_receipt,
)
from reference_evidence import (
    CHECKPOINT_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_INPUT_SCHEMA_VERSION,
    FANOUT_ATTEMPTS,
    discovery_anchor_paths as _discovery_anchor_paths,
    evidence_input_receipt as _evidence_input_receipt,
    source_snapshot as _source_snapshot,
)
from reference_contracts import (
    _verify_evidence_input_receipt,
    _verify_science_metric_summary,
    _verify_worker_runtime_summary,
)
from science_audit import canonical_sha256
import selector_verifiers
from strict_data import (
    StrictDataError,
    iter_jsonl,
    load_json,
    load_yaml,
    loads_json,
    loads_yaml,
    validate_structured_paths,
)

REFERENCE_SCHEMA_VERSION = "pair-stability-v4/reference/v1"
REFERENCE_COMPARISON_SCHEMA_VERSION = (
    "pair-stability-v4/reference-comparison/v1"
)
COMPARISON_LAYOUT_LOGICAL_PATH = (
    "experiments/hooke/pair_stability_v4/reference/layout_maps/v1.json"
)
COMPARISON_LAYOUT_PATH = REPO_ROOT / COMPARISON_LAYOUT_LOGICAL_PATH
COMPARISON_LAYOUT_SCHEMA_VERSION = "pair-stability-v4/layout-map/v1"
COMPARATOR_SCHEMA_VERSION = "pair-stability-v4/comparator/v1"
INVENTORY_CONTRACT_SCHEMA_VERSION = (
    "pair-stability-v4/reference-inventory/v1"
)
RAW_TABLE_LIMIT = 1_048_576
REFERENCE_OWNER_ROOT = STUDY_DIR / "reference" / "v3_smoke"
LOW_STAGE_ATTEMPTS = {
    "00_grid": "grid",
    "03_collect": "collection",
    "04_select": "selection",
    "05_final_grid": "final_grid",
    "08_final_collect": "final_collect",
    "09_final_report": "report",
}


@dataclass(frozen=True)
class ReferenceArtifact:
    """Describe stored encoding and verified logical bytes for one artifact."""

    logical_role: str
    logical_path: str
    source_path: str
    stored_path: str
    media_type: str
    encoding: str
    raw_sha256: str
    stored_sha256: str
    raw_size: int
    stored_size: int
    table_header: tuple[str, ...] = ()
    row_count: int | None = None
    column_types: Mapping[str, str] = field(default_factory=dict)
    reference_dir: Path | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the portable descriptor representation."""

        value: dict[str, Any] = {
            "logical_role": self.logical_role,
            "logical_path": self.logical_path,
            "source_path": self.source_path,
            "stored_path": self.stored_path,
            "media_type": self.media_type,
            "encoding": self.encoding,
            "raw_sha256": self.raw_sha256,
            "stored_sha256": self.stored_sha256,
            "raw_size": self.raw_size,
            "stored_size": self.stored_size,
        }
        if self.row_count is not None:
            value["table"] = {
                "header": list(self.table_header),
                "row_count": self.row_count,
                "column_types": dict(self.column_types),
            }
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        reference_dir: Path,
    ) -> "ReferenceArtifact":
        """Build one artifact from a strictly validated descriptor row."""

        required = {
            "logical_role",
            "logical_path",
            "source_path",
            "stored_path",
            "media_type",
            "encoding",
            "raw_sha256",
            "stored_sha256",
            "raw_size",
            "stored_size",
        }
        fields = set(value)
        if fields != required and fields != required | {"table"}:
            raise ValueError("reference artifact descriptor fields mismatch")
        table = value.get("table")
        header: tuple[str, ...] = ()
        row_count: int | None = None
        column_types: Mapping[str, str] = {}
        if table is not None:
            if not isinstance(table, dict) or set(table) != {
                "header",
                "row_count",
                "column_types",
            }:
                raise ValueError("reference table descriptor fields mismatch")
            raw_header = table["header"]
            raw_types = table["column_types"]
            if not isinstance(raw_header, list) or not all(
                isinstance(item, str) for item in raw_header
            ):
                raise ValueError("reference table header is invalid")
            if not isinstance(raw_types, dict) or set(raw_types) != set(
                raw_header
            ):
                raise ValueError("reference table column types mismatch")
            header = tuple(raw_header)
            row_count = int(table["row_count"])
            column_types = {
                str(key): str(item) for key, item in raw_types.items()
            }
        artifact = cls(
            logical_role=str(value["logical_role"]),
            logical_path=_safe_relative_text(value["logical_path"]),
            source_path=_safe_relative_text(value["source_path"]),
            stored_path=_safe_relative_text(value["stored_path"]),
            media_type=str(value["media_type"]),
            encoding=str(value["encoding"]),
            raw_sha256=_digest_text(value["raw_sha256"]),
            stored_sha256=_digest_text(value["stored_sha256"]),
            raw_size=int(value["raw_size"]),
            stored_size=int(value["stored_size"]),
            table_header=header,
            row_count=row_count,
            column_types=column_types,
            reference_dir=reference_dir,
        )
        if artifact.encoding not in {"raw", "gzip"}:
            raise ValueError("unsupported reference artifact encoding")
        if artifact.raw_size < 0 or artifact.stored_size < 0:
            raise ValueError("reference artifact sizes must be nonnegative")
        return artifact


def enumerate_inventory(
    results_root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[Path, ...]:
    """Return ordered protected paths from explicit lineage manifests."""

    root = _isolated_absolute_root(results_root)
    normalized = _normalize_attempts(attempts)
    paths: list[Path] = []

    grid_dir = root / "00_grid" / normalized["grid"]
    grid = _read_json_object(grid_dir / "manifest.json")
    paths.extend(
        grid_dir / name
        for name in (
            "manifest.json",
            "unblind.json",
            "commands.sh",
            "grid.yaml",
            "train_config.yaml",
            "validation_config.yaml",
        )
    )
    jobs = grid.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("grid manifest jobs are not a list")
    for row in jobs:
        if not isinstance(row, dict):
            raise ValueError("grid manifest job is not an object")
        run_id = _safe_component(row.get("run_id"), "grid run_id")
        paths.append(grid_dir / "jobs" / f"{run_id}.json")

    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        plan_dir = (
            root
            / stage
            / "stage_plans"
            / normalized[attempt_key]
        )
        paths.extend(
            plan_dir / name
            for name in (
                "stage_manifest.json",
                "tasks.jsonl",
                "execution_records.jsonl",
            )
        )
        for line in (plan_dir / "tasks.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            task = loads_json(line, source=f"{plan_dir / 'tasks.jsonl'}")
            if not isinstance(task, dict):
                raise ValueError(f"{stage} task row is not an object")
            result_dir = Path(str(task.get("result_dir") or ""))
            paths.append(result_dir / "submission.json")

    collect_dir = root / "03_collect" / normalized["collection"]
    paths.extend(
        collect_dir / name
        for name in (
            "summary.csv",
            "failures.csv",
            "collection_report.json",
            "cost_by_run.csv",
            "cost_by_axis.csv",
            "cost_by_task.csv",
            "source_grid_attempt.json",
            "source_validation_attempts.json",
            "task_lineage.jsonl",
        )
    )

    selection_dir = root / "04_select" / normalized["selection"]
    paths.extend(
        selection_dir / name
        for name in (
            "champions.csv",
            "selection_report.json",
            "source_collection_attempt.json",
            "task_lineage.jsonl",
        )
    )

    final_grid_dir = root / "05_final_grid" / normalized["final_grid"]
    paths.extend(
        final_grid_dir / name
        for name in (
            "source_champions.csv",
            "source_selection_attempt.json",
            "final_jobs.csv",
            "manifest.json",
            "manifest.yaml",
            "task_lineage.jsonl",
        )
    )
    final_jobs = _read_csv_rows(final_grid_dir / "final_jobs.csv")
    for row in final_jobs:
        run_id = _safe_component(
            row.get("final_run_id"),
            "final-grid final_run_id",
        )
        paths.append(final_grid_dir / "jobs" / f"{run_id}.json")

    final_collect_dir = (
        root / "08_final_collect" / normalized["final_collect"]
    )
    final_collect_manifest = _read_yaml_object(
        final_collect_dir / "manifest.yaml"
    )
    paths.append(final_collect_dir / "manifest.yaml")
    paths.extend(
        final_collect_dir / name
        for name in _manifest_table_names(final_collect_manifest)
    )

    report_dir = root / "09_final_report" / normalized["report"]
    report = _read_json_object(report_dir / "final_report.json")
    paths.extend((report_dir / "final_report.json", report_dir / "report.md"))
    report_tables = report.get("tables")
    if not isinstance(report_tables, dict):
        raise ValueError("final report tables are not a mapping")
    for name in report_tables:
        paths.append(report_dir / "tables" / _safe_nested_path(name))

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        canonical = _required_regular_file(path, root=root)
        if canonical in seen:
            raise ValueError(
                f"duplicate protected logical path: {path.relative_to(root)}"
            )
        seen.add(canonical)
        ordered.append(canonical)
    return tuple(sorted(ordered, key=lambda path: path.relative_to(root).as_posix()))


def freeze_reference(
    results_root: Path,
    destination: Path,
    *,
    attempts: Mapping[str, str],
) -> Path:
    """Atomically create one immutable reference from an audited V3 lineage."""

    root = _isolated_absolute_root(results_root)
    if (root / ROOT_SENTINEL).exists() or (root / ROOT_SENTINEL).is_symlink():
        raise ValueError("reference source must be an isolated V3 lineage")
    normalized = _normalize_attempts(attempts)
    discovery_anchors = _discovery_anchor_paths(
        root,
        attempts=normalized,
    )
    anchor_snapshots = {
        path: _source_snapshot(path) for path in discovery_anchors
    }
    inventory = enumerate_inventory(root, attempts=normalized)
    validate_structured_paths(inventory)
    snapshots = {path: _source_snapshot(path) for path in inventory}
    evidence_inputs = _evidence_input_receipt(
        root,
        attempts=normalized,
        protected_paths=inventory,
    )
    for path, before in anchor_snapshots.items():
        if _source_snapshot(path) != before:
            raise RuntimeError(
                f"discovery anchor mutated while deriving read set: {path}"
            )
    audit_errors = audit_completed_lineage(root, attempts=normalized)
    if audit_errors:
        raise ValueError(
            "source lineage audit failed: " + "; ".join(audit_errors)
        )
    audit_evidence = reference_evidence(root, attempts=normalized)
    source_provenance = _source_provenance()
    # Capture closure facts before any output directory can alter git status.
    legacy_closure = legacy_source_receipt(REPO_ROOT)
    runtime_closure = runtime_source_receipt(REPO_ROOT)
    config_closure = config_source_receipt(REPO_ROOT)
    _require_worker_commit(root, normalized, source_provenance["commit"])

    canonical_destination = _reference_destination(
        destination,
        owner=REFERENCE_OWNER_ROOT,
    )
    if canonical_destination == root or root in canonical_destination.parents:
        raise ValueError("reference destination may not be inside source results")

    # Source/runtime provenance is already captured.  Stage beside the final
    # owner so publication can be one atomic directory rename without letting
    # the staging directory contaminate the captured clean source closure.
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=canonical_destination.parent,
        )
    )
    artifacts: list[ReferenceArtifact] = []
    try:
        for source in inventory:
            relative = source.relative_to(root)
            role = _logical_role(relative)
            logical = _logical_path(relative, normalized)
            artifacts.append(
                _store_artifact(
                    source,
                    root=root,
                    destination=temporary,
                    logical_role=role,
                    logical_path=logical,
                )
            )
        for source, before in snapshots.items():
            if _source_snapshot(source) != before:
                raise RuntimeError(f"source mutated during freeze: {source}")
        if _evidence_input_receipt(
            root,
            attempts=normalized,
            protected_paths=inventory,
        ) != evidence_inputs:
            raise RuntimeError("raw audit evidence mutated during freeze")
        figures = _figure_metadata(root, normalized)
        descriptor = {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "study": "pair_stability_v3",
            "attempts": normalized,
            "source": source_provenance,
            "source_results": _source_results_identity(root, normalized),
            "legacy_closure": legacy_closure,
            "runtime_closure": runtime_closure,
            "config_closure": config_closure,
            "lineage_summary": _lineage_summary(root, normalized),
            "audit_evidence": audit_evidence,
            "evidence_inputs": evidence_inputs,
            "figures": figures,
            "artifacts": [artifact.to_dict() for artifact in artifacts],
            "inventory_contract": _inventory_contract(artifacts),
            "comparison_contract": _comparison_contract(),
        }
        _write_json(temporary / "reference.json", descriptor)
        verification_errors = verify_reference(temporary)
        if verification_errors:
            raise ValueError(
                "new reference failed verification: "
                + "; ".join(verification_errors)
            )
        if _evidence_input_receipt(
            root,
            attempts=normalized,
            protected_paths=inventory,
        ) != evidence_inputs:
            raise RuntimeError(
                "source read set mutated before reference publication"
            )
        _rename_noreplace(temporary, canonical_destination)
    except BaseException:
        # Preserve the sibling temporary tree as failure evidence.
        raise
    return canonical_destination


def verify_reference(reference_dir: Path) -> tuple[str, ...]:
    """Verify descriptor, stored bytes, decoded bytes, schemas, and counts."""

    errors: list[str] = []
    try:
        root = Path(reference_dir).resolve(strict=True)
    except OSError as exc:
        return (f"reference directory is unavailable: {exc}",)
    if not root.is_dir() or Path(reference_dir).is_symlink():
        return ("reference path is not a regular directory",)
    descriptor_path = root / "reference.json"
    try:
        descriptor = _read_json_object(descriptor_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (f"invalid reference descriptor: {exc}",)
    expected_fields = {
        "schema_version",
        "study",
        "attempts",
        "source",
        "source_results",
        "legacy_closure",
        "runtime_closure",
        "config_closure",
        "lineage_summary",
        "audit_evidence",
        "evidence_inputs",
        "figures",
        "artifacts",
        "inventory_contract",
        "comparison_contract",
    }
    if set(descriptor) != expected_fields:
        errors.append("reference descriptor fields mismatch")
    if descriptor.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        errors.append("reference schema_version mismatch")
    if descriptor.get("study") != "pair_stability_v3":
        errors.append("reference study is not pair_stability_v3")
    try:
        _normalize_attempts(descriptor.get("attempts", {}))
    except (KeyError, ValueError) as exc:
        errors.append(f"reference attempts invalid: {exc}")
    raw_artifacts = descriptor.get("artifacts")
    if not isinstance(raw_artifacts, list):
        errors.append("reference artifacts are not a list")
        return tuple(errors)
    artifacts: list[ReferenceArtifact] = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, dict):
            errors.append(f"reference artifact {index} is not an object")
            continue
        try:
            artifacts.append(
                ReferenceArtifact.from_dict(raw, reference_dir=root)
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid reference artifact {index}: {exc}")
    logical_paths = [artifact.logical_path for artifact in artifacts]
    source_paths = [artifact.source_path for artifact in artifacts]
    stored_paths = [artifact.stored_path for artifact in artifacts]
    if len(set(logical_paths)) != len(logical_paths):
        errors.append("reference contains duplicate logical paths")
    if len(set(stored_paths)) != len(stored_paths):
        errors.append("reference contains duplicate stored paths")
    if len(set(source_paths)) != len(source_paths):
        errors.append("reference contains duplicate source paths")
    if logical_paths != sorted(logical_paths):
        errors.append("reference artifacts are not in deterministic path order")
    try:
        normalized_attempts = _normalize_attempts(
            descriptor.get("attempts", {})
        )
    except ValueError:
        normalized_attempts = {}
    for artifact in artifacts:
        errors.extend(
            _verify_artifact_descriptor(
                artifact,
                attempts=normalized_attempts,
            )
        )
        errors.extend(_verify_structured_artifact(artifact))

    contract = descriptor.get("inventory_contract")
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version",
        "artifact_count",
        "logical_paths_sha256",
    }:
        errors.append("reference inventory contract is invalid")
    elif contract != _inventory_contract(artifacts):
        errors.append("reference inventory contract mismatch")
    errors.extend(
        _verify_comparison_contract(descriptor.get("comparison_contract"))
    )

    expected_stored = {"reference.json"}
    for artifact in artifacts:
        expected_stored.add(artifact.stored_path)
        errors.extend(_verify_artifact(artifact))
    actual_stored = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_stored != expected_stored:
        errors.append(
            "reference stored-file population mismatch; "
            f"missing={sorted(expected_stored - actual_stored)}, "
            f"extra={sorted(actual_stored - expected_stored)}"
        )
    try:
        derived_paths, derived_counts = _derive_expected_inventory(artifacts)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        errors.append(f"cannot derive protected inventory: {exc}")
    else:
        if set(logical_paths) != derived_paths:
            errors.append(
                "reference protected logical-path population mismatch; "
                f"missing={sorted(derived_paths - set(logical_paths))}, "
                f"extra={sorted(set(logical_paths) - derived_paths)}"
            )
        expected_counts = {
            "grid_jobs": 64,
            "01_train": 64,
            "02_validation": 64,
            "final_jobs": 8,
            "06_final_train": 8,
            "07_final_eval": 8,
        }
        if derived_counts != expected_counts:
            errors.append(
                "reference protected population counts mismatch; "
                f"observed={derived_counts!r}"
            )
    figures = descriptor.get("figures")
    if not isinstance(figures, list) or not all(
        isinstance(row, dict)
        and set(row) == {"logical_path", "size"}
        and isinstance(row["size"], int)
        and row["size"] > 0
        for row in figures
    ):
        errors.append("reference figure presence metadata is invalid")
    else:
        try:
            expected_figures = _expected_figure_logical_paths(artifacts)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot derive reference figure contract: {exc}")
        else:
            actual_figures = [str(row["logical_path"]) for row in figures]
            if len(set(actual_figures)) != len(actual_figures):
                errors.append("reference figure metadata contains duplicates")
            if actual_figures != expected_figures:
                errors.append("reference figure metadata population mismatch")
    summary = descriptor.get("lineage_summary")
    if not isinstance(summary, dict):
        errors.append("reference lineage summary is not an object")
    else:
        expected_populations = {
            stage: int(spec["count"])
            for stage, spec in STAGE_EXPECTATIONS.items()
        }
        if summary.get("fanout_populations") != expected_populations:
            errors.append("reference fan-out population summary mismatch")
        errors.extend(_verify_lineage_summary(summary, descriptor.get("source")))
    errors.extend(_verify_source_provenance(descriptor.get("source")))
    errors.extend(
        _verify_source_results_identity(
            descriptor.get("source_results"),
            descriptor.get("attempts"),
        )
    )
    errors.extend(_verify_legacy_closure(descriptor.get("legacy_closure")))
    errors.extend(
        _verify_runtime_closure(
            descriptor.get("runtime_closure"),
            descriptor.get("source"),
        )
    )
    errors.extend(_verify_config_closure(descriptor.get("config_closure")))
    errors.extend(
        _verify_audit_evidence(
            descriptor.get("audit_evidence"),
            artifacts,
        )
    )
    errors.extend(
        _verify_evidence_input_receipt(
            descriptor.get("evidence_inputs"),
            artifacts=artifacts,
            attempts=descriptor.get("attempts"),
            source_results=descriptor.get("source_results"),
        )
    )
    return tuple(dict.fromkeys(errors))


def open_logical_content(entry: ReferenceArtifact) -> BinaryIO:
    """Open verified raw logical bytes regardless of stored encoding."""

    if entry.reference_dir is None:
        raise ValueError("reference artifact is not bound to a reference root")
    stored = _required_regular_file(
        entry.reference_dir / entry.stored_path,
        root=entry.reference_dir,
    )
    if _sha256_file(stored) != entry.stored_sha256:
        raise ValueError(f"stored digest mismatch: {entry.logical_path}")
    if stored.stat().st_size != entry.stored_size:
        raise ValueError(f"stored size mismatch: {entry.logical_path}")
    if entry.encoding == "raw":
        return stored.open("rb")
    if entry.encoding == "gzip":
        return gzip.open(stored, "rb")
    raise ValueError(f"unsupported encoding: {entry.encoding}")


def load_reference(reference_dir: Path) -> tuple[dict[str, Any], tuple[ReferenceArtifact, ...]]:
    """Return one verified descriptor and its bound artifact records."""

    errors = verify_reference(reference_dir)
    if errors:
        raise ValueError("reference verification failed: " + "; ".join(errors))
    root = Path(reference_dir).resolve(strict=True)
    descriptor = _read_json_object(root / "reference.json")
    artifacts = tuple(
        ReferenceArtifact.from_dict(row, reference_dir=root)
        for row in descriptor["artifacts"]
    )
    return descriptor, artifacts


def _inventory_contract(
    artifacts: Sequence[ReferenceArtifact],
) -> dict[str, Any]:
    logical_paths = [artifact.logical_path for artifact in artifacts]
    payload = json.dumps(
        logical_paths,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return {
        "schema_version": INVENTORY_CONTRACT_SCHEMA_VERSION,
        "artifact_count": len(logical_paths),
        "logical_paths_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _comparison_contract() -> dict[str, Any]:
    """Pin the immutable layout and comparator versions used for parity."""

    path = COMPARISON_LAYOUT_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("comparison layout map is not a regular file")
    raw = _read_json_object(path)
    if raw.get("schema_version") != COMPARISON_LAYOUT_SCHEMA_VERSION:
        raise ValueError("comparison layout-map schema is incompatible")
    if raw.get("comparator_schema_version") != COMPARATOR_SCHEMA_VERSION:
        raise ValueError("comparison comparator schema is incompatible")
    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": REFERENCE_COMPARISON_SCHEMA_VERSION,
        "layout_map_path": COMPARISON_LAYOUT_LOGICAL_PATH,
        "layout_map_schema_version": COMPARISON_LAYOUT_SCHEMA_VERSION,
        "comparator_schema_version": COMPARATOR_SCHEMA_VERSION,
        "layout_map_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _verify_comparison_contract(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "layout_map_path",
        "layout_map_schema_version",
        "comparator_schema_version",
        "layout_map_sha256",
    }:
        return ["reference comparison-contract schema mismatch"]
    try:
        expected = _comparison_contract()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot verify immutable comparison layout map: {exc}"]
    if value != expected:
        return ["reference comparison contract/layout digest mismatch"]
    return []


def _verify_artifact_descriptor(
    artifact: ReferenceArtifact,
    *,
    attempts: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    source = Path(artifact.source_path)
    try:
        expected_logical = _logical_path(source, attempts)
    except ValueError as exc:
        errors.append(
            f"invalid source/logical mapping {artifact.logical_path}: {exc}"
        )
        expected_logical = ""
    if artifact.logical_path != expected_logical:
        errors.append(f"logical path mismatch: {artifact.logical_path}")
    if artifact.logical_role != _logical_role(source):
        errors.append(f"logical role mismatch: {artifact.logical_path}")
    if artifact.media_type != _media_type(source):
        errors.append(f"media type mismatch: {artifact.logical_path}")
    tabular = source.suffix.lower() in {".csv", ".tsv"}
    expected_encoding = (
        "gzip"
        if tabular and artifact.raw_size > RAW_TABLE_LIMIT
        else "raw"
    )
    if artifact.encoding != expected_encoding:
        errors.append(f"encoding policy mismatch: {artifact.logical_path}")
    expected_stored = Path("inventory") / source
    if expected_encoding == "gzip":
        expected_stored = expected_stored.with_name(
            expected_stored.name + ".gz"
        )
    if artifact.stored_path != expected_stored.as_posix():
        errors.append(f"stored path mismatch: {artifact.logical_path}")
    if tabular != (artifact.row_count is not None):
        errors.append(f"table descriptor presence mismatch: {artifact.logical_path}")
    if artifact.row_count is not None:
        if artifact.row_count < 0:
            errors.append(f"negative table row count: {artifact.logical_path}")
        if len(set(artifact.table_header)) != len(artifact.table_header):
            errors.append(f"duplicate table header: {artifact.logical_path}")
        if set(artifact.column_types) != set(artifact.table_header):
            errors.append(f"table column type keys mismatch: {artifact.logical_path}")
        allowed = {"empty", "boolean", "integer", "float", "string"}
        if any(value not in allowed for value in artifact.column_types.values()):
            errors.append(f"invalid table column type: {artifact.logical_path}")
    return errors


def _derive_expected_inventory(
    artifacts: Sequence[ReferenceArtifact],
) -> tuple[set[str], dict[str, int]]:
    by_logical = {artifact.logical_path: artifact for artifact in artifacts}
    expected = {
        "00_grid/{grid}/manifest.json",
        "00_grid/{grid}/unblind.json",
        "00_grid/{grid}/commands.sh",
        "00_grid/{grid}/grid.yaml",
        "00_grid/{grid}/train_config.yaml",
        "00_grid/{grid}/validation_config.yaml",
        "03_collect/{collection}/summary.csv",
        "03_collect/{collection}/failures.csv",
        "03_collect/{collection}/collection_report.json",
        "03_collect/{collection}/cost_by_run.csv",
        "03_collect/{collection}/cost_by_axis.csv",
        "03_collect/{collection}/cost_by_task.csv",
        "03_collect/{collection}/source_grid_attempt.json",
        "03_collect/{collection}/source_validation_attempts.json",
        "03_collect/{collection}/task_lineage.jsonl",
        "04_select/{selection}/champions.csv",
        "04_select/{selection}/selection_report.json",
        "04_select/{selection}/source_collection_attempt.json",
        "04_select/{selection}/task_lineage.jsonl",
        "05_final_grid/{final_grid}/source_champions.csv",
        "05_final_grid/{final_grid}/source_selection_attempt.json",
        "05_final_grid/{final_grid}/final_jobs.csv",
        "05_final_grid/{final_grid}/manifest.json",
        "05_final_grid/{final_grid}/manifest.yaml",
        "05_final_grid/{final_grid}/task_lineage.jsonl",
        "08_final_collect/{final_collect}/manifest.yaml",
        "09_final_report/{report}/final_report.json",
        "09_final_report/{report}/report.md",
    }
    counts: dict[str, int] = {}
    grid = _artifact_json(
        by_logical,
        "00_grid/{grid}/manifest.json",
    )
    raw_grid_jobs = grid.get("jobs")
    if not isinstance(raw_grid_jobs, list):
        raise ValueError("frozen grid manifest jobs are not a list")
    grid_run_ids: list[str] = []
    for row in raw_grid_jobs:
        if not isinstance(row, dict):
            raise ValueError("frozen grid manifest job is not an object")
        run_id = _safe_component(row.get("run_id"), "frozen grid run_id")
        grid_run_ids.append(run_id)
        expected.add(f"00_grid/{{grid}}/jobs/{run_id}.json")
    if len(set(grid_run_ids)) != len(grid_run_ids):
        raise ValueError("frozen grid manifest contains duplicate run ids")
    counts["grid_jobs"] = len(grid_run_ids)

    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        plan_prefix = f"{stage}/stage_plans/{{{attempt_key}}}"
        task_path = f"{plan_prefix}/tasks.jsonl"
        expected.update(
            {
                f"{plan_prefix}/stage_manifest.json",
                task_path,
                f"{plan_prefix}/execution_records.jsonl",
            }
        )
        tasks = _artifact_jsonl(by_logical, task_path)
        run_ids: list[str] = []
        for row in tasks:
            run_id = _safe_component(
                row.get("run_id"),
                f"frozen {stage} run_id",
            )
            run_ids.append(run_id)
            expected.add(
                f"{stage}/{run_id}/{{{attempt_key}}}/submission.json"
            )
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(f"frozen {stage} tasks contain duplicate run ids")
        counts[stage] = len(run_ids)

    final_rows = _artifact_csv(
        by_logical,
        "05_final_grid/{final_grid}/final_jobs.csv",
    )
    final_run_ids: list[str] = []
    for row in final_rows:
        run_id = _safe_component(
            row.get("final_run_id"),
            "frozen final run_id",
        )
        final_run_ids.append(run_id)
        expected.add(
            f"05_final_grid/{{final_grid}}/jobs/{run_id}.json"
        )
    if len(set(final_run_ids)) != len(final_run_ids):
        raise ValueError("frozen final jobs contain duplicate run ids")
    counts["final_jobs"] = len(final_run_ids)

    final_collect = _artifact_yaml(
        by_logical,
        "08_final_collect/{final_collect}/manifest.yaml",
    )
    raw_collect_tables = final_collect.get("tables")
    if not isinstance(raw_collect_tables, dict):
        raise ValueError("frozen final-collect tables are not a mapping")
    for name in raw_collect_tables:
        safe_name = _safe_nested_path(name).as_posix()
        expected.add(f"08_final_collect/{{final_collect}}/{safe_name}")

    report = _artifact_json(
        by_logical,
        "09_final_report/{report}/final_report.json",
    )
    raw_report_tables = report.get("tables")
    if not isinstance(raw_report_tables, dict):
        raise ValueError("frozen final-report tables are not a mapping")
    if not set(raw_collect_tables).issubset(raw_report_tables):
        raise ValueError("frozen final report omits final-collect tables")
    for name in raw_report_tables:
        safe_name = _safe_nested_path(name).as_posix()
        expected.add(f"09_final_report/{{report}}/tables/{safe_name}")
    return expected, counts


def _expected_figure_logical_paths(
    artifacts: Sequence[ReferenceArtifact],
) -> list[str]:
    by_logical = {artifact.logical_path: artifact for artifact in artifacts}
    report = _artifact_json(
        by_logical,
        "09_final_report/{report}/final_report.json",
    )
    figures = report.get("figures")
    if not isinstance(figures, list) or not all(
        isinstance(name, str) for name in figures
    ):
        raise ValueError("frozen final-report figures are not filenames")
    logical: list[str] = []
    for name in figures:
        relative = _safe_nested_path(name).as_posix()
        logical.append(
            f"09_final_report/{{report}}/figures/{relative}"
        )
    return logical


def _artifact_bytes(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> bytes:
    artifact = by_logical.get(logical_path)
    if artifact is None:
        raise ValueError(f"missing protected contract artifact: {logical_path}")
    with open_logical_content(artifact) as handle:
        return handle.read()


def _artifact_json(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> dict[str, Any]:
    value = loads_json(
        _artifact_bytes(by_logical, logical_path),
        source=f"frozen artifact {logical_path}",
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected frozen JSON object: {logical_path}")
    return value


def _artifact_jsonl(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _artifact_bytes(by_logical, logical_path).decode().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = loads_json(
            line,
            source=f"frozen artifact {logical_path}:{line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"frozen JSONL row is not an object: {logical_path}:{line_number}"
            )
        rows.append(value)
    return rows


def _artifact_csv(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> list[dict[str, str]]:
    text = io.StringIO(_artifact_bytes(by_logical, logical_path).decode())
    reader = csv.DictReader(text)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(
        reader.fieldnames
    ):
        raise ValueError(f"invalid frozen CSV header: {logical_path}")
    return list(reader)


def _artifact_yaml(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> dict[str, Any]:
    value = loads_yaml(
        _artifact_bytes(by_logical, logical_path),
        source=f"frozen artifact {logical_path}",
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected frozen YAML object: {logical_path}")
    return value


def _verify_source_provenance(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "commit",
        "branch",
        "dirty",
    }:
        return ["reference source provenance schema mismatch"]
    errors: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit") or "")):
        errors.append("reference source commit is invalid")
    if not isinstance(value.get("branch"), str) or not value.get("branch"):
        errors.append("reference source branch is invalid")
    if value.get("dirty") is not False:
        errors.append("reference source provenance is dirty")
    return errors


def _source_results_identity(
    root: Path,
    attempts: Mapping[str, str],
) -> dict[str, Any]:
    stat = root.stat()
    return {
        "canonical_root": str(root),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "lineage_ids": sorted(set(attempts.values())),
    }


def _verify_source_results_identity(
    value: object,
    attempts: object,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "canonical_root",
        "device",
        "inode",
        "lineage_ids",
    }:
        return ["reference source-results identity schema mismatch"]
    errors: list[str] = []
    canonical_root = value.get("canonical_root")
    if (
        not isinstance(canonical_root, str)
        or not Path(canonical_root).is_absolute()
        or ".." in Path(canonical_root).parts
    ):
        errors.append("reference source-results canonical root is invalid")
    if not isinstance(value.get("device"), int) or value["device"] < 0:
        errors.append("reference source-results device is invalid")
    if not isinstance(value.get("inode"), int) or value["inode"] <= 0:
        errors.append("reference source-results inode is invalid")
    if isinstance(attempts, Mapping):
        expected_ids = sorted(set(str(item) for item in attempts.values()))
    else:
        expected_ids = []
    if value.get("lineage_ids") != expected_ids:
        errors.append("reference source-results lineage ids mismatch")
    return errors


def _verify_legacy_closure(value: object) -> list[str]:
    """Validate frozen pinned-source closure without reading live source."""

    required = {
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "closure_sha256",
        "files",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["reference legacy-closure schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/legacy-source/v1":
        errors.append("reference legacy-closure version mismatch")
    for name in ("manifest_sha256", "closure_sha256"):
        try:
            _digest_text(value.get(name))
        except ValueError:
            errors.append(f"reference legacy-closure {name} is invalid")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        errors.append("reference legacy-closure files are invalid")
        return errors
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append("reference legacy-closure file row is invalid")
            continue
        try:
            paths.append(_safe_relative_text(row["path"]))
            _digest_text(row["sha256"])
        except ValueError:
            errors.append("reference legacy-closure file value is invalid")
    if len(set(paths)) != len(paths):
        errors.append("reference legacy-closure paths are not unique")
    return errors


def _verify_config_closure(value: object) -> list[str]:
    """Validate frozen V4 config closure without comparing checkout commits."""

    required = {"schema_version", "closure_sha256", "files"}
    if not isinstance(value, dict) or set(value) != required:
        return ["reference config-closure schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/config-source/v1":
        errors.append("reference config-closure version mismatch")
    try:
        _digest_text(value.get("closure_sha256"))
    except ValueError:
        errors.append("reference config-closure digest is invalid")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        errors.append("reference config-closure files are invalid")
        return errors
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append("reference config-closure file row is invalid")
            continue
        try:
            paths.append(_safe_relative_text(row["path"]))
            _digest_text(row["sha256"])
        except ValueError:
            errors.append("reference config-closure file value is invalid")
    if len(set(paths)) != len(paths):
        errors.append("reference config-closure paths are not unique")
    return errors


def _verify_runtime_closure(
    value: object,
    source: object,
) -> list[str]:
    required = {
        "schema_version",
        "closure_sha256",
        "n_files",
        "git_commit",
        "git_branch",
        "dirty",
        "python_executable",
        "python_version",
        "uv_project_environment",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["reference runtime-closure schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/runtime-source/v1":
        errors.append("reference runtime-closure version mismatch")
    try:
        _digest_text(value.get("closure_sha256"))
    except ValueError:
        errors.append("reference runtime-closure digest is invalid")
    if not isinstance(value.get("n_files"), int) or value["n_files"] <= 0:
        errors.append("reference runtime-closure file count is invalid")
    if value.get("dirty") is not False:
        errors.append("reference runtime closure is dirty")
    if isinstance(source, dict) and value.get("git_commit") != source.get(
        "commit"
    ):
        errors.append("reference runtime/source commits differ")
    if not isinstance(value.get("python_executable"), str) or not value.get(
        "python_executable"
    ):
        errors.append("reference runtime Python executable is invalid")
    if not isinstance(value.get("python_version"), str) or not value.get(
        "python_version"
    ):
        errors.append("reference runtime Python version is invalid")
    if not isinstance(value.get("cuda_available"), bool):
        errors.append("reference runtime CUDA availability is invalid")
    return errors


def _verify_audit_evidence(
    value: object,
    artifacts: Sequence[ReferenceArtifact],
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "science_metrics",
        "worker_runtime",
        "selection",
    }:
        return ["reference audit-evidence schema mismatch"]
    errors: list[str] = []
    errors.extend(
        _verify_science_metric_summary(
            value["science_metrics"],
            artifacts=artifacts,
        )
    )
    errors.extend(
        _verify_worker_runtime_summary(
            value["worker_runtime"],
            artifacts=artifacts,
        )
    )

    selection = value["selection"]
    if not isinstance(selection, dict):
        errors.append("reference selection contract is not an object")
        return errors
    try:
        by_logical = {
            artifact.logical_path: artifact for artifact in artifacts
        }
        grid_logical = "00_grid/{grid}/manifest.json"
        summary_logical = "03_collect/{collection}/summary.csv"
        report_logical = "04_select/{selection}/selection_report.json"
        champions_logical = "04_select/{selection}/champions.csv"
        grid = _artifact_json(by_logical, grid_logical)
        summary = _artifact_csv(by_logical, summary_logical)
        selection_report = _artifact_json(by_logical, report_logical)
        champions = _artifact_csv(by_logical, champions_logical)
        selection_errors = selector_verifiers.verify_contract(
            selection,
            grid=grid,
            summary_rows=summary,
            selection_report=selection_report,
            champion_rows=champions,
            artifact_sha256={
                "grid_manifest": by_logical[grid_logical].raw_sha256,
                "summary_csv": by_logical[summary_logical].raw_sha256,
                "selection_report": by_logical[
                    report_logical
                ].raw_sha256,
                "champions_csv": by_logical[
                    champions_logical
                ].raw_sha256,
            },
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"cannot verify frozen selection contract: {exc}")
    else:
        errors.extend(selection_errors)
    return errors


def _verify_lineage_summary(
    value: Mapping[str, Any],
    source: object,
) -> list[str]:
    if set(value) != {
        "fanout_populations",
        "worker_commits",
        "profiles",
        "terminal_audit",
    }:
        return ["reference lineage-summary schema mismatch"]
    errors: list[str] = []
    if value.get("terminal_audit") != "passed":
        errors.append("reference terminal audit did not pass")
    commits = value.get("worker_commits")
    source_commit = source.get("commit") if isinstance(source, dict) else None
    if commits != [source_commit]:
        errors.append("reference worker/source commit summary differs")
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(
        STAGE_EXPECTATIONS
    ):
        errors.append("reference profile summary stage set mismatch")
        return errors
    for stage, spec in STAGE_EXPECTATIONS.items():
        profile = profiles.get(stage)
        if not isinstance(profile, dict) or set(profile) != {
            "resources",
            "task_count",
        }:
            errors.append(f"reference profile summary invalid for {stage}")
            continue
        if profile.get("task_count") != int(spec["count"]):
            errors.append(f"reference profile task count differs for {stage}")
        resources = profile.get("resources")
        if not isinstance(resources, dict):
            errors.append(f"reference resources missing for {stage}")
            continue
        expected_resources = {
            "profile": "cuda",
            "device": "cuda",
            "partition": "gpu_test",
            "threads": 4,
            "mem_gb": 32,
            "gpus": 1,
            "timeout_min": int(spec["timeout_min"]),
            "uv_environment": ".venv-gpu",
            "uv_extras": ["cu126"],
            "metadata": {},
        }
        if resources != expected_resources:
            errors.append(f"reference resources differ for {stage}")
    return errors


def _store_artifact(
    source: Path,
    *,
    root: Path,
    destination: Path,
    logical_role: str,
    logical_path: str,
) -> ReferenceArtifact:
    relative = source.relative_to(root).as_posix()
    raw_size = source.stat().st_size
    tabular = source.suffix.lower() in {".csv", ".tsv"}
    encoding = (
        "gzip"
        if tabular and raw_size > RAW_TABLE_LIMIT
        else "raw"
    )
    stored_relative = Path("inventory") / Path(relative)
    if encoding == "gzip":
        stored_relative = stored_relative.with_name(
            stored_relative.name + ".gz"
        )
    stored = destination / stored_relative
    stored.parent.mkdir(parents=True, exist_ok=True)
    if encoding == "raw":
        shutil.copyfile(source, stored)
    else:
        with source.open("rb") as input_handle, stored.open("wb") as output_handle:
            with gzip.GzipFile(
                fileobj=output_handle,
                mode="wb",
                compresslevel=9,
                filename="",
                mtime=0,
            ) as compressed:
                shutil.copyfileobj(input_handle, compressed)
    header: tuple[str, ...] = ()
    row_count: int | None = None
    column_types: Mapping[str, str] = {}
    if tabular:
        delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
        header, row_count, column_types = _table_metadata(
            source,
            delimiter=delimiter,
        )
    return ReferenceArtifact(
        logical_role=logical_role,
        logical_path=logical_path,
        source_path=relative,
        stored_path=stored_relative.as_posix(),
        media_type=_media_type(source),
        encoding=encoding,
        raw_sha256=_sha256_file(source),
        stored_sha256=_sha256_file(stored),
        raw_size=raw_size,
        stored_size=stored.stat().st_size,
        table_header=header,
        row_count=row_count,
        column_types=column_types,
        reference_dir=destination,
    )


def _verify_artifact(artifact: ReferenceArtifact) -> list[str]:
    errors: list[str] = []
    try:
        if artifact.reference_dir is None:
            raise ValueError("reference artifact is not bound to a root")
        stored = _required_regular_file(
            artifact.reference_dir / artifact.stored_path,
            root=artifact.reference_dir,
        )
        if artifact.encoding == "raw":
            if artifact.raw_sha256 != artifact.stored_sha256:
                errors.append(
                    f"raw/stored digest differs for raw artifact: "
                    f"{artifact.logical_path}"
                )
            if artifact.raw_size != artifact.stored_size:
                errors.append(
                    f"raw/stored size differs for raw artifact: "
                    f"{artifact.logical_path}"
                )
        else:
            with stored.open("rb") as stored_handle:
                header = stored_handle.read(10)
            if len(header) < 10 or header[:3] != b"\x1f\x8b\x08":
                errors.append(f"invalid gzip header: {artifact.logical_path}")
            else:
                flags = header[3]
                mtime = int.from_bytes(header[4:8], "little")
                if flags & 0x08:
                    errors.append(
                        f"gzip embeds a filename: {artifact.logical_path}"
                    )
                if mtime != 0:
                    errors.append(
                        f"gzip mtime is not zero: {artifact.logical_path}"
                    )
        with open_logical_content(artifact) as handle:
            raw_digest = hashlib.sha256()
            raw_size = 0
            raw = tempfile.SpooledTemporaryFile(max_size=2 * RAW_TABLE_LIMIT)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                raw_digest.update(chunk)
                raw_size += len(chunk)
                raw.write(chunk)
            if raw_digest.hexdigest() != artifact.raw_sha256:
                errors.append(f"raw digest mismatch: {artifact.logical_path}")
            if raw_size != artifact.raw_size:
                errors.append(f"raw size mismatch: {artifact.logical_path}")
            if artifact.row_count is not None:
                raw.seek(0)
                delimiter = (
                    "\t"
                    if artifact.media_type == "text/tab-separated-values"
                    else ","
                )
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(text, delimiter=delimiter)
                header = tuple(next(reader, ()))
                values: dict[str, list[str]] = {
                    column: [] for column in header
                }
                count = 0
                ragged = False
                for row in reader:
                    if len(row) != len(header):
                        ragged = True
                        continue
                    count += 1
                    for column, value in zip(header, row, strict=True):
                        if value != "":
                            values[column].append(value)
                if header != artifact.table_header:
                    errors.append(
                        f"table header mismatch: {artifact.logical_path}"
                    )
                if len(set(header)) != len(header):
                    errors.append(
                        f"duplicate decoded table header: {artifact.logical_path}"
                    )
                if ragged:
                    errors.append(
                        f"ragged decoded table row: {artifact.logical_path}"
                    )
                if count != artifact.row_count:
                    errors.append(
                        f"table row count mismatch: {artifact.logical_path}"
                    )
                inferred = {
                    column: _column_type(items)
                    for column, items in values.items()
                }
                if inferred != artifact.column_types:
                    errors.append(
                        f"table column types mismatch: {artifact.logical_path}"
                    )
    except (OSError, EOFError, ValueError, gzip.BadGzipFile) as exc:
        errors.append(f"cannot decode {artifact.logical_path}: {exc}")
    return errors


def _verify_structured_artifact(artifact: ReferenceArtifact) -> list[str]:
    """Reject ambiguous stored structured evidence before any policy use."""

    parser = {
        "application/json": loads_json,
        "application/x-ndjson": None,
        "application/yaml": loads_yaml,
    }.get(artifact.media_type)
    if parser is None and artifact.media_type != "application/x-ndjson":
        return []
    try:
        with open_logical_content(artifact) as handle:
            payload = handle.read().decode("utf-8")
        if artifact.media_type == "application/x-ndjson":
            for line_number, line in enumerate(payload.splitlines(), start=1):
                if line.strip():
                    loads_json(
                        line,
                        source=f"frozen artifact {artifact.logical_path}:{line_number}",
                    )
        else:
            assert parser is not None
            parser(payload, source=f"frozen artifact {artifact.logical_path}")
    except (OSError, UnicodeDecodeError, StrictDataError, ValueError) as exc:
        return [f"invalid structured frozen artifact {artifact.logical_path}: {exc}"]
    return []


def _lineage_summary(
    root: Path,
    attempts: Mapping[str, str],
) -> dict[str, Any]:
    worker_commits: set[str] = set()
    profiles: dict[str, Any] = {}
    for stage, key in FANOUT_ATTEMPTS.items():
        plan_dir = root / stage / "stage_plans" / attempts[key]
        tasks = list(iter_jsonl(plan_dir / "tasks.jsonl"))
        profiles[stage] = {
            "resources": tasks[0]["resources"] if tasks else {},
            "task_count": len(tasks),
        }
        for task in tasks:
            result_dir = Path(task["result_dir"])
            start = _read_json_object(result_dir / "run_start.json")
            git = start.get("git")
            if isinstance(git, dict) and git.get("sha"):
                worker_commits.add(str(git["sha"]))
    return {
        "fanout_populations": {
            stage: int(spec["count"])
            for stage, spec in STAGE_EXPECTATIONS.items()
        },
        "worker_commits": sorted(worker_commits),
        "profiles": profiles,
        "terminal_audit": "passed",
    }


def _figure_metadata(
    root: Path,
    attempts: Mapping[str, str],
) -> list[dict[str, Any]]:
    report_dir = root / "09_final_report" / attempts["report"]
    report = _read_json_object(report_dir / "final_report.json")
    raw_figures = report.get("figures")
    if not isinstance(raw_figures, list) or not all(
        isinstance(item, str) for item in raw_figures
    ):
        raise ValueError("final report figures are not a list of filenames")
    rows: list[dict[str, Any]] = []
    for name in raw_figures:
        relative = _safe_nested_path(name)
        path = _required_regular_file(
            report_dir / "figures" / relative,
            root=root,
        )
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"final report figure is empty: {path}")
        rows.append(
            {
                "logical_path": (
                    "09_final_report/{report}/figures/"
                    + relative.as_posix()
                ),
                "size": size,
            }
        )
    return rows


def _source_provenance() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("reference source checkout is dirty")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit or not branch:
        raise ValueError("reference source commit/branch is unavailable")
    return {"commit": commit, "branch": branch, "dirty": False}


def _require_worker_commit(
    root: Path,
    attempts: Mapping[str, str],
    expected: str,
) -> None:
    commits = set(_lineage_summary(root, attempts)["worker_commits"])
    if commits != {expected}:
        raise ValueError(
            f"worker commits {sorted(commits)!r} do not equal source {expected}"
        )


def _logical_path(
    relative: Path,
    attempts: Mapping[str, str],
) -> str:
    parts = list(relative.parts)
    stage = parts[0]
    if stage in LOW_STAGE_ATTEMPTS:
        key = LOW_STAGE_ATTEMPTS[stage]
        if len(parts) < 2 or parts[1] != attempts[key]:
            raise ValueError(f"protected path attempt mismatch: {relative}")
        parts[1] = "{" + key + "}"
    elif stage in FANOUT_ATTEMPTS:
        key = FANOUT_ATTEMPTS[stage]
        if len(parts) >= 4 and parts[1] == "stage_plans":
            if parts[2] != attempts[key]:
                raise ValueError(
                    f"fan-out protected path attempt mismatch: {relative}"
                )
            parts[2] = "{" + key + "}"
        elif len(parts) == 4 and parts[2] == attempts[key]:
            parts[2] = "{" + key + "}"
        else:
            raise ValueError(
                f"fan-out protected path is not a stage plan/submission: {relative}"
            )
    else:
        raise ValueError(f"unsupported protected stage: {stage}")
    return Path(*parts).as_posix()


def _logical_role(relative: Path) -> str:
    stage = relative.parts[0]
    suffix = relative.suffix.lower().removeprefix(".") or "text"
    if "jobs" in relative.parts:
        return f"{stage}:job:{suffix}"
    if stage == "08_final_collect" and relative.suffix.lower() in {
        ".csv",
        ".tsv",
    }:
        return f"{stage}:table:{suffix}"
    if "tables" in relative.parts:
        return f"{stage}:table:{suffix}"
    return f"{stage}:{relative.name}:{suffix}"


def _normalize_attempts(attempts: Mapping[str, str]) -> dict[str, str]:
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
    if not isinstance(attempts, Mapping) or set(attempts) != required:
        actual = set(attempts) if isinstance(attempts, Mapping) else set()
        raise ValueError(
            "attempt map mismatch; "
            f"missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required)}"
        )
    return {
        key: validate_lineage_id(str(attempts[key]))
        for key in sorted(required)
    }


def _isolated_absolute_root(path: Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("results root must be absolute without traversal")
    if requested.is_symlink():
        raise ValueError("results root may not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("results root must be a directory")
    return root


def _reference_destination(destination: Path, *, owner: Path) -> Path:
    """Return one new reference child below an explicit V4-owned namespace."""

    requested = Path(destination)
    owner = Path(owner)
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("reference destination must be absolute without traversal")
    if not owner.is_absolute() or ".." in owner.parts:
        raise ValueError("reference owner must be absolute without traversal")
    if owner.is_symlink():
        raise ValueError("reference owner may not be a symlink")
    owner.mkdir(parents=True, exist_ok=True)
    canonical_owner = owner.resolve(strict=True)
    if not canonical_owner.is_dir() or canonical_owner.is_symlink():
        raise ValueError("reference owner is not a regular directory")
    if requested.parent.resolve(strict=False) != canonical_owner:
        raise ValueError(
            "reference destination must be a direct child of its V4 reference owner"
        )
    reference_id = validate_lineage_id(requested.name)
    canonical_destination = canonical_owner / reference_id
    if canonical_destination.exists() or canonical_destination.is_symlink():
        raise FileExistsError(f"reference destination exists: {canonical_destination}")
    return canonical_destination


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacement on Linux."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - Cannon is Linux
        raise RuntimeError("atomic no-replace reference publication is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure == errno.EEXIST:
        raise FileExistsError(f"reference destination exists: {destination}")
    raise OSError(failure, os.strerror(failure), destination)


def _required_regular_file(path: Path, *, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"protected artifact is not a regular file: {path}")
    canonical = path.resolve(strict=True)
    if canonical == root or root not in canonical.parents:
        raise ValueError(f"protected artifact escapes lineage root: {path}")
    return canonical


def _table_metadata(
    path: Path,
    *,
    delimiter: str,
) -> tuple[tuple[str, ...], int, Mapping[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = tuple(next(reader, ()))
        if len(set(header)) != len(header):
            raise ValueError(f"duplicate table header: {path}")
        values: dict[str, list[str]] = {column: [] for column in header}
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"ragged table row in {path}")
            count += 1
            for column, value in zip(header, row, strict=True):
                if value != "":
                    values[column].append(value)
    return (
        header,
        count,
        {column: _column_type(items) for column, items in values.items()},
    )


def _column_type(values: Sequence[str]) -> str:
    if not values:
        return "empty"
    if all(value.lower() in {"true", "false"} for value in values):
        return "boolean"
    if all(re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", value) for value in values):
        return "integer"
    try:
        for value in values:
            float(value)
    except ValueError:
        return "string"
    return "float"


def _manifest_table_names(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("final-collect manifest tables are not a mapping")
    return tuple(_safe_nested_path(name) for name in tables)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    value = load_yaml(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError(f"invalid CSV header: {path}")
        return list(reader)


def _safe_component(value: object, name: str) -> str:
    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
    ):
        raise ValueError(f"invalid {name}: {text!r}")
    return text


def _safe_nested_path(value: object) -> Path:
    path = Path(str(value))
    if (
        not str(value)
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"unsafe nested artifact path: {value!r}")
    return path


def _safe_relative_text(value: object) -> str:
    return _safe_nested_path(value).as_posix()


def _digest_text(value: object) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError("invalid SHA-256 digest")
    return text


def _media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".md": "text/markdown",
        ".sh": "text/x-shellscript",
    }.get(path.suffix.lower(), "text/plain")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _parse_cli_attempts(value: str) -> dict[str, str]:
    parsed = loads_json(value, source="--attempts")
    if not isinstance(parsed, dict):
        raise ValueError("--attempts must be a JSON object")
    return _normalize_attempts(
        {str(key): str(item) for key, item in parsed.items()}
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze or verify one immutable V3 parity reference."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--results-root", type=Path, required=True)
    freeze.add_argument("--destination", type=Path, required=True)
    freeze.add_argument("--attempts", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            destination = freeze_reference(
                args.results_root,
                args.destination,
                attempts=_parse_cli_attempts(args.attempts),
            )
            print(destination)
            return 0
        errors = verify_reference(args.reference)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
