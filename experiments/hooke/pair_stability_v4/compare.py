"""Streaming, fail-closed V3-reference to V4-candidate comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import yaml

import control_audit
from audit import (
    audit_completed_lineage,
    reference_evidence as summarize_reference_evidence,
)
from layout import (
    COMPARATOR_SCHEMA_VERSION,
    DEFAULT_LAYOUT_MAP,
    TOKEN_SUBSTITUTIONS,
    ArtifactPolicy,
    JsonRecordArrayPolicy,
    LayoutMap,
    load_layout_map as _load_layout_map,
    materialize_layout,
)
from reference import (
    COMPARISON_LAYOUT_LOGICAL_PATH,
    REFERENCE_COMPARISON_SCHEMA_VERSION,
    ReferenceArtifact,
    enumerate_inventory,
    load_reference,
    open_logical_content,
)
from roots import require_beneath_root, require_v4_root, root_metadata, validate_lineage_id
from strict_data import StrictDataError, loads_json, loads_yaml

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
V3_STUDY_DIR = STUDY_DIR.parent / "pair_stability_v3"
MAX_DIFFERENCES = 100
MAX_VALUE_CHARS = 240
FINITE_DECIMAL_STRING = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
ATTEMPT_KEYS = {
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
TokenMap = Mapping[
    str,
    Mapping[str, tuple[tuple[str, str], ...]],
]


@dataclass(frozen=True)
class Difference:
    """Describe one bounded machine-readable mismatch."""

    artifact: str
    location: str
    kind: str
    message: str
    reference: str = ""
    candidate: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return the stable report representation."""

        return asdict(self)


class _DifferenceSink:
    def __init__(self, *, limit: int = MAX_DIFFERENCES) -> None:
        self.limit = limit
        self.values: list[Difference] = []
        self.truncated = False

    def add(
        self,
        *,
        artifact: str,
        location: str,
        kind: str,
        message: str,
        reference: object = "",
        candidate: object = "",
    ) -> None:
        if len(self.values) >= self.limit:
            if not self.truncated:
                self.values[-1] = Difference(
                    artifact="<comparison>",
                    location="/",
                    kind="truncated",
                    message=(
                        f"difference output exceeded limit {self.limit}"
                    ),
                )
                self.truncated = True
            return
        self.values.append(
            Difference(
                artifact=artifact,
                location=location,
                kind=kind,
                message=message,
                reference=_bounded(reference),
                candidate=_bounded(candidate),
            )
        )


def load_layout_map(path: Path = DEFAULT_LAYOUT_MAP) -> LayoutMap:
    """Validate and return the complete versioned layout map."""

    return _load_layout_map(path)


def compare_reference(
    reference_dir: Path,
    candidate_root: Path,
    *,
    candidate_attempts: Mapping[str, str],
    layout_map_path: Path = DEFAULT_LAYOUT_MAP,
) -> tuple[Difference, ...]:
    """Compare one verified frozen V3 inventory with an audited V4 lineage."""

    descriptor, reference_artifacts = load_reference(reference_dir)
    layout = load_layout_map(layout_map_path)
    attempts = _normalize_attempts(candidate_attempts)
    candidate = Path(candidate_root).resolve(strict=True)
    _reject_same_source(
        descriptor,
        candidate,
        candidate_attempts=attempts,
    )
    _require_bound_layout(descriptor, layout)
    candidate = require_v4_root(candidate)

    sink = _DifferenceSink()
    control_errors = control_audit.audit_control_closure(
        candidate,
        attempts=attempts,
    )
    for error in control_errors:
        sink.add(
            artifact="<control>",
            location="/",
            kind="control_audit",
            message=error,
        )
    if control_errors:
        return tuple(sink.values)
    control = control_audit.control_provenance(
        candidate,
        attempts=attempts,
    )
    for error in _closure_differences(descriptor, control):
        sink.add(
            artifact="<closure>",
            location="/",
            kind="closure",
            message=error,
        )
    if sink.values:
        return tuple(sink.values)
    audit_errors = audit_completed_lineage(candidate, attempts=attempts)
    for error in audit_errors:
        sink.add(
            artifact="<candidate>",
            location="/",
            kind="candidate_audit",
            message=error,
        )
    if audit_errors:
        return tuple(sink.values)

    try:
        candidate_inventory = enumerate_inventory(
            candidate,
            attempts=attempts,
        )
        candidate_by_logical = {
            _logical_path(path.relative_to(candidate), attempts): path
            for path in candidate_inventory
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sink.add(
            artifact="<candidate>",
            location="/",
            kind="candidate_inventory",
            message=str(exc),
        )
        return tuple(sink.values)
    if len(candidate_by_logical) != len(candidate_inventory):
        sink.add(
            artifact="<candidate>",
            location="/",
            kind="candidate_inventory",
            message="candidate logical inventory contains duplicates",
        )
        return tuple(sink.values)

    return _compare_reference_contents(
        descriptor,
        reference_artifacts,
        candidate,
        candidate_by_logical,
        candidate_attempts=attempts,
        layout=layout,
    )


def _compare_reference_contents(
    descriptor: Mapping[str, Any],
    reference_artifacts: Sequence[ReferenceArtifact],
    candidate_root: Path,
    candidate_by_logical: Mapping[str, Path],
    *,
    candidate_attempts: Mapping[str, str],
    layout: LayoutMap,
) -> tuple[Difference, ...]:
    """Compare verified contents after public acceptance guards have run.

    This private boundary permits exact self-comparison in unit tests. It must
    never be exposed as an acceptance CLI bypass.
    """

    sink = _DifferenceSink()
    reference_by_logical = {
        artifact.logical_path: artifact for artifact in reference_artifacts
    }
    try:
        expansions = _semantic_expansions(reference_artifacts)
        policies = materialize_layout(
            layout,
            expansions=expansions,
            reference_paths=set(reference_by_logical),
            candidate_paths=set(candidate_by_logical),
        )
    except ValueError as exc:
        sink.add(
            artifact="<layout>",
            location="/",
            kind="layout",
            message=str(exc),
        )
        return tuple(sink.values)

    tokens = _comparison_tokens(
        descriptor,
        reference_by_logical,
        candidate_root,
        candidate_by_logical,
        candidate_attempts=candidate_attempts,
    )
    for policy in policies:
        if sink.truncated:
            break
        reference_artifact = reference_by_logical[
            policy.reference_logical_path
        ]
        candidate_path = candidate_by_logical[
            policy.candidate_logical_path
        ]
        _compare_artifact(
            reference_artifact,
            candidate_path,
            policy=policy,
            layout=layout,
            tokens=tokens,
            sink=sink,
        )
    return tuple(sink.values)


def write_comparison_report(
    candidate_root: Path,
    comparison_id: str,
    differences: Sequence[Difference],
    *,
    provenance: Mapping[str, object],
) -> Path:
    """Create one guarded, immutable comparison report below candidate root."""

    root = require_v4_root(Path(candidate_root).resolve(strict=True))
    comparison_id = validate_lineage_id(comparison_id)
    comparison_root = require_beneath_root(
        root / "_v4" / "comparison" / comparison_id,
        root,
    )
    if comparison_root.exists() or comparison_root.is_symlink():
        raise FileExistsError(
            f"comparison evidence already exists: {comparison_root}"
        )
    comparison_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(comparison_root)
    target = require_beneath_root(comparison_root / "comparison.json", root)
    payload = {
        "schema_version": COMPARATOR_SCHEMA_VERSION,
        "outcome": "passed" if not differences else "failed",
        "n_differences": len(differences),
        "provenance": dict(provenance),
        "differences": [difference.to_dict() for difference in differences],
    }
    descriptor = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(handle, "w", encoding="utf-8") as output:
        output.write(descriptor)
        output.flush()
        os.fsync(output.fileno())
    return target


def comparison_provenance(
    reference_dir: Path,
    candidate_root: Path,
    *,
    candidate_attempts: Mapping[str, str],
    layout_map_path: Path = DEFAULT_LAYOUT_MAP,
    allow_incomplete_candidate: bool = False,
) -> dict[str, object]:
    """Return compact, digest-backed inputs for an acceptance report."""

    reference_root = Path(reference_dir).resolve(strict=True)
    candidate = require_v4_root(Path(candidate_root).resolve(strict=True))
    attempts = _normalize_attempts(candidate_attempts)
    descriptor, artifacts = load_reference(reference_root)
    layout = load_layout_map(layout_map_path)
    _reject_same_source(
        descriptor,
        candidate,
        candidate_attempts=attempts,
    )
    _require_bound_layout(descriptor, layout)

    try:
        control: dict[str, object] = control_audit.control_provenance(
            candidate,
            attempts=attempts,
        )
        closure_errors = _closure_differences(descriptor, control)
        if closure_errors:
            raise ValueError(
                "candidate/reference closure mismatch: "
                + "; ".join(closure_errors)
            )
    except ValueError as exc:
        if not allow_incomplete_candidate:
            raise
        control = {
            "schema_version": control_audit.CONTROL_SCHEMA_VERSION,
            "verification_status": "unavailable",
            "error": str(exc),
        }

    logical_paths = [artifact.logical_path for artifact in artifacts]
    reference_value = {
        "path": str(reference_root),
        "descriptor_sha256": _sha256_file(
            reference_root / "reference.json"
        ),
        "attempts": dict(descriptor["attempts"]),
        "source": descriptor["source"],
        "runtime_closure_sha256": descriptor["runtime_closure"][
            "closure_sha256"
        ],
        "legacy_closure_sha256": descriptor["legacy_closure"][
            "closure_sha256"
        ],
        "config_closure_sha256": descriptor["config_closure"][
            "closure_sha256"
        ],
        "artifact_count": len(artifacts),
        "logical_paths_sha256": _canonical_sha256(logical_paths),
    }
    candidate_value: dict[str, object] = {
        "root": str(candidate),
        "root_metadata": root_metadata(candidate),
        "attempts": attempts,
        "control": control,
    }
    try:
        inventory = enumerate_inventory(candidate, attempts=attempts)
        candidate_logical = [
            _logical_path(path.relative_to(candidate), attempts)
            for path in inventory
        ]
        evidence = summarize_reference_evidence(
            candidate,
            attempts=attempts,
        )
        worker = evidence["worker_runtime"]
        stages = worker["stages"]
        candidate_value.update(
            {
                "artifact_count": len(inventory),
                "logical_paths_sha256": _canonical_sha256(
                    candidate_logical
                ),
                "science_metric_schema_sha256": evidence[
                    "science_metrics"
                ]["schema_sha256"],
                "selection_contract_sha256": evidence["selection"][
                    "contract_sha256"
                ],
                "slurm_evidence": {
                    "worker_runtime_aggregate_sha256": worker[
                        "aggregate_sha256"
                    ],
                    "stages": {
                        stage: {
                            "task_count": value["task_count"],
                            "profiles": [
                                {
                                    "profile_sha256": profile[
                                        "profile_sha256"
                                    ],
                                    "count": profile["count"],
                                }
                                for profile in value["profiles"]
                            ],
                        }
                        for stage, value in sorted(stages.items())
                    },
                },
                "evidence_status": "verified",
            }
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if not allow_incomplete_candidate:
            raise
        candidate_value.update(
            {
                "evidence_status": "unavailable",
                "evidence_error": str(exc),
            }
        )
    return {
        "schema_version": COMPARATOR_SCHEMA_VERSION,
        "reference": reference_value,
        "candidate": candidate_value,
        "layout_map": {
            "path": str(layout.source_path),
            "schema_version": layout.schema_version,
            "sha256": layout.sha256,
            "rel_tol": layout.rel_tol,
            "abs_tol": layout.abs_tol,
        },
        "comparator": {
            "schema_version": COMPARATOR_SCHEMA_VERSION,
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": _sha256_file(Path(__file__).resolve()),
        },
    }


def _closure_differences(
    descriptor: Mapping[str, Any],
    control: Mapping[str, object],
) -> tuple[str, ...]:
    """Compare only reviewed closure facts, never repository commit identity."""

    projection = control.get("closure_projection")
    if not isinstance(projection, Mapping):
        return ("candidate control closure projection is unavailable",)
    expected = {
        "legacy_closure_sha256": descriptor.get("legacy_closure", {}).get(
            "closure_sha256"
        ) if isinstance(descriptor.get("legacy_closure"), Mapping) else None,
        "runtime_closure_sha256": descriptor.get("runtime_closure", {}).get(
            "closure_sha256"
        ) if isinstance(descriptor.get("runtime_closure"), Mapping) else None,
        "config_closure_sha256": descriptor.get("config_closure", {}).get(
            "closure_sha256"
        ) if isinstance(descriptor.get("config_closure"), Mapping) else None,
    }
    errors: list[str] = []
    for key, reference_value in expected.items():
        if not isinstance(reference_value, str) or not reference_value:
            errors.append(f"reference {key} is unavailable")
        elif projection.get(key) != reference_value:
            errors.append(f"candidate {key} differs from frozen reference")
    return tuple(errors)


def _compare_artifact(
    reference: ReferenceArtifact,
    candidate: Path,
    *,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
) -> None:
    logical = reference.logical_path
    expected_format = {
        "application/json": "json",
        "application/x-ndjson": "jsonl",
        "application/yaml": "yaml",
        "text/csv": "csv",
        "text/markdown": "markdown",
        "text/x-shellscript": "text",
        "text/plain": "text",
    }.get(reference.media_type)
    if policy.format != expected_format:
        sink.add(
            artifact=logical,
            location="/",
            kind="policy",
            message=(
                f"layout format {policy.format!r} differs from "
                f"reference media type {reference.media_type!r}"
            ),
        )
        return
    if candidate.is_symlink() or not candidate.is_file():
        sink.add(
            artifact=logical,
            location="/",
            kind="missing",
            message="candidate artifact is not a regular file",
        )
        return
    if policy.presence_only:
        if candidate.stat().st_size <= 0:
            sink.add(
                artifact=logical,
                location="/",
                kind="empty",
                message="presence-only candidate artifact is empty",
            )
        return
    try:
        with open_logical_content(reference) as left:
            if policy.format == "csv":
                with candidate.open("rb") as right:
                    _compare_csv_streams(
                        left,
                        right,
                        policy=policy,
                        layout=layout,
                        tokens=tokens,
                        sink=sink,
                    )
            elif policy.format == "jsonl":
                with candidate.open("rb") as right:
                    _compare_jsonl_streams(
                        left,
                        right,
                        policy=policy,
                        layout=layout,
                        tokens=tokens,
                        sink=sink,
                    )
            else:
                left_bytes = left.read()
                right_bytes = candidate.read_bytes()
                _compare_small_content(
                    left_bytes,
                    right_bytes,
                    policy=policy,
                    layout=layout,
                    tokens=tokens,
                    sink=sink,
                )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        sink.add(
            artifact=logical,
            location="/",
            kind="parse",
            message=str(exc),
        )


def _compare_small_content(
    left_bytes: bytes,
    right_bytes: bytes,
    *,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
) -> None:
    logical = policy.reference_logical_path
    left_text = left_bytes.decode("utf-8")
    right_text = right_bytes.decode("utf-8")
    if policy.format == "json":
        left = _strict_json_loads(left_text)
        right = _strict_json_loads(right_text)
    elif policy.format == "yaml":
        left = loads_yaml(left_text, source=f"reference {logical}")
        right = loads_yaml(right_text, source=f"candidate {logical}")
    elif policy.format in {"text", "markdown"}:
        left = _normalize_string(
            left_text,
            tokens=tokens,
            approved=policy.approved_token_substitutions,
            side="reference",
        )
        right = _normalize_string(
            right_text,
            tokens=tokens,
            approved=policy.approved_token_substitutions,
            side="candidate",
        )
        if left != right:
            sink.add(
                artifact=logical,
                location="/",
                kind="value",
                message="text content differs",
                reference=left,
                candidate=right,
            )
        return
    else:
        raise ValueError(f"unsupported small-content format {policy.format!r}")
    _compare_structured_value(
        left,
        right,
        artifact=logical,
        policy=policy,
        layout=layout,
        tokens=tokens,
        sink=sink,
    )


def _compare_jsonl_streams(
    left_binary: BinaryIO,
    right_binary: BinaryIO,
    *,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
) -> None:
    left_text = io.TextIOWrapper(left_binary, encoding="utf-8")
    right_text = io.TextIOWrapper(right_binary, encoding="utf-8")
    for row_index, pair in enumerate(
        itertools.zip_longest(left_text, right_text),
    ):
        left_line, right_line = pair
        if left_line is None or right_line is None:
            sink.add(
                artifact=policy.reference_logical_path,
                location=f"/{row_index}",
                kind="row_count",
                message="JSONL row count differs",
            )
            return
        left = _strict_json_loads(left_line)
        right = _strict_json_loads(right_line)
        _compare_structured_value(
            left,
            right,
            artifact=policy.reference_logical_path,
            policy=policy,
            layout=layout,
            tokens=tokens,
            sink=sink,
            location_prefix=f"/{row_index}",
        )


def _compare_csv_streams(
    left_binary: BinaryIO,
    right_binary: BinaryIO,
    *,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
) -> None:
    left_text = io.TextIOWrapper(left_binary, encoding="utf-8", newline="")
    right_text = io.TextIOWrapper(right_binary, encoding="utf-8", newline="")
    left_reader = csv.reader(left_text)
    right_reader = csv.reader(right_text)
    left_header = next(left_reader, None)
    right_header = next(right_reader, None)
    if (
        left_header is None
        or right_header is None
        or len(set(left_header)) != len(left_header)
        or len(set(right_header)) != len(right_header)
    ):
        sink.add(
            artifact=policy.reference_logical_path,
            location="/header",
            kind="header",
            message="CSV header is missing or duplicated",
            reference=left_header,
            candidate=right_header,
        )
        return
    if left_header != right_header:
        sink.add(
            artifact=policy.reference_logical_path,
            location="/header",
            kind="header",
            message="CSV header names/order differ",
            reference=left_header,
            candidate=right_header,
        )
        return
    header = left_header
    volatile = set(policy.volatile_csv_columns)
    tolerant = set(policy.float_tolerant_csv_columns)
    unknown = (volatile | tolerant) - set(header)
    if unknown:
        sink.add(
            artifact=policy.reference_logical_path,
            location="/header",
            kind="policy",
            message=f"CSV policy names absent columns {sorted(unknown)!r}",
        )
        return
    for row_index, pair in enumerate(
        itertools.zip_longest(left_reader, right_reader),
        start=1,
    ):
        left_row, right_row = pair
        if left_row is None or right_row is None:
            sink.add(
                artifact=policy.reference_logical_path,
                location=f"/rows/{row_index}",
                kind="row_count",
                message="CSV row count differs",
            )
            return
        if len(left_row) != len(header) or len(right_row) != len(header):
            sink.add(
                artifact=policy.reference_logical_path,
                location=f"/rows/{row_index}",
                kind="row_width",
                message="CSV row width differs from header",
            )
            continue
        for column, left, right in zip(
            header,
            left_row,
            right_row,
            strict=True,
        ):
            if column in volatile:
                continue
            left = _normalize_string(
                left,
                tokens=tokens,
                approved=policy.approved_token_substitutions,
                side="reference",
            )
            right = _normalize_string(
                right,
                tokens=tokens,
                approved=policy.approved_token_substitutions,
                side="candidate",
            )
            equal = (
                left == right
                if column not in tolerant or left == ""
                else _csv_float_equal(
                    left,
                    right,
                    rel_tol=layout.rel_tol,
                    abs_tol=layout.abs_tol,
                )
            )
            if not equal:
                sink.add(
                    artifact=policy.reference_logical_path,
                    location=f"/rows/{row_index}/{_pointer_token(column)}",
                    kind="float" if column in tolerant else "value",
                    message="CSV cell differs",
                    reference=left,
                    candidate=right,
                )


def _compare_structured_value(
    left: object,
    right: object,
    *,
    artifact: str,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
    location_prefix: str = "",
) -> None:
    seen_pointers: set[str] = set()
    seen_record_arrays: set[str] = set()
    record_arrays = {
        record.array_pointer: record
        for record in policy.json_record_arrays
    }
    _compare_value(
        left,
        right,
        pointer="",
        artifact=artifact,
        policy=policy,
        layout=layout,
        tokens=tokens,
        sink=sink,
        location_prefix=location_prefix,
        seen_pointers=seen_pointers,
        seen_record_arrays=seen_record_arrays,
        record_arrays=record_arrays,
    )
    absent = (
        set(policy.volatile_json_pointers)
        | set(policy.float_tolerant_json_pointers)
        | set(policy.float_string_tolerant_json_pointers)
    ) - seen_pointers
    if absent:
        sink.add(
            artifact=artifact,
            location=f"{location_prefix}/",
            kind="policy",
            message=f"JSON policy pointers are absent: {sorted(absent)!r}",
        )
    absent_arrays = set(record_arrays) - seen_record_arrays
    if absent_arrays:
        sink.add(
            artifact=artifact,
            location=f"{location_prefix}/",
            kind="policy",
            message=(
                "JSON record-array policy pointers are absent: "
                f"{sorted(absent_arrays)!r}"
            ),
        )


def _compare_value(
    left: object,
    right: object,
    *,
    pointer: str,
    artifact: str,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
    location_prefix: str = "",
    seen_pointers: set[str],
    seen_record_arrays: set[str],
    record_arrays: Mapping[str, JsonRecordArrayPolicy],
) -> None:
    location = f"{location_prefix}{pointer or '/'}"
    seen_pointers.add(pointer)
    record_policy = record_arrays.get(pointer)
    if record_policy is not None:
        seen_record_arrays.add(pointer)
        _compare_record_array(
            left,
            right,
            pointer=pointer,
            artifact=artifact,
            record_policy=record_policy,
            policy=policy,
            layout=layout,
            tokens=tokens,
            sink=sink,
            location_prefix=location_prefix,
            seen_pointers=seen_pointers,
            seen_record_arrays=seen_record_arrays,
            record_arrays=record_arrays,
        )
        return
    if pointer in policy.volatile_json_pointers:
        return
    if pointer in policy.float_string_tolerant_json_pointers:
        _compare_float_string(
            left,
            right,
            artifact=artifact,
            location=location,
            layout=layout,
            sink=sink,
        )
        return
    if (
        _json_number(left)
        and not math.isfinite(float(left))
    ) or (
        _json_number(right)
        and not math.isfinite(float(right))
    ):
        sink.add(
            artifact=artifact,
            location=location,
            kind="nonfinite",
            message="non-finite structured numeric value is forbidden",
            reference=left,
            candidate=right,
        )
        return
    if type(left) is not type(right):
        if pointer in policy.float_tolerant_json_pointers and _json_number(
            left
        ) and _json_number(right):
            if _float_equal(
                float(left),
                float(right),
                rel_tol=layout.rel_tol,
                abs_tol=layout.abs_tol,
            ):
                return
        sink.add(
            artifact=artifact,
            location=location,
            kind="type",
            message="JSON types differ",
            reference=type(left).__name__,
            candidate=type(right).__name__,
        )
        return
    if isinstance(left, dict):
        if left.keys() != right.keys():
            sink.add(
                artifact=artifact,
                location=location,
                kind="keys",
                message="JSON object key sets differ",
                reference=sorted(left),
                candidate=sorted(right),
            )
            return
        for key in left:
            child = f"{pointer}/{_pointer_token(str(key))}"
            _compare_value(
                left[key],
                right[key],
                pointer=child,
                artifact=artifact,
                policy=policy,
                layout=layout,
                tokens=tokens,
                sink=sink,
                location_prefix=location_prefix,
                seen_pointers=seen_pointers,
                seen_record_arrays=seen_record_arrays,
                record_arrays=record_arrays,
            )
        return
    if isinstance(left, list):
        if len(left) != len(right):
            sink.add(
                artifact=artifact,
                location=location,
                kind="length",
                message="JSON array lengths differ",
                reference=len(left),
                candidate=len(right),
            )
            return
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _compare_value(
                left_item,
                right_item,
                pointer=f"{pointer}/{index}",
                artifact=artifact,
                policy=policy,
                layout=layout,
                tokens=tokens,
                sink=sink,
                location_prefix=location_prefix,
                seen_pointers=seen_pointers,
                seen_record_arrays=seen_record_arrays,
                record_arrays=record_arrays,
            )
        return
    if isinstance(left, str):
        left = _normalize_string(
            left,
            tokens=tokens,
            approved=policy.approved_token_substitutions,
            side="reference",
        )
        right = _normalize_string(
            right,
            tokens=tokens,
            approved=policy.approved_token_substitutions,
            side="candidate",
        )
    if pointer in policy.float_tolerant_json_pointers:
        equal = (
            left == right
            if left is None
            else (
                _json_number(left)
                and _json_number(right)
                and _float_equal(
                    float(left),
                    float(right),
                    rel_tol=layout.rel_tol,
                    abs_tol=layout.abs_tol,
                )
            )
        )
    else:
        equal = left == right
    if not equal:
        sink.add(
            artifact=artifact,
            location=location,
            kind=(
                "float"
                if pointer in policy.float_tolerant_json_pointers
                else "value"
            ),
            message="JSON value differs",
            reference=left,
            candidate=right,
        )


def _compare_record_array(
    left: object,
    right: object,
    *,
    pointer: str,
    artifact: str,
    record_policy: JsonRecordArrayPolicy,
    policy: ArtifactPolicy,
    layout: LayoutMap,
    tokens: TokenMap,
    sink: _DifferenceSink,
    location_prefix: str,
    seen_pointers: set[str],
    seen_record_arrays: set[str],
    record_arrays: Mapping[str, JsonRecordArrayPolicy],
) -> None:
    """Compare one ordered JSON array using literal per-record fields."""

    location = f"{location_prefix}{pointer}"
    if type(left) is not list or type(right) is not list:
        sink.add(
            artifact=artifact,
            location=location,
            kind="type",
            message="JSON record-array value is not an array on both sides",
            reference=type(left).__name__,
            candidate=type(right).__name__,
        )
        return
    if len(left) != len(right):
        sink.add(
            artifact=artifact,
            location=location,
            kind="length",
            message="JSON record-array lengths differ",
            reference=len(left),
            candidate=len(right),
        )
        return
    governed = {
        *record_policy.volatile_fields,
        *record_policy.float_string_tolerant_fields,
    }
    if not left:
        sink.add(
            artifact=artifact,
            location=location,
            kind="policy",
            message="JSON record-array policy is unused on an empty array",
        )
        return
    for index, (left_row, right_row) in enumerate(
        zip(left, right, strict=True)
    ):
        row_location = f"{location}/{index}"
        if type(left_row) is not dict or type(right_row) is not dict:
            sink.add(
                artifact=artifact,
                location=row_location,
                kind="type",
                message="JSON record-array row is not an object on both sides",
                reference=type(left_row).__name__,
                candidate=type(right_row).__name__,
            )
            continue
        missing_left = governed - set(left_row)
        missing_right = governed - set(right_row)
        if missing_left or missing_right:
            sink.add(
                artifact=artifact,
                location=row_location,
                kind="policy",
                message="JSON record-array policy fields are absent",
                reference=sorted(missing_left),
                candidate=sorted(missing_right),
            )
            continue
        if left_row.keys() != right_row.keys():
            sink.add(
                artifact=artifact,
                location=row_location,
                kind="keys",
                message="JSON record-array row key sets differ",
                reference=sorted(left_row),
                candidate=sorted(right_row),
            )
            continue
        for field in left_row:
            child = f"{pointer}/{index}/{_pointer_token(str(field))}"
            if field in record_policy.volatile_fields:
                continue
            if field in record_policy.float_string_tolerant_fields:
                _compare_float_string(
                    left_row[field],
                    right_row[field],
                    artifact=artifact,
                    location=f"{location_prefix}{child}",
                    layout=layout,
                    sink=sink,
                )
                continue
            _compare_value(
                left_row[field],
                right_row[field],
                pointer=child,
                artifact=artifact,
                policy=policy,
                layout=layout,
                tokens=tokens,
                sink=sink,
                location_prefix=location_prefix,
                seen_pointers=seen_pointers,
                seen_record_arrays=seen_record_arrays,
                record_arrays=record_arrays,
            )


def _compare_float_string(
    left: object,
    right: object,
    *,
    artifact: str,
    location: str,
    layout: LayoutMap,
    sink: _DifferenceSink,
) -> None:
    """Compare strict finite-decimal strings under the approved tolerance."""

    if type(left) is not str or type(right) is not str:
        sink.add(
            artifact=artifact,
            location=location,
            kind="type",
            message="float-string policy requires strings on both sides",
            reference=type(left).__name__,
            candidate=type(right).__name__,
        )
        return
    if left == "" or right == "":
        equal = left == right
    elif (
        FINITE_DECIMAL_STRING.fullmatch(left) is None
        or FINITE_DECIMAL_STRING.fullmatch(right) is None
    ):
        sink.add(
            artifact=artifact,
            location=location,
            kind="float_string",
            message="float-string value is not a strict finite decimal",
            reference=left,
            candidate=right,
        )
        return
    else:
        left_float = float(left)
        right_float = float(right)
        equal = (
            math.isfinite(left_float)
            and math.isfinite(right_float)
            and _float_equal(
                left_float,
                right_float,
                rel_tol=layout.rel_tol,
                abs_tol=layout.abs_tol,
            )
        )
    if not equal:
        sink.add(
            artifact=artifact,
            location=location,
            kind="float_string",
            message="finite-decimal string values differ",
            reference=left,
            candidate=right,
        )


def _comparison_tokens(
    descriptor: Mapping[str, Any],
    reference_artifacts: Mapping[str, ReferenceArtifact],
    candidate_root: Path,
    candidate_paths: Mapping[str, Path],
    *,
    candidate_attempts: Mapping[str, str],
) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
    reference_attempts = descriptor["attempts"]
    reference_root = str(descriptor["source_results"]["canonical_root"])
    values: dict[str, dict[str, dict[str, str]]] = {
        side: {category: {} for category in sorted(TOKEN_SUBSTITUTIONS)}
        for side in ("reference", "candidate")
    }

    def add(kind: str, left: object, right: object, token: str) -> None:
        if isinstance(left, str) and left:
            values["reference"][kind][left] = token
        if isinstance(right, str) and right:
            values["candidate"][kind][right] = token

    add("results_root", reference_root, str(candidate_root), "<RESULTS_ROOT>")
    add(
        "study_identity",
        str(descriptor.get("study") or "pair_stability_v3"),
        "pair_stability_v4",
        "<STUDY>",
    )
    # Synthetic source fixtures exercise the private content comparator with
    # V4-shaped raw artifacts stored in a V3-labelled reference descriptor.
    values["reference"]["study_identity"]["pair_stability_v4"] = "<STUDY>"
    values["reference"]["study_identity"][
        "Pair Stability V4"
    ] = "Pair Stability <STUDY_VERSION>"
    add(
        "study_identity",
        "Pair Stability V3",
        "Pair Stability V4",
        "Pair Stability <STUDY_VERSION>",
    )
    add(
        "study_identity",
        "pair-stability V3",
        "pair-stability V4",
        "pair-stability <STUDY_VERSION>",
    )
    add(
        "study_path",
        str(V3_STUDY_DIR),
        str(STUDY_DIR),
        "<STUDY_DIR>",
    )
    add(
        "study_path",
        "experiments/hooke/pair_stability_v3",
        "experiments/hooke/pair_stability_v4",
        "<STUDY_DIR_REL>",
    )
    # Synthetic source fixtures contain V4-shaped files under the descriptor's
    # V3 identity.  Keep those exact aliases on the reference side; this does
    # not broaden the approved category or match patterns.
    values["reference"]["study_path"][str(STUDY_DIR)] = "<STUDY_DIR>"
    values["reference"]["study_path"][
        "experiments/hooke/pair_stability_v4"
    ] = "<STUDY_DIR_REL>"
    for key in sorted(ATTEMPT_KEYS):
        add(
            "attempt_ids",
            str(reference_attempts[key]),
            str(candidate_attempts[key]),
            "<ATTEMPT_ID>",
        )
    for label, logical_path in (
        ("train", "00_grid/{grid}/train_config.yaml"),
        ("validation", "00_grid/{grid}/validation_config.yaml"),
    ):
        reference_config = reference_artifacts[logical_path]
        candidate_config = candidate_paths[logical_path]
        add(
            "config_digests",
            reference_config.raw_sha256,
            _sha256_file(candidate_config),
            f"<CONFIG_DIGEST:{label}>",
        )
    if Path(reference_root).resolve() == candidate_root.resolve():
        # The public comparator rejects this case.  The private content
        # boundary keeps a side-neutral token vocabulary solely so unit tests
        # can prove that a valid fixture compares with itself.
        for category in sorted(TOKEN_SUBSTITUTIONS):
            shared = {
                **values["reference"][category],
                **values["candidate"][category],
            }
            values["reference"][category] = dict(shared)
            values["candidate"][category] = dict(shared)
    result: dict[
        str,
        dict[str, tuple[tuple[str, str], ...]],
    ] = {}
    for side, categories in values.items():
        result[side] = {
            category: tuple(
                sorted(
                    mapping.items(),
                    key=lambda item: (-len(item[0]), item[0]),
                )
            )
            for category, mapping in categories.items()
        }
    return result


def _normalize_string(
    value: str,
    *,
    tokens: TokenMap,
    approved: Sequence[str],
    side: str,
) -> str:
    # Token lists contain only reviewed exact concrete values. The category
    # allowlist is validated by the layout contract before this function runs.
    if not approved:
        return value
    text = value
    substitutions = [
        item
        for category in approved
        for item in tokens[side][category]
    ]
    for source, replacement in sorted(
        substitutions,
        key=lambda item: (-len(item[0]), item[0]),
    ):
        text = text.replace(source, replacement)
    return text


def _semantic_expansions(
    artifacts: Sequence[ReferenceArtifact],
) -> dict[str, list[str]]:
    result = {"scan_runs": [], "final_runs": []}
    for artifact in artifacts:
        logical = artifact.logical_path
        for key, prefix in (
            ("scan_runs", "00_grid/{grid}/jobs/"),
            ("final_runs", "05_final_grid/{final_grid}/jobs/"),
        ):
            if logical.startswith(prefix) and logical.endswith(".json"):
                result[key].append(
                    logical[len(prefix) : -len(".json")]
                )
    for paths in result.values():
        paths.sort()
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("semantic layout expansion is empty/duplicated")
    return result


def _logical_path(
    relative: Path,
    attempts: Mapping[str, str],
) -> str:
    parts = list(relative.parts)
    stage = parts[0]
    low = {
        "00_grid": "grid",
        "03_collect": "collection",
        "04_select": "selection",
        "05_final_grid": "final_grid",
        "08_final_collect": "final_collect",
        "09_final_report": "report",
    }
    fanout = {
        "01_train": "train",
        "02_validation": "validation",
        "06_final_train": "final_train",
        "07_final_eval": "final_eval",
    }
    if stage in low:
        key = low[stage]
        if len(parts) < 2 or parts[1] != attempts[key]:
            raise ValueError(f"candidate protected path attempt mismatch: {relative}")
        parts[1] = "{" + key + "}"
    elif stage in fanout:
        key = fanout[stage]
        if len(parts) >= 4 and parts[1] == "stage_plans":
            if parts[2] != attempts[key]:
                raise ValueError(
                    f"candidate plan attempt mismatch: {relative}"
                )
            parts[2] = "{" + key + "}"
        elif len(parts) == 4 and parts[2] == attempts[key]:
            parts[2] = "{" + key + "}"
        else:
            raise ValueError(f"candidate fan-out path is invalid: {relative}")
    else:
        raise ValueError(f"candidate protected stage is unknown: {stage}")
    return Path(*parts).as_posix()


def _reject_same_source(
    descriptor: Mapping[str, Any],
    candidate: Path,
    *,
    candidate_attempts: Mapping[str, str],
) -> None:
    identity = descriptor.get("source_results")
    if not isinstance(identity, dict):
        raise ValueError("reference source-results identity is unavailable")
    candidate_stat = candidate.stat()
    if (
        str(identity.get("canonical_root") or "") == str(candidate)
        or (
            identity.get("device") == int(candidate_stat.st_dev)
            and identity.get("inode") == int(candidate_stat.st_ino)
        )
    ):
        raise ValueError("candidate and reference source resolve to the same root")
    reference_attempts = descriptor.get("attempts")
    if isinstance(reference_attempts, Mapping) and set(
        str(value) for value in reference_attempts.values()
    ) & set(candidate_attempts.values()):
        raise ValueError("candidate and reference reuse the same lineage id")


def _require_bound_layout(
    descriptor: Mapping[str, Any],
    layout: LayoutMap,
) -> None:
    contract = descriptor.get("comparison_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "layout_map_path",
        "layout_map_schema_version",
        "comparator_schema_version",
        "layout_map_sha256",
    }:
        raise ValueError("reference comparison contract is unavailable")
    expected_source = (
        REPO_ROOT / COMPARISON_LAYOUT_LOGICAL_PATH
    ).resolve(strict=True)
    expected = {
        "schema_version": REFERENCE_COMPARISON_SCHEMA_VERSION,
        "layout_map_path": COMPARISON_LAYOUT_LOGICAL_PATH,
        "layout_map_schema_version": layout.schema_version,
        "comparator_schema_version": layout.comparator_schema_version,
        "layout_map_sha256": layout.sha256,
    }
    if dict(contract) != expected or layout.source_path != expected_source:
        raise ValueError(
            "layout map does not match the reference comparison contract"
        )


def _normalize_attempts(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != ATTEMPT_KEYS:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            "candidate attempt map mismatch; "
            f"missing={sorted(ATTEMPT_KEYS - actual)}, "
            f"unknown={sorted(actual - ATTEMPT_KEYS)}"
        )
    result: dict[str, str] = {}
    for key in sorted(ATTEMPT_KEYS):
        item = value[key]
        if (
            not isinstance(item, str)
            or not item
            or "/" in item
            or item in {".", ".."}
        ):
            raise ValueError(f"invalid candidate attempt id for {key}")
        result[key] = item
    return result


def _json_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _float_equal(
    left: float,
    right: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)


def _csv_float_equal(
    left: str,
    right: str,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if (
        FINITE_DECIMAL_STRING.fullmatch(left) is None
        or FINITE_DECIMAL_STRING.fullmatch(right) is None
    ):
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except ValueError:
        return False
    return _float_equal(
        left_float,
        right_float,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _bounded(value: object) -> str:
    if value == "":
        return ""
    text = repr(value)
    if len(text) <= MAX_VALUE_CHARS:
        return text
    return text[: MAX_VALUE_CHARS - 3] + "..."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_attempts(value: str) -> dict[str, str]:
    parsed = _strict_json_loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--candidate-attempts must be a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


def _strict_json_loads(value: str) -> Any:
    return loads_json(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run an acceptance comparison and optionally write its JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-attempts", required=True)
    parser.add_argument("--layout-map", type=Path, default=DEFAULT_LAYOUT_MAP)
    parser.add_argument("--comparison-id")
    args = parser.parse_args(argv)
    try:
        attempts = _parse_attempts(args.candidate_attempts)
        differences = compare_reference(
            args.reference,
            args.candidate_root,
            candidate_attempts=attempts,
            layout_map_path=args.layout_map,
        )
        if args.comparison_id is not None:
            provenance = comparison_provenance(
                args.reference,
                args.candidate_root,
                candidate_attempts=attempts,
                layout_map_path=args.layout_map,
                allow_incomplete_candidate=bool(differences),
            )
            write_comparison_report(
                args.candidate_root,
                args.comparison_id,
                differences,
                provenance=provenance,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for difference in differences:
        print(
            f"{difference.kind}: {difference.artifact}"
            f"{difference.location}: {difference.message}"
        )
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
