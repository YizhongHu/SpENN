"""Strict versioned layout-map contracts for V4-0 parity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from strict_data import load_json

LAYOUT_SCHEMA_VERSION = "pair-stability-v4/layout-map/v1"
COMPARATOR_SCHEMA_VERSION = "pair-stability-v4/comparator/v1"
DEFAULT_LAYOUT_MAP = (
    Path(__file__).resolve().parent
    / "reference"
    / "layout_maps"
    / "v1.json"
)
FORMATS = {"json", "jsonl", "csv", "yaml", "text", "markdown"}
EXPANSION_TOKENS = {
    "single": None,
    "scan_runs": "<scan_run_id>",
    "final_runs": "<final_run_id>",
}
TOKEN_SUBSTITUTIONS = {
    "study_identity",
    "results_root",
    "study_path",
    "attempt_ids",
    "config_digests",
}


@dataclass(frozen=True)
class LayoutEntry:
    """One typed logical-path mapping and its literal comparison policy."""

    logical_role: str
    reference_logical_path: str
    candidate_logical_path: str
    expansion: str
    format: str
    approved_token_substitutions: tuple[str, ...]
    volatile_json_pointers: tuple[str, ...]
    volatile_csv_columns: tuple[str, ...]
    float_tolerant_json_pointers: tuple[str, ...]
    float_string_tolerant_json_pointers: tuple[str, ...]
    float_tolerant_csv_columns: tuple[str, ...]
    json_record_arrays: tuple["JsonRecordArrayPolicy", ...]
    presence_only: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayoutEntry":
        required = {
            "logical_role",
            "reference_logical_path",
            "candidate_logical_path",
            "expansion",
            "format",
            "approved_token_substitutions",
            "volatile_json_pointers",
            "volatile_csv_columns",
            "float_tolerant_json_pointers",
            "float_string_tolerant_json_pointers",
            "float_tolerant_csv_columns",
            "json_record_arrays",
            "presence_only",
        }
        if set(value) != required:
            raise ValueError("layout entry fields mismatch")
        entry = cls(
            logical_role=_nonempty(value["logical_role"], "logical_role"),
            reference_logical_path=_logical_template(
                value["reference_logical_path"]
            ),
            candidate_logical_path=_logical_template(
                value["candidate_logical_path"]
            ),
            expansion=_nonempty(value["expansion"], "expansion"),
            format=_nonempty(value["format"], "format"),
            approved_token_substitutions=_string_tuple(
                value["approved_token_substitutions"],
                "approved_token_substitutions",
            ),
            volatile_json_pointers=_pointer_tuple(
                value["volatile_json_pointers"],
                "volatile_json_pointers",
            ),
            volatile_csv_columns=_column_tuple(
                value["volatile_csv_columns"],
                "volatile_csv_columns",
            ),
            float_tolerant_json_pointers=_pointer_tuple(
                value["float_tolerant_json_pointers"],
                "float_tolerant_json_pointers",
            ),
            float_string_tolerant_json_pointers=_pointer_tuple(
                value["float_string_tolerant_json_pointers"],
                "float_string_tolerant_json_pointers",
            ),
            float_tolerant_csv_columns=_column_tuple(
                value["float_tolerant_csv_columns"],
                "float_tolerant_csv_columns",
            ),
            json_record_arrays=_json_record_array_tuple(
                value["json_record_arrays"],
            ),
            presence_only=value["presence_only"],
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.expansion not in EXPANSION_TOKENS:
            raise ValueError(f"unsupported layout expansion {self.expansion!r}")
        if self.format not in FORMATS:
            raise ValueError(f"unsupported layout format {self.format!r}")
        if not isinstance(self.presence_only, bool):
            raise ValueError("layout presence_only must be boolean")
        if set(self.approved_token_substitutions) - TOKEN_SUBSTITUTIONS:
            raise ValueError("layout entry has unknown token substitution")
        token = EXPANSION_TOKENS[self.expansion]
        paths = (
            self.reference_logical_path,
            self.candidate_logical_path,
        )
        if token is None and any("<" in path or ">" in path for path in paths):
            raise ValueError("single layout entry contains expansion token")
        if token is not None and any(path.count(token) != 1 for path in paths):
            raise ValueError(
                f"layout expansion {self.expansion!r} must occur once"
            )
        if token is not None and any(
            "<" in path.replace(token, "")
            or ">" in path.replace(token, "")
            for path in paths
        ):
            raise ValueError("layout entry contains an unknown template token")
        other_tokens = {
            item
            for item in EXPANSION_TOKENS.values()
            if item is not None and item != token
        }
        if any(other in path for other in other_tokens for path in paths):
            raise ValueError("layout entry contains wrong expansion token")
        if set(self.volatile_json_pointers) & set(
            self.float_tolerant_json_pointers
        ):
            raise ValueError("JSON pointer cannot be volatile and tolerant")
        if set(self.volatile_json_pointers) & set(
            self.float_string_tolerant_json_pointers
        ):
            raise ValueError(
                "JSON pointer cannot be volatile and float-string tolerant"
            )
        if set(self.float_tolerant_json_pointers) & set(
            self.float_string_tolerant_json_pointers
        ):
            raise ValueError(
                "JSON pointer cannot have numeric and string float tolerance"
            )
        if set(self.volatile_csv_columns) & set(
            self.float_tolerant_csv_columns
        ):
            raise ValueError("CSV column cannot be volatile and tolerant")
        json_policy = (
            self.volatile_json_pointers
            or self.float_tolerant_json_pointers
            or self.float_string_tolerant_json_pointers
            or self.json_record_arrays
        )
        csv_policy = (
            self.volatile_csv_columns
            or self.float_tolerant_csv_columns
        )
        if json_policy and self.format not in {"json", "jsonl", "yaml"}:
            raise ValueError("JSON pointer policy used for non-structured entry")
        if self.json_record_arrays and self.format != "json":
            raise ValueError("JSON record-array policy used for non-JSON entry")
        if csv_policy and self.format != "csv":
            raise ValueError("CSV policy used for non-CSV entry")
        if self.presence_only and (json_policy or csv_policy):
            raise ValueError("presence-only entry cannot have value policies")
        record_pointers = {
            policy.array_pointer for policy in self.json_record_arrays
        }
        scalar_pointers = {
            *self.volatile_json_pointers,
            *self.float_tolerant_json_pointers,
            *self.float_string_tolerant_json_pointers,
        }
        if any(
            scalar == record
            or scalar.startswith(f"{record}/")
            or record.startswith(f"{scalar}/")
            for scalar in scalar_pointers
            for record in record_pointers
        ):
            raise ValueError(
                "scalar JSON policy overlaps a record-array policy"
            )


@dataclass(frozen=True)
class JsonRecordArrayPolicy:
    """Literal field policy for every object in one JSON array."""

    array_pointer: str
    volatile_fields: tuple[str, ...]
    float_string_tolerant_fields: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "JsonRecordArrayPolicy":
        if not isinstance(value, Mapping) or set(value) != {
            "array_pointer",
            "volatile_fields",
            "float_string_tolerant_fields",
        }:
            raise ValueError("JSON record-array policy fields mismatch")
        pointer = _pointer_tuple(
            [value["array_pointer"]],
            "json_record_arrays.array_pointer",
        )[0]
        policy = cls(
            array_pointer=pointer,
            volatile_fields=_field_tuple(
                value["volatile_fields"],
                "json_record_arrays.volatile_fields",
            ),
            float_string_tolerant_fields=_field_tuple(
                value["float_string_tolerant_fields"],
                "json_record_arrays.float_string_tolerant_fields",
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not (
            self.volatile_fields
            or self.float_string_tolerant_fields
        ):
            raise ValueError("JSON record-array policy has no governed fields")
        if set(self.volatile_fields) & set(
            self.float_string_tolerant_fields
        ):
            raise ValueError(
                "JSON record-array field cannot be volatile and tolerant"
            )


@dataclass(frozen=True)
class LayoutMap:
    """Validated top-level layout-map contract."""

    schema_version: str
    comparator_schema_version: str
    rel_tol: float
    abs_tol: float
    entries: tuple[LayoutEntry, ...]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class ArtifactPolicy:
    """Concrete one-to-one policy after manifest-driven expansion."""

    logical_role: str
    reference_logical_path: str
    candidate_logical_path: str
    format: str
    approved_token_substitutions: tuple[str, ...]
    volatile_json_pointers: tuple[str, ...]
    volatile_csv_columns: tuple[str, ...]
    float_tolerant_json_pointers: tuple[str, ...]
    float_string_tolerant_json_pointers: tuple[str, ...]
    float_tolerant_csv_columns: tuple[str, ...]
    json_record_arrays: tuple[JsonRecordArrayPolicy, ...]
    presence_only: bool


def load_layout_map(path: Path = DEFAULT_LAYOUT_MAP) -> LayoutMap:
    """Read and strictly validate one versioned layout map."""

    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"layout map is not a regular file: {requested}")
    raw = load_json(requested)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "comparator_schema_version",
        "tolerances",
        "entries",
    }:
        raise ValueError("layout map fields mismatch")
    if raw["schema_version"] != LAYOUT_SCHEMA_VERSION:
        raise ValueError("layout map schema version mismatch")
    if raw["comparator_schema_version"] != COMPARATOR_SCHEMA_VERSION:
        raise ValueError("layout comparator schema version mismatch")
    tolerances = raw["tolerances"]
    if not isinstance(tolerances, dict) or set(tolerances) != {
        "rel_tol",
        "abs_tol",
    }:
        raise ValueError("layout tolerance fields mismatch")
    rel_tol = _positive_float(tolerances["rel_tol"], "rel_tol")
    abs_tol = _positive_float(tolerances["abs_tol"], "abs_tol")
    if rel_tol != 1e-9 or abs_tol != 1e-12:
        raise ValueError("layout tolerances differ from approved values")
    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("layout entries must be a nonempty list")
    entries = tuple(LayoutEntry.from_dict(value) for value in raw_entries)
    reference_templates = [
        (entry.expansion, entry.reference_logical_path) for entry in entries
    ]
    candidate_templates = [
        (entry.expansion, entry.candidate_logical_path) for entry in entries
    ]
    roles = [entry.logical_role for entry in entries]
    if len(set(reference_templates)) != len(reference_templates):
        raise ValueError("layout reference templates are duplicated")
    if len(set(candidate_templates)) != len(candidate_templates):
        raise ValueError("layout candidate templates are duplicated")
    if len(set(roles)) != len(roles):
        raise ValueError("layout logical roles are duplicated")
    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return LayoutMap(
        schema_version=raw["schema_version"],
        comparator_schema_version=raw["comparator_schema_version"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        entries=entries,
        source_path=requested.resolve(),
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def materialize_layout(
    layout: LayoutMap,
    *,
    expansions: Mapping[str, Sequence[str]],
    reference_paths: set[str],
    candidate_paths: set[str],
) -> tuple[ArtifactPolicy, ...]:
    """Expand typed templates and require an exact protected-path bijection."""

    unknown = set(expansions) - (set(EXPANSION_TOKENS) - {"single"})
    if unknown:
        raise ValueError(f"unknown layout expansion values: {sorted(unknown)}")
    policies: list[ArtifactPolicy] = []
    for entry in layout.entries:
        token = EXPANSION_TOKENS[entry.expansion]
        values: Sequence[str | None]
        if token is None:
            values = (None,)
        else:
            raw_values = expansions.get(entry.expansion)
            if raw_values is None:
                raise ValueError(
                    f"missing layout expansion {entry.expansion!r}"
                )
            values = tuple(_safe_component(value) for value in raw_values)
        for value in values:
            reference_path = entry.reference_logical_path
            candidate_path = entry.candidate_logical_path
            if token is not None and value is not None:
                reference_path = reference_path.replace(token, value)
                candidate_path = candidate_path.replace(token, value)
            policies.append(
                ArtifactPolicy(
                    logical_role=(
                        entry.logical_role
                        if value is None
                        else f"{entry.logical_role}:{value}"
                    ),
                    reference_logical_path=reference_path,
                    candidate_logical_path=candidate_path,
                    format=entry.format,
                    approved_token_substitutions=(
                        entry.approved_token_substitutions
                    ),
                    volatile_json_pointers=entry.volatile_json_pointers,
                    volatile_csv_columns=entry.volatile_csv_columns,
                    float_tolerant_json_pointers=(
                        entry.float_tolerant_json_pointers
                    ),
                    float_string_tolerant_json_pointers=(
                        entry.float_string_tolerant_json_pointers
                    ),
                    float_tolerant_csv_columns=(
                        entry.float_tolerant_csv_columns
                    ),
                    json_record_arrays=entry.json_record_arrays,
                    presence_only=entry.presence_only,
                )
            )
    reference_mapped = [
        policy.reference_logical_path for policy in policies
    ]
    candidate_mapped = [
        policy.candidate_logical_path for policy in policies
    ]
    if len(set(reference_mapped)) != len(reference_mapped):
        raise ValueError("materialized reference layout is not one-to-one")
    if len(set(candidate_mapped)) != len(candidate_mapped):
        raise ValueError("materialized candidate layout is not one-to-one")
    if set(reference_mapped) != reference_paths:
        raise ValueError(
            "layout/reference inventory mismatch; "
            f"missing={sorted(reference_paths - set(reference_mapped))}, "
            f"extra={sorted(set(reference_mapped) - reference_paths)}"
        )
    if set(candidate_mapped) != candidate_paths:
        raise ValueError(
            "layout/candidate inventory mismatch; "
            f"missing={sorted(candidate_paths - set(candidate_mapped))}, "
            f"extra={sorted(set(candidate_mapped) - candidate_paths)}"
        )
    return tuple(
        sorted(policies, key=lambda policy: policy.reference_logical_path)
    )


def _logical_template(value: object) -> str:
    text = _nonempty(value, "logical path")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError(f"unsafe layout logical path {text!r}")
    if any(character in text for character in "*?["):
        raise ValueError("layout paths cannot contain globs")
    if "//" in text:
        raise ValueError("layout path contains an empty component")
    return text


def _pointer_tuple(value: object, label: str) -> tuple[str, ...]:
    pointers = _string_tuple(value, label)
    for pointer in pointers:
        if not pointer.startswith("/") or any(
            character in pointer for character in "*?["
        ):
            raise ValueError(f"{label} contains nonliteral JSON pointer")
        if re.search(r"~(?![01])", pointer):
            raise ValueError(f"{label} contains invalid JSON pointer escape")
    return pointers


def _column_tuple(value: object, label: str) -> tuple[str, ...]:
    columns = _string_tuple(value, label)
    if any(any(character in column for character in "*?[") for column in columns):
        raise ValueError(f"{label} contains nonliteral CSV column")
    return columns


def _field_tuple(value: object, label: str) -> tuple[str, ...]:
    fields = _string_tuple(value, label)
    if any(any(character in field for character in "*?[") for field in fields):
        raise ValueError(f"{label} contains nonliteral field")
    return fields


def _json_record_array_tuple(
    value: object,
) -> tuple[JsonRecordArrayPolicy, ...]:
    if not isinstance(value, list):
        raise ValueError("json_record_arrays must be a list")
    policies = tuple(JsonRecordArrayPolicy.from_dict(item) for item in value)
    pointers = tuple(policy.array_pointer for policy in policies)
    if len(set(pointers)) != len(pointers) or pointers != tuple(
        sorted(pointers)
    ):
        raise ValueError(
            "json_record_arrays must have sorted unique array pointers"
        )
    if any(
        left.startswith(f"{right}/") or right.startswith(f"{left}/")
        for index, left in enumerate(pointers)
        for right in pointers[index + 1 :]
    ):
        raise ValueError("JSON record-array policies overlap")
    return policies


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    result = tuple(value)
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _safe_component(value: object) -> str:
    text = _nonempty(value, "expansion component")
    if Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"unsafe layout expansion component {text!r}")
    return text
