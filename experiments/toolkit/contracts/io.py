"""Create-only publication and verified reading for V4 contract bundles."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence, TypeVar

from ._codec import (
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    require_exact_fields,
    require_identifier,
    require_sha256,
    require_text,
)
from .execution import ExecutionProfileV1
from .identities import (
    MetricKeyV1,
    ProducerAttemptV1,
    ProducerV1,
    RunV1,
    SeedAssignmentV1,
    TrialV1,
)
from .stages import StageResultV1


BUNDLE_SCHEMA_VERSION = "experiment-contract-bundle/v1"
MANIFEST_NAME = "manifest.json"
_TABLES: tuple[tuple[str, type[object]], ...] = (
    ("trials.jsonl", TrialV1),
    ("seed_assignments.jsonl", SeedAssignmentV1),
    ("runs.jsonl", RunV1),
    ("producers.jsonl", ProducerV1),
    ("producer_attempts.jsonl", ProducerAttemptV1),
    ("execution_profiles.jsonl", ExecutionProfileV1),
    ("metric_keys.jsonl", MetricKeyV1),
    ("stage_results.jsonl", StageResultV1),
)
_TABLE_NAMES = frozenset(name for name, _ in _TABLES)
_T = TypeVar("_T")


@dataclass(frozen=True)
class SourceDescriptorV1:
    """One allow-listed, root-relative source artifact bound by digest."""

    source_key: str
    root_relative_path: str
    logical_role: str
    artifact_kind: str
    schema: str
    sha256: str

    kind = "source-descriptor/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_key", require_identifier(self.source_key, "source.source_key"))
        object.__setattr__(self, "root_relative_path", _safe_relative_path(self.root_relative_path))
        object.__setattr__(self, "logical_role", require_identifier(self.logical_role, "source.logical_role"))
        object.__setattr__(self, "artifact_kind", require_identifier(self.artifact_kind, "source.artifact_kind"))
        object.__setattr__(self, "schema", require_text(self.schema, "source.schema"))
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "source.sha256"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "source_key": self.source_key,
            "root_relative_path": self.root_relative_path,
            "logical_role": self.logical_role,
            "artifact_kind": self.artifact_kind,
            "schema": self.schema,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceDescriptorV1":
        require_exact_fields(
            value,
            fields=frozenset(
                {
                    "kind", "source_key", "root_relative_path", "logical_role",
                    "artifact_kind", "schema", "sha256",
                }
            ),
            label=cls.kind,
        )
        if value.get("kind") != cls.kind:
            raise ContractError("source descriptor kind is unsupported")
        return cls(
            source_key=value["source_key"],
            root_relative_path=value["root_relative_path"],
            logical_role=value["logical_role"],
            artifact_kind=value["artifact_kind"],
            schema=value["schema"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class ContractBundleV1:
    """Closed graph of V4-1 semantic records for one successful route."""

    study: str
    bundle_scope_id: str
    sources: tuple[SourceDescriptorV1, ...]
    trials: tuple[TrialV1, ...]
    seed_assignments: tuple[SeedAssignmentV1, ...]
    runs: tuple[RunV1, ...]
    producers: tuple[ProducerV1, ...]
    producer_attempts: tuple[ProducerAttemptV1, ...]
    execution_profiles: tuple[ExecutionProfileV1, ...]
    metric_keys: tuple[MetricKeyV1, ...]
    stage_results: tuple[StageResultV1, ...]

    def __post_init__(self) -> None:
        study = require_identifier(self.study, "bundle.study")
        scope = require_identifier(self.bundle_scope_id, "bundle.bundle_scope_id")
        object.__setattr__(self, "study", study)
        object.__setattr__(self, "bundle_scope_id", scope)
        for field in _bundle_fields():
            object.__setattr__(self, field, tuple(getattr(self, field)))
        self.validate()

    def validate(self) -> "ContractBundleV1":
        """Validate closed graph links, cardinalities, ordering, and sources."""

        sources = self.sources
        if not sources:
            raise ContractError("bundle.sources must not be empty")
        if tuple(item.source_key for item in sources) != tuple(
            sorted(item.source_key for item in sources)
        ):
            raise ContractError("bundle sources must be sorted by source_key")
        source_keys = [item.source_key for item in sources]
        if len(source_keys) != len(set(source_keys)):
            raise ContractError("bundle source keys must be unique")
        known_sources = set(source_keys)

        tables = self.tables()
        all_ids: set[str] = set()
        by_table: dict[str, dict[str, object]] = {}
        for table_name, rows in tables.items():
            if not rows:
                raise ContractError(f"{table_name} must not be empty")
            ids = [str(getattr(row, "id")) for row in rows]
            if ids != sorted(ids):
                raise ContractError(f"{table_name} must be sorted by id")
            if len(ids) != len(set(ids)):
                raise ContractError(f"{table_name} contains duplicate ids")
            if all_ids.intersection(ids):
                raise ContractError("bundle record IDs collide across tables")
            all_ids.update(ids)
            for row in rows:
                if getattr(row, "bundle_scope_id") != self.bundle_scope_id:
                    raise ContractError(f"{table_name} row has foreign bundle scope")
                missing_sources = set(getattr(row, "source_keys")) - known_sources
                if missing_sources:
                    raise ContractError(
                        f"{table_name} row references unknown source keys: "
                        f"{sorted(missing_sources)}"
                    )
            by_table[table_name] = {str(getattr(row, "id")): row for row in rows}

        trials = by_table["trials"]
        seeds = by_table["seed_assignments"]
        runs = by_table["runs"]
        producers = by_table["producers"]
        attempts = by_table["producer_attempts"]
        profiles = by_table["execution_profiles"]
        stages = by_table["stage_results"]

        run_keys: set[tuple[str, str]] = set()
        for row in runs.values():
            assert isinstance(row, RunV1)
            if row.trial_id not in trials or row.seed_assignment_id not in seeds:
                raise ContractError("run parent identity is absent from bundle")
            seed = seeds[row.seed_assignment_id]
            assert isinstance(seed, SeedAssignmentV1)
            if seed.assignment_kind != row.lane:
                raise ContractError("run lane does not match seed-assignment kind")
            if row.source_champion_key is not None:
                if row.source_champion_key not in known_sources:
                    raise ContractError("run source champion key is not an allowed source")
                if row.source_champion_key not in row.source_keys:
                    raise ContractError(
                        "run source champion key is not cited by run source_keys"
                    )
            key = (row.lane, row.run_key)
            if key in run_keys:
                raise ContractError("bundle contains duplicate lane/run_key")
            run_keys.add(key)

        producers_by_run: dict[str, list[ProducerV1]] = {}
        for row in producers.values():
            assert isinstance(row, ProducerV1)
            if row.run_id not in runs:
                raise ContractError("producer parent run is absent from bundle")
            producers_by_run.setdefault(row.run_id, []).append(row)
        if set(producers_by_run) != set(runs):
            raise ContractError("every run requires exactly one training producer")
        for run_id, rows in producers_by_run.items():
            if len(rows) != 1:
                raise ContractError("each run requires exactly one training producer")
            run = runs[run_id]
            assert isinstance(run, RunV1)
            expected_role = "screen_train" if run.lane == "scan" else "confirm_train"
            if rows[0].role != expected_role:
                raise ContractError("producer role does not match run lane")

        attempts_by_producer: dict[str, list[ProducerAttemptV1]] = {}
        for row in attempts.values():
            assert isinstance(row, ProducerAttemptV1)
            if row.producer_id not in producers:
                raise ContractError("producer attempt parent is absent from bundle")
            attempts_by_producer.setdefault(row.producer_id, []).append(row)
        if set(attempts_by_producer) != set(producers):
            raise ContractError("every producer requires exactly one semantic attempt")
        if any(len(rows) != 1 for rows in attempts_by_producer.values()):
            raise ContractError("V4-1A permits exactly one attempt per producer")

        roles: set[str] = set()
        for row in stages.values():
            assert isinstance(row, StageResultV1)
            if row.execution_profile_id not in profiles:
                raise ContractError("stage result references absent execution profile")
            if row.logical_role in roles:
                raise ContractError("bundle contains duplicate logical stage result")
            roles.add(row.logical_role)
        for row in self.metric_keys:
            if row.stage_result_id not in stages:
                raise ContractError("metric key references absent stage result")
        return self

    def tables(self) -> dict[str, tuple[object, ...]]:
        """Return rows keyed by canonical on-disk table stem."""

        return {
            "trials": self.trials,
            "seed_assignments": self.seed_assignments,
            "runs": self.runs,
            "producers": self.producers,
            "producer_attempts": self.producer_attempts,
            "execution_profiles": self.execution_profiles,
            "metric_keys": self.metric_keys,
            "stage_results": self.stage_results,
        }


@dataclass(frozen=True)
class BundleManifestV1:
    """Create-last index for one complete set of row files and source evidence."""

    study: str
    bundle_scope_id: str
    sources: tuple[SourceDescriptorV1, ...]
    source_table_sha256: str
    row_files: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "study", require_identifier(self.study, "manifest.study"))
        object.__setattr__(self, "bundle_scope_id", require_identifier(self.bundle_scope_id, "manifest.bundle_scope_id"))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "source_table_sha256", require_sha256(self.source_table_sha256, "manifest.source_table_sha256"))
        object.__setattr__(self, "row_files", tuple(dict(item) for item in self.row_files))
        self.validate()

    def validate(self) -> "BundleManifestV1":
        source_rows = [item.to_dict() for item in self.sources]
        if self.source_table_sha256 != canonical_sha256(source_rows):
            raise ContractError("manifest source table digest mismatch")
        if tuple(item.source_key for item in self.sources) != tuple(sorted(item.source_key for item in self.sources)):
            raise ContractError("manifest sources must be sorted")
        if len({item.source_key for item in self.sources}) != len(self.sources):
            raise ContractError("manifest source keys must be unique")
        if len(self.row_files) != len(_TABLES):
            raise ContractError("manifest row file population mismatch")
        names: list[str] = []
        for row in self.row_files:
            require_exact_fields(
                row,
                fields=frozenset({"name", "kind", "count", "ids", "sha256"}),
                label="manifest row file",
            )
            name = row.get("name")
            kind = row.get("kind")
            count = row.get("count")
            identifiers = row.get("ids")
            require_text(name, "manifest row file name")
            require_text(kind, "manifest row file kind")
            if type(count) is not int or count <= 0:
                raise ContractError("manifest row file count must be positive integer")
            if not isinstance(identifiers, list) or not all(isinstance(item, str) for item in identifiers):
                raise ContractError("manifest row file ids must be a string list")
            if len(identifiers) != count or identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
                raise ContractError("manifest row file ids are not canonical")
            require_sha256(row.get("sha256"), "manifest row file sha256")
            names.append(str(name))
        if tuple(names) != tuple(name for name, _ in _TABLES):
            raise ContractError("manifest row files are not canonical")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "study": self.study,
            "bundle_scope_id": self.bundle_scope_id,
            "sources": [item.to_dict() for item in self.sources],
            "source_table_sha256": self.source_table_sha256,
            "row_files": [dict(item) for item in self.row_files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BundleManifestV1":
        require_exact_fields(
            value,
            fields=frozenset(
                {
                    "schema_version", "study", "bundle_scope_id", "sources",
                    "source_table_sha256", "row_files",
                }
            ),
            label="bundle manifest",
        )
        if value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ContractError("bundle manifest schema_version is unsupported")
        raw_sources = value.get("sources")
        raw_files = value.get("row_files")
        if not isinstance(raw_sources, list) or not all(isinstance(item, Mapping) for item in raw_sources):
            raise ContractError("bundle manifest sources must be objects")
        if not isinstance(raw_files, list) or not all(isinstance(item, Mapping) for item in raw_files):
            raise ContractError("bundle manifest row_files must be objects")
        return cls(
            study=value["study"],
            bundle_scope_id=value["bundle_scope_id"],
            sources=tuple(SourceDescriptorV1.from_dict(item) for item in raw_sources),
            source_table_sha256=value["source_table_sha256"],
            row_files=tuple(dict(item) for item in raw_files),
        )


def publish_bundle(destination: Path, bundle: ContractBundleV1) -> Path:
    """Publish validated bundle rows and manifest with create-only semantics.

    Row files are staged beside destination.  Destination itself is created
    exclusively, rows move into it, and manifest moves last.  An interrupted
    publication therefore remains an invalid, non-reusable partial directory;
    it is never repaired or overwritten by a later call.
    """

    bundle.validate()
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"contract bundle destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ContractError(f"contract bundle parent is unsafe: {parent}")
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=parent))
    try:
        row_files: list[dict[str, object]] = []
        for filename, _record_type in _TABLES:
            table_name = filename.removesuffix(".jsonl")
            rows = bundle.tables()[table_name]
            payload = b"".join(
                canonical_json_bytes(getattr(row, "to_dict")()) + b"\n"
                for row in rows
            )
            _write_new_bytes(staged / filename, payload)
            row_files.append(
                {
                    "name": filename,
                    "kind": str(getattr(rows[0], "kind")),
                    "count": len(rows),
                    "ids": [str(getattr(row, "id")) for row in rows],
                    "sha256": _sha256_file(staged / filename),
                }
            )
        manifest = BundleManifestV1(
            study=bundle.study,
            bundle_scope_id=bundle.bundle_scope_id,
            sources=bundle.sources,
            source_table_sha256=canonical_sha256([item.to_dict() for item in bundle.sources]),
            row_files=tuple(row_files),
        )
        _write_new_bytes(staged / MANIFEST_NAME, canonical_json_bytes(manifest.to_dict()) + b"\n")
        os.mkdir(destination, 0o755)
        for filename, _record_type in _TABLES:
            os.replace(staged / filename, destination / filename)
        os.replace(staged / MANIFEST_NAME, destination / MANIFEST_NAME)
        _fsync_directory(destination)
        _fsync_directory(parent)
    finally:
        _remove_empty_directory(staged)
    return destination


def read_bundle(destination: Path, *, source_root: Path | None = None) -> ContractBundleV1:
    """Read one complete bundle and recompute every declared integrity edge."""

    destination = Path(destination)
    if destination.is_symlink() or not destination.is_dir():
        raise ContractError("contract bundle directory is missing or unsafe")
    expected_entries = {MANIFEST_NAME, *_TABLE_NAMES}
    actual_entries = {entry.name for entry in destination.iterdir()}
    if actual_entries != expected_entries:
        raise ContractError(
            "contract bundle file population mismatch; "
            f"missing={sorted(expected_entries - actual_entries)}, "
            f"extra={sorted(actual_entries - expected_entries)}"
        )
    manifest_path = destination / MANIFEST_NAME
    manifest = BundleManifestV1.from_dict(_load_json_object(manifest_path))
    parsed: dict[str, tuple[object, ...]] = {}
    manifest_files = {str(item["name"]): item for item in manifest.row_files}
    for filename, record_type in _TABLES:
        path = destination / filename
        metadata = manifest_files.get(filename)
        if metadata is None:
            raise ContractError(f"manifest lacks row file {filename}")
        if _sha256_file(path) != metadata["sha256"]:
            raise ContractError(f"contract row file digest mismatch: {filename}")
        rows = _read_jsonl_records(path, record_type)
        ids = [str(getattr(row, "id")) for row in rows]
        if len(rows) != metadata["count"] or ids != metadata["ids"]:
            raise ContractError(f"contract row file population mismatch: {filename}")
        if not rows or str(getattr(rows[0], "kind")) != metadata["kind"]:
            raise ContractError(f"contract row file kind mismatch: {filename}")
        parsed[filename.removesuffix(".jsonl")] = rows
    bundle = ContractBundleV1(
        study=manifest.study,
        bundle_scope_id=manifest.bundle_scope_id,
        sources=manifest.sources,
        trials=_typed_rows(parsed, "trials", TrialV1),
        seed_assignments=_typed_rows(parsed, "seed_assignments", SeedAssignmentV1),
        runs=_typed_rows(parsed, "runs", RunV1),
        producers=_typed_rows(parsed, "producers", ProducerV1),
        producer_attempts=_typed_rows(parsed, "producer_attempts", ProducerAttemptV1),
        execution_profiles=_typed_rows(parsed, "execution_profiles", ExecutionProfileV1),
        metric_keys=_typed_rows(parsed, "metric_keys", MetricKeyV1),
        stage_results=_typed_rows(parsed, "stage_results", StageResultV1),
    )
    if source_root is not None:
        _verify_sources(bundle.sources, Path(source_root))
    return bundle


def bundle_manifest_sha256(destination: Path) -> str:
    """Return exact on-disk manifest digest for an external verifier receipt."""

    return _sha256_file(Path(destination) / MANIFEST_NAME)


def _verify_sources(sources: Sequence[SourceDescriptorV1], root: Path) -> None:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("contract source root is missing or unsafe")
    canonical_root = root.resolve(strict=True)
    for descriptor in sources:
        candidate = canonical_root / descriptor.root_relative_path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"declared source is unavailable: {descriptor.root_relative_path}") from exc
        if canonical_root not in resolved.parents or candidate.is_symlink() or not candidate.is_file():
            raise ContractError(f"declared source is unsafe: {descriptor.root_relative_path}")
        if _sha256_file(candidate) != descriptor.sha256:
            raise ContractError(f"declared source digest mismatch: {descriptor.root_relative_path}")


def _read_jsonl_records(path: Path, record_type: type[_T]) -> tuple[_T, ...]:
    _require_regular_file(path)
    rows: list[_T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ContractError(f"invalid JSONL row framing in {path.name}:{line_number}")
            try:
                payload = _json_loads_strict(line)
            except (json.JSONDecodeError, ContractError) as exc:
                raise ContractError(f"invalid JSONL row in {path.name}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ContractError(f"JSONL row is not an object in {path.name}:{line_number}")
            parser = getattr(record_type, "from_dict")
            rows.append(parser(payload))
    if not rows:
        raise ContractError(f"contract row file is empty: {path.name}")
    return tuple(rows)


def _typed_rows(rows: Mapping[str, tuple[object, ...]], name: str, kind: type[_T]) -> tuple[_T, ...]:
    values = rows[name]
    if not all(isinstance(value, kind) for value in values):
        raise ContractError(f"contract row type mismatch: {name}")
    return tuple(values)  # type: ignore[return-value]


def _safe_relative_path(value: object) -> str:
    text = require_text(value, "source.root_relative_path")
    if "\\" in text:
        raise ContractError("source.root_relative_path must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError("source.root_relative_path is unsafe")
    return path.as_posix()


def _bundle_fields() -> tuple[str, ...]:
    return (
        "sources", "trials", "seed_assignments", "runs", "producers",
        "producer_attempts", "execution_profiles", "metric_keys", "stage_results",
    )


def _load_json_object(path: Path) -> Mapping[str, object]:
    _require_regular_file(path)
    try:
        value = _json_loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"invalid contract JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"contract JSON is not an object: {path}")
    return value


def _json_loads_strict(text: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise ContractError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def invalid_constant(value: str) -> object:
        raise ContractError(f"invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ContractError(f"contract file is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError(f"contract file is not regular: {path}")


def _sha256_file(path: Path) -> str:
    _require_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
