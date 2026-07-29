"""Versioned scientific identity records for one closed experiment bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ._codec import (
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    require_exact_fields,
    require_identifier,
    require_mapping,
    require_text,
    semantic_id,
    source_keys,
    thaw_json,
)


def _record_payload(
    *,
    kind: str,
    semantic: Mapping[str, object],
    identifier: str,
) -> None:
    expected = semantic_id(kind, semantic)
    if identifier and identifier != expected:
        raise ContractError(f"{kind} id does not match semantic fields")


def _source_mapping(value: object, name: str) -> Mapping[str, object]:
    return require_mapping(value, name)


@dataclass(frozen=True)
class TrialV1:
    """One blinded configuration projection within a bundle scope."""

    bundle_scope_id: str
    trial_key: str
    blinded_choices: Mapping[str, object]
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "trial/v1"

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "trial.bundle_scope_id")
        key = require_identifier(self.trial_key, "trial.trial_key")
        choices = _source_mapping(self.blinded_choices, "trial.blinded_choices")
        keys = source_keys(self.source_keys, "trial.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "trial_key": key,
            "blinded_choices": thaw_json(choices),
        }
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "trial_key", key)
        object.__setattr__(self, "blinded_choices", choices)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "trial_key": self.trial_key,
            "blinded_choices": thaw_json(self.blinded_choices),
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrialV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            trial_key=value["trial_key"],
            blinded_choices=value["blinded_choices"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return (
            "kind", "schema_version", "id", "bundle_scope_id", "trial_key",
            "blinded_choices", "source_keys",
        )


@dataclass(frozen=True)
class SeedAssignmentV1:
    """One named, literal seed-value assignment with no seed-policy semantics."""

    bundle_scope_id: str
    assignment_kind: str
    values: Mapping[str, object]
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "seed-assignment/v1"
    _KINDS: ClassVar[frozenset[str]] = frozenset({"scan", "confirm"})

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "seed.bundle_scope_id")
        assignment_kind = require_identifier(self.assignment_kind, "seed.assignment_kind")
        if assignment_kind not in self._KINDS:
            raise ContractError("seed.assignment_kind is unsupported")
        values = _source_mapping(self.values, "seed.values")
        keys = source_keys(self.source_keys, "seed.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "assignment_kind": assignment_kind,
            "values": thaw_json(values),
        }
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "assignment_kind", assignment_kind)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "assignment_kind": self.assignment_kind,
            "values": thaw_json(self.values),
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SeedAssignmentV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            assignment_kind=value["assignment_kind"],
            values=value["values"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return (
            "kind", "schema_version", "id", "bundle_scope_id", "assignment_kind",
            "values", "source_keys",
        )


@dataclass(frozen=True)
class RunV1:
    """One trial and seed assignment under a closed scan or confirm lane."""

    bundle_scope_id: str
    trial_id: str
    seed_assignment_id: str
    lane: str
    run_key: str
    source_champion_key: str | None
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "run/v1"
    _LANES: ClassVar[frozenset[str]] = frozenset({"scan", "confirm"})

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "run.bundle_scope_id")
        trial_id = require_identifier(self.trial_id, "run.trial_id")
        seed_id = require_identifier(self.seed_assignment_id, "run.seed_assignment_id")
        lane = require_identifier(self.lane, "run.lane")
        if lane not in self._LANES:
            raise ContractError("run.lane is unsupported")
        run_key = require_identifier(self.run_key, "run.run_key")
        champion = (
            None
            if self.source_champion_key is None
            else require_identifier(self.source_champion_key, "run.source_champion_key")
        )
        if (lane == "scan") != (champion is None):
            raise ContractError("scan runs omit source_champion_key; confirm runs require it")
        keys = source_keys(self.source_keys, "run.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "trial_id": trial_id,
            "seed_assignment_id": seed_id,
            "lane": lane,
            "run_key": run_key,
        }
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "seed_assignment_id", seed_id)
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "run_key", run_key)
        object.__setattr__(self, "source_champion_key", champion)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "trial_id": self.trial_id,
            "seed_assignment_id": self.seed_assignment_id,
            "lane": self.lane,
            "run_key": self.run_key,
            "source_champion_key": self.source_champion_key,
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            trial_id=value["trial_id"],
            seed_assignment_id=value["seed_assignment_id"],
            lane=value["lane"],
            run_key=value["run_key"],
            source_champion_key=value["source_champion_key"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return (
            "kind", "schema_version", "id", "bundle_scope_id", "trial_id",
            "seed_assignment_id", "lane", "run_key", "source_champion_key",
            "source_keys",
        )


@dataclass(frozen=True)
class ProducerV1:
    """One logical training producer for one supported run."""

    bundle_scope_id: str
    run_id: str
    role: str
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "producer/v1"
    _ROLES: ClassVar[frozenset[str]] = frozenset({"screen_train", "confirm_train"})

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "producer.bundle_scope_id")
        run_id = require_identifier(self.run_id, "producer.run_id")
        role = require_identifier(self.role, "producer.role")
        if role not in self._ROLES:
            raise ContractError("producer.role is unsupported")
        keys = source_keys(self.source_keys, "producer.source_keys")
        semantic = {"bundle_scope_id": scope, "run_id": run_id, "role": role}
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "run_id": self.run_id,
            "role": self.role,
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProducerV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            run_id=value["run_id"],
            role=value["role"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return ("kind", "schema_version", "id", "bundle_scope_id", "run_id", "role", "source_keys")


@dataclass(frozen=True)
class ProducerAttemptV1:
    """One authorized singleton semantic production history for one producer."""

    bundle_scope_id: str
    producer_id: str
    source_task_id: str
    source_execution_task_id: str
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "producer-attempt/v1"

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "attempt.bundle_scope_id")
        producer_id = require_identifier(self.producer_id, "attempt.producer_id")
        task_id = require_identifier(self.source_task_id, "attempt.source_task_id")
        execution_id = require_identifier(
            self.source_execution_task_id,
            "attempt.source_execution_task_id",
        )
        if task_id != execution_id:
            raise ContractError("attempt task and execution task identities differ")
        keys = source_keys(self.source_keys, "attempt.source_keys")
        semantic = {"bundle_scope_id": scope, "producer_id": producer_id}
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "producer_id", producer_id)
        object.__setattr__(self, "source_task_id", task_id)
        object.__setattr__(self, "source_execution_task_id", execution_id)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "producer_id": self.producer_id,
            "source_task_id": self.source_task_id,
            "source_execution_task_id": self.source_execution_task_id,
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProducerAttemptV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            producer_id=value["producer_id"],
            source_task_id=value["source_task_id"],
            source_execution_task_id=value["source_execution_task_id"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return (
            "kind", "schema_version", "id", "bundle_scope_id", "producer_id",
            "source_task_id", "source_execution_task_id", "source_keys",
        )


@dataclass(frozen=True)
class MetricKeyV1:
    """Literal observed metric-key schema; intentionally no scientific policy."""

    bundle_scope_id: str
    stage_result_id: str
    namespace: str
    key: str
    scalar_representation: str
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "metric-key/v1"
    _SCALARS: ClassVar[frozenset[str]] = frozenset({"null", "bool", "int", "float", "str"})

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "metric.bundle_scope_id")
        stage_id = require_identifier(self.stage_result_id, "metric.stage_result_id")
        namespace = require_text(self.namespace, "metric.namespace")
        key = require_text(self.key, "metric.key")
        scalar = require_identifier(self.scalar_representation, "metric.scalar_representation")
        if scalar not in self._SCALARS:
            raise ContractError("metric.scalar_representation is unsupported")
        keys = source_keys(self.source_keys, "metric.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "stage_result_id": stage_id,
            "namespace": namespace,
            "key": key,
            "scalar_representation": scalar,
        }
        _record_payload(kind=self.kind, semantic=semantic, identifier=self.id)
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "stage_result_id", stage_id)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "scalar_representation", scalar)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", semantic_id(self.kind, semantic))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "stage_result_id": self.stage_result_id,
            "namespace": self.namespace,
            "key": self.key,
            "scalar_representation": self.scalar_representation,
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MetricKeyV1":
        require_exact_fields(value, fields=frozenset(cls._fields()), label=cls.kind)
        _require_record_header(value, cls.kind)
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            stage_result_id=value["stage_result_id"],
            namespace=value["namespace"],
            key=value["key"],
            scalar_representation=value["scalar_representation"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )

    @classmethod
    def _fields(cls) -> tuple[str, ...]:
        return (
            "kind", "schema_version", "id", "bundle_scope_id", "stage_result_id",
            "namespace", "key", "scalar_representation", "source_keys",
        )


def _require_record_header(value: Mapping[str, object], kind: str) -> None:
    if value.get("kind") != kind:
        raise ContractError(f"record kind is not {kind}")
    if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("record schema_version is unsupported")
    require_identifier(value.get("id"), "record.id")
