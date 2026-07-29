"""Requested execution-profile records for V4 contract bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ._codec import (
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    require_exact_fields,
    require_identifier,
    require_mapping,
    semantic_id,
    source_keys,
    thaw_json,
)


@dataclass(frozen=True)
class ExecutionProfileV1:
    """One requested controller or fan-out resource projection.

    This is requested profile identity, never a host, scheduler-job, process,
    or actual-allocation record.
    """

    bundle_scope_id: str
    profile_kind: str
    requested: Mapping[str, object]
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "execution-profile/v1"
    _KINDS: ClassVar[frozenset[str]] = frozenset({"controller", "fanout"})

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "profile.bundle_scope_id")
        profile_kind = require_identifier(self.profile_kind, "profile.profile_kind")
        if profile_kind not in self._KINDS:
            raise ContractError("profile.profile_kind is unsupported")
        requested = require_mapping(self.requested, "profile.requested")
        keys = source_keys(self.source_keys, "profile.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "profile_kind": profile_kind,
            "requested": thaw_json(requested),
        }
        expected = semantic_id(self.kind, semantic)
        if self.id and self.id != expected:
            raise ContractError("execution-profile id does not match semantic fields")
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "profile_kind", profile_kind)
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "profile_kind": self.profile_kind,
            "requested": thaw_json(self.requested),
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionProfileV1":
        fields = frozenset(
            {
                "kind", "schema_version", "id", "bundle_scope_id", "profile_kind",
                "requested", "source_keys",
            }
        )
        require_exact_fields(value, fields=fields, label=cls.kind)
        if value.get("kind") != cls.kind:
            raise ContractError(f"record kind is not {cls.kind}")
        if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ContractError("record schema_version is unsupported")
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            profile_kind=value["profile_kind"],
            requested=value["requested"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )
