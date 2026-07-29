"""Terminal stage-result records for closed V4 contract bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ._codec import (
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    require_exact_fields,
    require_identifier,
    require_sha256,
    semantic_id,
    source_keys,
)


@dataclass(frozen=True)
class StageResultV1:
    """One successfully completed logical stage, never a dispatch submission."""

    bundle_scope_id: str
    logical_role: str
    physical_stage: str
    execution_profile_id: str
    terminal_population_sha256: str
    source_keys: tuple[str, ...]
    id: str = ""

    kind: ClassVar[str] = "stage-result/v1"

    def __post_init__(self) -> None:
        scope = require_identifier(self.bundle_scope_id, "stage.bundle_scope_id")
        role = require_identifier(self.logical_role, "stage.logical_role")
        physical = require_identifier(self.physical_stage, "stage.physical_stage")
        profile = require_identifier(self.execution_profile_id, "stage.execution_profile_id")
        population = require_sha256(
            self.terminal_population_sha256,
            "stage.terminal_population_sha256",
        )
        keys = source_keys(self.source_keys, "stage.source_keys")
        semantic = {
            "bundle_scope_id": scope,
            "logical_role": role,
            "physical_stage": physical,
            "execution_profile_id": profile,
            "status": "completed",
        }
        expected = semantic_id(self.kind, semantic)
        if self.id and self.id != expected:
            raise ContractError("stage-result id does not match semantic fields")
        object.__setattr__(self, "bundle_scope_id", scope)
        object.__setattr__(self, "logical_role", role)
        object.__setattr__(self, "physical_stage", physical)
        object.__setattr__(self, "execution_profile_id", profile)
        object.__setattr__(self, "terminal_population_sha256", population)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "id", expected)

    @property
    def status(self) -> str:
        """Return V4-1A's only exercised terminal state."""

        return "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "bundle_scope_id": self.bundle_scope_id,
            "logical_role": self.logical_role,
            "physical_stage": self.physical_stage,
            "execution_profile_id": self.execution_profile_id,
            "status": self.status,
            "terminal_population_sha256": self.terminal_population_sha256,
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StageResultV1":
        fields = frozenset(
            {
                "kind", "schema_version", "id", "bundle_scope_id", "logical_role",
                "physical_stage", "execution_profile_id", "status",
                "terminal_population_sha256", "source_keys",
            }
        )
        require_exact_fields(value, fields=fields, label=cls.kind)
        if value.get("kind") != cls.kind:
            raise ContractError(f"record kind is not {cls.kind}")
        if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ContractError("record schema_version is unsupported")
        if value.get("status") != "completed":
            raise ContractError("V4-1A stage result must be completed")
        return cls(
            bundle_scope_id=value["bundle_scope_id"],
            logical_role=value["logical_role"],
            physical_stage=value["physical_stage"],
            execution_profile_id=value["execution_profile_id"],
            terminal_population_sha256=value["terminal_population_sha256"],
            source_keys=tuple(value["source_keys"]),
            id=value["id"],
        )
