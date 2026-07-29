"""Strict primitives for versioned experiment contract records.

This module intentionally does not depend on a study, a runner, or ``spenn``.
It owns the small amount of JSON canonicalization shared by V4 contract
records.  Existing toolkit JSON helpers remain unchanged because contract
publication needs stronger create-only and closed-schema guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


CONTRACT_SCHEMA_VERSION = "experiment-contracts/v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when durable contract data is malformed or inconsistent."""


def require_identifier(value: object, name: str) -> str:
    """Return a non-empty durable identifier with a conservative grammar."""

    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{name} is not a valid identifier")
    return value


def require_text(value: object, name: str) -> str:
    """Return one non-empty text field without coercing arbitrary objects."""

    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def require_sha256(value: object, name: str) -> str:
    """Return one lower-case SHA-256 digest."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lower-case SHA-256 digest")
    return value


def require_exact_fields(
    value: Mapping[str, object],
    *,
    fields: frozenset[str],
    label: str,
) -> None:
    """Reject both missing and unknown serialized fields."""

    actual = set(value)
    if actual != fields:
        raise ContractError(
            f"{label} fields mismatch; missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def freeze_json(value: object, *, label: str = "value") -> object:
    """Validate JSON data and recursively make container values immutable."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ContractError(f"{label} mapping keys must be non-empty strings")
            frozen[key] = freeze_json(value[key], label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(
            freeze_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ContractError(f"{label} is not JSON-serializable contract data")


def thaw_json(value: object) -> object:
    """Return a mutable JSON-compatible copy of frozen contract data."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode validated JSON data in one deterministic representation."""

    frozen = freeze_json(value)
    return json.dumps(
        thaw_json(frozen),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return a SHA-256 digest over canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def semantic_id(kind: str, payload: Mapping[str, object]) -> str:
    """Return a deterministic typed record identifier from semantic fields only."""

    # Record kinds are versioned protocol names (for example ``trial/v1``),
    # not filesystem identifiers.  Parent/record IDs still use the stricter
    # identifier grammar above.
    require_text(kind, "record kind")
    return f"{kind.replace('/', '-')}:{canonical_sha256({'kind': kind, 'payload': payload})}"


def require_string_tuple(
    value: object,
    *,
    name: str,
    identifiers: bool = False,
    sorted_unique: bool = False,
    nonempty: bool = True,
) -> tuple[str, ...]:
    """Validate one JSON string tuple without treating a string as a sequence."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a JSON sequence")
    result: list[str] = []
    for index, item in enumerate(value):
        item_name = f"{name}[{index}]"
        result.append(
            require_identifier(item, item_name)
            if identifiers
            else require_text(item, item_name)
        )
    if nonempty and not result:
        raise ContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ContractError(f"{name} must be unique")
    if sorted_unique and result != sorted(result):
        raise ContractError(f"{name} must be sorted")
    return tuple(result)


def require_mapping(value: object, name: str, *, nonempty: bool = True) -> Mapping[str, object]:
    """Validate and deep-freeze one JSON object."""

    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    frozen = freeze_json(value, label=name)
    assert isinstance(frozen, Mapping)
    if nonempty and not frozen:
        raise ContractError(f"{name} must not be empty")
    return frozen


def source_keys(value: object, name: str = "source_keys") -> tuple[str, ...]:
    """Validate canonical references into a bundle source table."""

    return require_string_tuple(
        value,
        name=name,
        identifiers=True,
        sorted_unique=True,
        nonempty=True,
    )
