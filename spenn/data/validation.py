"""Runtime validation contracts, kept separate from equivariance contracts.

Validation is a typed, per-object concern: a value validates its own
runtime/schema invariants and may expose explicit JSON-safe validity metrics.
This is deliberately distinct from `spenn.data.equivariant_state` (permutation +
comparison) -- there is no generic ``validate_tree`` / recursive probing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

JsonScalar = int | float | bool | str | None


class RuntimeValidatable(Protocol):
    """Typed object that validates its own runtime/schema invariants."""

    def validate(self) -> None:
        ...


class RuntimeValidityMetrics(Protocol):
    """Typed object that exposes JSON-safe runtime validity metrics."""

    def validity_metrics(self) -> Mapping[str, JsonScalar]:
        ...


__all__ = ["JsonScalar", "RuntimeValidatable", "RuntimeValidityMetrics"]
