"""Typed artifacts shared by local-energy operators.

The context is intentionally keyed by ContextKey objects, rather than by
their diagnostic names. A name is useful in an error message, but it is not
a dispatch key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True, eq=False)
class ContextKey[T]:
    """The typed identity of one context artifact.

    Keys use object identity deliberately: two distinct keys that happen to
    have equal fields are distinct artifacts.
    """

    name: str
    value_type: type[T]


type ContextRequirements = frozenset[ContextKey[object]]


class ReadOnlyContext:
    """A context which permits typed reads but no writes.

    Entries are indexed by their key objects. This is deliberate:
    ContextKey.name is diagnostic metadata and carries no dispatch meaning.
    """

    def __init__(self, values: Mapping[ContextKey[object], object]) -> None:
        entries: dict[ContextKey[object], object] = {}
        for key, value in values.items():
            if not isinstance(key, ContextKey):
                raise TypeError("context keys must be ContextKey instances")
            if not isinstance(value, key.value_type):
                raise TypeError(
                    # Interpolating the type object is total for runtime type-like values,
                    # including PEP 604 unions, unlike reading a name attribute.
                    f"context artifact {key.name!r} must be {key.value_type}, "
                    f"got {type(value).__name__}"
                )
            entries[key] = value
        # Keep the mutable construction buffer private and expose only its
        # immutable proxy. No consumer-reachable reference can mutate entries.
        self._entries = MappingProxyType(entries)

    def __getitem__[T](self, key: ContextKey[T]) -> T:
        if not isinstance(key, ContextKey):
            raise TypeError("context lookup requires a ContextKey")
        value = self._entries.get(key)
        if value is None and key not in self._entries:
            raise KeyError(key.name)
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, ContextKey):
            return False
        return key in self._entries


class ContextProvider[T](Protocol):
    """A pure producer of one declared context artifact."""

    key: ContextKey[T]
    dependencies: ContextRequirements

    def build(self, context: ReadOnlyContext) -> T:
        """Build the artifact from the declared read-only context."""
        ...


def requirements(*keys: ContextKey[object]) -> ContextRequirements:
    """Return an immutable requirement set for a provider declaration."""

    return frozenset(keys)


def validate_provider_graph(
    providers: Sequence[object],
    required: Iterable[ContextKey[object]],
    *,
    operator: object,
) -> None:
    """Validate provider availability and dependency cycles before execution.

    This is only declaration validation; provider execution and operator
    planning remain owned by the later planner slice.
    """

    # object is intentional: typeguard cannot runtime-check a Protocol with
    # data attributes. Providers are validated by their declarations below;
    # no structural probe selects a computation.
    by_key: dict[ContextKey[object], object] = {provider.key: provider for provider in providers}
    available = set(required)
    for provider in providers:
        available.add(provider.key)

    for provider in providers:
        for dependency in provider.dependencies:
            if dependency not in available:
                raise ValueError(
                    f"operator {operator} requires missing context key {dependency.name!r}"
                )

    visited: set[ContextKey[object]] = set()
    for provider in providers:
        root = provider.key
        if root in visited:
            continue
        visiting: set[ContextKey[object]] = set()
        stack: list[tuple[ContextKey[object], bool]] = [(root, False)]
        while stack:
            key, leaving = stack.pop()
            if leaving:
                visiting.remove(key)
                visited.add(key)
                continue
            if key in visited:
                continue
            if key in visiting:
                raise ValueError(f"operator {operator} has cyclic context dependency at {key.name!r}")
            if key not in by_key:
                continue
            visiting.add(key)
            stack.append((key, True))
            stack.extend((dependency, False) for dependency in by_key[key].dependencies)


Context = ReadOnlyContext

__all__ = [
    "Context",
    "ContextKey",
    "ContextProvider",
    "ContextRequirements",
    "ReadOnlyContext",
    "requirements",
    "validate_provider_graph",
]
