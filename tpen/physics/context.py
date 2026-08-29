"""Typed artifacts shared by local-energy operators.

The context is intentionally keyed by ContextKey objects, rather than by
their diagnostic names. A name is useful in an error message, but it is not
a dispatch key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ContextKey[T]:
    """The typed identity of one context artifact."""

    name: str
    value_type: type[T]


type ContextRequirements = frozenset[ContextKey[object]]


class ReadOnlyContext:
    """A context which permits typed reads but no writes.

    Entries are indexed by the identity of their key object. This is
    deliberate: ContextKey.name is diagnostic metadata and carries no
    dispatch meaning.
    """

    def __init__(self, values: Mapping[ContextKey[object], object]) -> None:
        entries: dict[int, tuple[ContextKey[object], object]] = {}
        for key, value in values.items():
            if not isinstance(key, ContextKey):
                raise TypeError("context keys must be ContextKey instances")
            if not isinstance(value, key.value_type):
                raise TypeError(
                    f"context artifact {key.name!r} must be {key.value_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            entries[id(key)] = (key, value)
        self._entries = entries

    def __getitem__[T](self, key: ContextKey[T]) -> T:
        if not isinstance(key, ContextKey):
            raise TypeError("context lookup requires a ContextKey")
        entry = self._entries.get(id(key))
        if entry is None or entry[0] is not key:
            raise KeyError(key.name)
        value = entry[1]
        if not isinstance(value, key.value_type):
            raise TypeError(
                f"context artifact {key.name!r} must be {key.value_type.__name__}, "
                f"got {type(value).__name__}"
            )
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, ContextKey):
            return False
        entry = self._entries.get(id(key))
        return entry is not None and entry[0] is key


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
    by_key: dict[int, object] = {id(provider.key): provider for provider in providers}
    available = {id(key): key for key in required}
    for provider in providers:
        available[id(provider.key)] = provider.key

    for provider in providers:
        for dependency in provider.dependencies:
            if id(dependency) not in available:
                raise ValueError(
                    f"operator {operator} requires missing context key {dependency.name!r}"
                )

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(key: ContextKey[object]) -> None:
        marker = id(key)
        if marker in visited or marker not in by_key:
            return
        if marker in visiting:
            raise ValueError(f"operator {operator} has cyclic context dependency at {key.name!r}")
        visiting.add(marker)
        for dependency in by_key[marker].dependencies:
            visit(dependency)
        visiting.remove(marker)
        visited.add(marker)

    for provider in providers:
        visit(provider.key)


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
