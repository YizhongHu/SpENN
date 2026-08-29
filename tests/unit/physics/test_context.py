from __future__ import annotations

from dataclasses import dataclass

import pytest

from tpen.physics.context import (
    ContextKey,
    ReadOnlyContext,
    requirements,
    validate_provider_graph,
)


def test_context_resolves_by_key_identity_after_renaming() -> None:
    key = ContextKey("original", int)
    context = ReadOnlyContext({key: 7})

    object.__setattr__(key, "name", "renamed")

    assert context[key] == 7


def test_context_is_read_only() -> None:
    key = ContextKey("answer", int)
    context = ReadOnlyContext({key: 7})

    assert key in context
    with pytest.raises((AttributeError, TypeError)):
        context[key] = 8  # type: ignore[index]


def test_provider_graph_reports_missing_dependency_and_operator() -> None:
    missing = ContextKey("missing_distances", tuple)
    output = ContextKey("potential", float)

    @dataclass
    class Provider:
        key = output
        dependencies = requirements(missing)

        def build(self, context: ReadOnlyContext) -> float:
            return 0.0

    with pytest.raises(ValueError, match="operator op-7.*missing_distances"):
        validate_provider_graph([Provider()], (), operator="op-7")


def test_provider_graph_reports_cycles() -> None:
    first = ContextKey("first", int)
    second = ContextKey("second", int)

    @dataclass
    class First:
        key = first
        dependencies = requirements(second)

        def build(self, context: ReadOnlyContext) -> int:
            return 1

    @dataclass
    class Second:
        key = second
        dependencies = requirements(first)

        def build(self, context: ReadOnlyContext) -> int:
            return 2

    with pytest.raises(ValueError, match="operator op-8.*cyclic"):
        validate_provider_graph([First(), Second()], (), operator="op-8")
