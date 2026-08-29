from __future__ import annotations

from dataclasses import dataclass

import pytest

from tpen.physics.context import (
    ContextKey,
    ReadOnlyContext,
    requirements,
    validate_provider_graph,
)


def test_context_distinguishes_distinct_same_name_keys() -> None:
    first = ContextKey("distances", int)
    second = ContextKey("distances", int)
    context = ReadOnlyContext({first: 1, second: 2})

    assert context[first] == 1
    assert context[second] == 2


def test_context_does_not_resolve_value_equal_key_from_elsewhere() -> None:
    original = ContextKey("distances", int)
    elsewhere = ContextKey("distances", int)
    context = ReadOnlyContext({original: 1})

    assert original != elsewhere
    assert elsewhere not in context
    with pytest.raises(KeyError):
        context[elsewhere]


def test_context_is_read_only() -> None:
    key = ContextKey("answer", int)
    context = ReadOnlyContext({key: 7})

    assert key in context
    with pytest.raises((AttributeError, TypeError)):
        context[key] = 8  # type: ignore[index]
    with pytest.raises(TypeError):
        context._entries[key] = 8  # type: ignore[index]


def test_context_rejects_value_with_wrong_type() -> None:
    key = ContextKey("answer", int)

    with pytest.raises(TypeError, match="answer.*int.*str"):
        ReadOnlyContext({key: "not an answer"})  # type: ignore[dict-item]


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


def test_provider_graph_reports_deep_cycle_without_recursion() -> None:
    keys = [ContextKey(f"key-{index}", int) for index in range(1_201)]

    @dataclass
    class Provider:
        key: ContextKey[int]
        dependencies: frozenset[ContextKey[object]]

        def build(self, context: ReadOnlyContext) -> int:
            return 0

    providers = [
        Provider(key, requirements(keys[index + 1]))
        for index, key in enumerate(keys[:-1])
    ]
    providers.append(Provider(keys[-1], requirements(keys[0])))

    with pytest.raises(ValueError, match="operator op-deep.*cyclic"):
        validate_provider_graph(providers, (), operator="op-deep")
