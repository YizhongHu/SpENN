"""Command override rewriting helpers."""

from __future__ import annotations

from typing import Sequence

_SPECIAL_STRING_CHARS = set(" ,=[]{}:'\"\\")


def format_override_value(value: object) -> str:
    """Serialize ``value`` into Hydra override-string syntax.

    Handles the value types that already appear in this study's smoke/grid
    overrides today (int, float, list) plus bool/None/dict/str, which the
    previous plain ``f"{key}={value}"`` formatting could not render
    correctly (``str(True)`` -> ``"True"``, which Hydra does not parse as a
    boolean; ``str({...})`` -> a Python repr, not valid Hydra mapping
    syntax). Round-tripped against ``hydra.core.override_parser`` in tests.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if any(char in _SPECIAL_STRING_CHARS for char in value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{format_override_value(val)}" for key, val in value.items()) + "}"
    raise TypeError(f"Unsupported override value type: {type(value)!r}")


def rewrite_cli_overrides(command: Sequence[str], overrides: dict[str, object]) -> list[str]:
    """Return ``command`` with exact ``key=`` entries replaced by ``overrides``."""

    keys = set(overrides)
    rewritten = []
    for part in command:
        text = str(part)
        key, separator, _value = text.partition("=")
        if separator and key in keys:
            continue
        rewritten.append(text)
    rewritten.extend(f"{key}={format_override_value(value)}" for key, value in overrides.items())
    return rewritten
