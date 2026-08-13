"""Command override rewriting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

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


AxisOverrideSpec = str | dict[str, Any]
_AXIS_OVERRIDE_STAGES = {"train", "validation", "final_train", "final_eval"}


def normalize_axis_override_specs(
    configured: Mapping[str, Any] | None,
    axes: Sequence[str],
    *,
    context: str,
) -> dict[str, AxisOverrideSpec]:
    """Return normalized axis override specs for the requested axes."""

    if not isinstance(configured, Mapping):
        raise ValueError(f"{context} axis_overrides must be a mapping")
    missing = [axis for axis in axes if axis not in configured]
    if missing:
        raise ValueError(f"{context} axis_overrides is missing axes: {', '.join(missing)}")

    specs: dict[str, AxisOverrideSpec] = {}
    for axis in axes:
        spec = configured[axis]
        if isinstance(spec, str):
            specs[axis] = spec
            continue
        if isinstance(spec, Mapping):
            if not spec:
                raise ValueError(f"{context} axis_overrides.{axis} must not be empty")
            specs[axis] = {str(path): value for path, value in spec.items()}
            continue
        raise ValueError(f"{context} axis_overrides.{axis} must be a string or mapping")
    return specs


def _mapped_axis_value(value: Any, value_map: Mapping[Any, Any], *, axis: str) -> Any:
    if value in value_map:
        return value_map[value]
    text = str(value)
    if text in value_map:
        return value_map[text]
    raise ValueError(f"axis_overrides.{axis} has no mapped override for value {value!r}")


def _stage_override_spec(spec: AxisOverrideSpec, stage: str | None) -> Any:
    if not isinstance(spec, Mapping) or stage is None:
        return spec
    if not (set(str(key) for key in spec) & _AXIS_OVERRIDE_STAGES):
        return spec
    return spec.get(stage)


def axis_value_overrides(
    point: Mapping[str, Any],
    *,
    axes: Sequence[str],
    override_specs: Mapping[str, AxisOverrideSpec],
    stage: str | None = None,
) -> list[str]:
    """Return Hydra overrides for scalar axes.

    A simple spec maps one axis to one override path. A mapping spec maps one
    semantic axis value to one or more concrete override paths, which keeps
    coupled experiment mechanisms out of the base SpENN configs.
    """

    overrides: list[str] = []
    for axis in axes:
        if axis not in point:
            raise ValueError(f"job choices are missing configured axis {axis!r}")
        spec = _stage_override_spec(override_specs[axis], stage)
        if spec is None:
            continue
        value = point[axis]
        if isinstance(spec, str):
            overrides.append(f"{spec}={format_override_value(value)}")
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"axis_overrides.{axis} must be a string or mapping")
        for path, value_spec in spec.items():
            override_value = (
                _mapped_axis_value(value, value_spec, axis=axis)
                if isinstance(value_spec, Mapping)
                else value_spec
            )
            overrides.append(f"{path}={format_override_value(override_value)}")
    return overrides
