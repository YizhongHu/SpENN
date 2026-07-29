"""Fail-closed structured-data readers for V4-0 acceptance evidence.

V4-0 treats structured files as evidence, not permissive configuration.  The
standard JSON and YAML convenience loaders silently accept duplicate mapping
keys and can construct non-finite floats.  That ambiguity is unsafe when a
later comparison deliberately ignores volatile fields, so this module owns the
single strict parsing policy used by acceptance paths.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


class StrictDataError(ValueError):
    """Raised when acceptance evidence is not unambiguous finite data."""


def loads_json(value: str | bytes, *, source: str = "JSON") -> Any:
    """Parse one JSON value without duplicate keys or non-finite floats."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise StrictDataError(
                    f"duplicate JSON object key in {source}: {key!r}"
                )
            result[key] = item
        return result

    def parse_constant(constant: str) -> None:
        raise StrictDataError(
            f"non-finite JSON constant is forbidden in {source}: {constant}"
        )

    def parse_float(text: str) -> float:
        parsed = float(text)
        if not math.isfinite(parsed):
            raise StrictDataError(
                f"non-finite JSON number is forbidden in {source}: {text}"
            )
        return parsed

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=parse_constant,
            parse_float=parse_float,
        )
    except json.JSONDecodeError as exc:
        raise StrictDataError(f"invalid JSON in {source}: {exc}") from exc
    _reject_nonfinite(parsed, source=source)
    return parsed


def load_json(path: Path) -> Any:
    """Read one strict JSON file."""

    path = Path(path)
    try:
        return loads_json(path.read_text(encoding="utf-8"), source=str(path))
    except OSError as exc:
        raise StrictDataError(f"cannot read JSON {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Yield strict nonblank JSONL rows with useful source locations."""

    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield loads_json(
                        line,
                        source=f"{path}:{line_number}",
                    )
    except OSError as exc:
        raise StrictDataError(f"cannot read JSONL {path}: {exc}") from exc


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe loader with duplicate-key rejection for every mapping node."""


def _construct_mapping(
    loader: _StrictSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise StrictDataError("YAML mapping key is not hashable") from exc
        if duplicate:
            raise StrictDataError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def loads_yaml(value: str | bytes, *, source: str = "YAML") -> Any:
    """Parse one YAML value without duplicate keys or non-finite floats."""

    try:
        parsed = yaml.load(value, Loader=_StrictSafeLoader)
    except (yaml.YAMLError, StrictDataError) as exc:
        if isinstance(exc, StrictDataError):
            raise StrictDataError(f"invalid YAML in {source}: {exc}") from exc
        raise StrictDataError(f"invalid YAML in {source}: {exc}") from exc
    _reject_nonfinite(parsed, source=source)
    return parsed


def load_yaml(path: Path) -> Any:
    """Read one strict YAML file."""

    path = Path(path)
    try:
        return loads_yaml(path.read_text(encoding="utf-8"), source=str(path))
    except OSError as exc:
        raise StrictDataError(f"cannot read YAML {path}: {exc}") from exc


def validate_structured_path(path: Path) -> None:
    """Validate one supported evidence file, preserving its native semantics."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        load_json(path)
    elif suffix == ".jsonl":
        tuple(iter_jsonl(path))
    elif suffix in {".yaml", ".yml"}:
        load_yaml(path)


def validate_structured_paths(paths: Iterable[Path]) -> None:
    """Validate every supported file in an explicit protected read set."""

    for path in paths:
        validate_structured_path(Path(path))


def validate_structured_tree(root: Path) -> None:
    """Validate all supported regular files below one guarded evidence root."""

    root = Path(root)
    for current, directory_names, filenames in os.walk(root, followlinks=False):
        directory_names.sort()
        filenames.sort()
        current_path = Path(current)
        for name in filenames:
            path = current_path / name
            if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise StrictDataError(
                    f"structured evidence is not a regular file: {path}"
                )
            validate_structured_path(path)


def _reject_nonfinite(value: Any, *, source: str) -> None:
    """Reject constructed non-finite floats, including YAML ``.inf`` values."""

    seen: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, float):
            if not math.isfinite(item):
                raise StrictDataError(
                    f"non-finite number is forbidden in {source}: {item!r}"
                )
            return
        if isinstance(item, (str, bytes, int, bool, type(None))):
            return
        identifier = id(item)
        if identifier in seen:
            raise StrictDataError(f"recursive structured value is forbidden in {source}")
        if isinstance(item, dict):
            seen.add(identifier)
            for key, child in item.items():
                visit(key)
                visit(child)
            seen.remove(identifier)
            return
        if isinstance(item, (list, tuple, set)):
            seen.add(identifier)
            for child in item:
                visit(child)
            seen.remove(identifier)
            return
        raise StrictDataError(
            f"unsupported structured value in {source}: {type(item).__name__}"
        )

    visit(value)
