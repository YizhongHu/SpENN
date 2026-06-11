"""Virtual-support path metadata for equivariant real-space mixing.

Path generation is intentionally separated from neural modules. The metadata
objects here are deterministic constructors/readers; training code should load
saved cache files instead of silently regenerating path orderings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spenn.data.indices import ordered_tuples


OutputEmbedding = Literal["canonical", "full"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEFAULT_PATH_FILES = {
    "canonical": CACHE_DIR / "paths_canonical.json",
    "full": CACHE_DIR / "paths_full.json",
}


@dataclass(frozen=True)
class VirtualPath:
    """Describe one bilinear virtual-support mixing path.

    Parameters
    ----------
    s : int
        Virtual support order.
    m : int
        Output tuple order.
    m1, m2 : int
        Left and right input tuple orders.
    local_id : int
        Stable path index inside a fixed ``(s, m, m1, m2)`` block.
    global_id : int
        Stable path index across the whole path file.
    tau, tau1, tau2 : tuple of int
        Injective maps into the virtual support, represented as zero-based
        images.
    """

    s: int
    m: int
    m1: int
    m2: int
    local_id: int
    global_id: int
    tau: tuple[int, ...]
    tau1: tuple[int, ...]
    tau2: tuple[int, ...]

    @property
    def input_support(self) -> set[int]:
        """Return virtual labels covered by the two input injections."""

        return set(self.tau1) | set(self.tau2)

    def as_tuple(self) -> tuple[int, int, int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return the mathematical tuple ``(s, m, m1, m2, tau, tau1, tau2)``."""

        return (self.s, self.m, self.m1, self.m2, self.tau, self.tau1, self.tau2)


PathFamily = dict[int, dict[int, dict[int, dict[int, list[VirtualPath]]]]]


class PathMetadata:
    """Thin reader/constructor for saved virtual-support path metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON metadata path.
    """

    schema_version = 1
    path_order_version = "lexicographic-v1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_json(self.path)
        self.max_order = int(self.data["max_order"])
        self.max_virtual_order = int(self.data["max_virtual_order"])
        self.output_embedding = self.data["output_embedding"]
        self.paths = self._parse_paths(self.data["paths"])

    @classmethod
    def load(cls, path: str | Path) -> "PathMetadata":
        """Load metadata from `path`."""

        return cls(path)

    @classmethod
    def generate(
        cls,
        *,
        max_order: int,
        max_virtual_order: int,
        output_embedding: OutputEmbedding,
    ) -> "PathMetadata":
        """Generate deterministic metadata without writing it to disk."""

        paths = generate_virtual_paths(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            output_embedding=output_embedding,
        )
        metadata = cls.__new__(cls)
        metadata.path = None
        metadata.max_order = int(max_order)
        metadata.max_virtual_order = int(max_virtual_order)
        metadata.output_embedding = output_embedding
        metadata.paths = paths
        metadata.data = metadata.to_json_data()
        return metadata

    def save(self, path: str | Path) -> None:
        """Write this metadata to `path` as stable compact JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_data(), separators=(",", ":"), sort_keys=True) + "\n")
        self.path = output

    def get(self, s: int, m: int, m1: int, m2: int) -> list[VirtualPath]:
        """Return paths for one fixed ``(s, m, m1, m2)`` block."""

        return self.paths.get(s, {}).get(m, {}).get(m1, {}).get(m2, [])

    def paths_for_output_order(self, m: int) -> list[VirtualPath]:
        """Return all paths contributing to output order `m`."""

        return [path for path in self.all_paths() if path.m == m]

    def paths_for_virtual_order(self, s: int) -> list[VirtualPath]:
        """Return all paths with virtual support order `s`."""

        return [path for path in self.all_paths() if path.s == s]

    def all_paths(self) -> list[VirtualPath]:
        """Return every path in stable global-id order."""

        paths = [
            path
            for by_m in self.paths.values()
            for by_m1 in by_m.values()
            for by_m2 in by_m1.values()
            for block in by_m2.values()
            for path in block
        ]
        return sorted(paths, key=lambda path: path.global_id)

    def to_json_data(self) -> dict[str, object]:
        """Return the JSON-serializable metadata representation."""

        return {
            "schema_version": self.schema_version,
            "index_base": 0,
            "max_order": self.max_order,
            "max_virtual_order": self.max_virtual_order,
            "output_embedding": self.output_embedding,
            "path_order_version": self.path_order_version,
            "path_storage_format": "nested-injections-v1",
            "paths": _serialize_paths(self.paths),
        }

    @staticmethod
    def _load_json(path: Path) -> dict[str, object]:
        data = json.loads(path.read_text())
        if int(data.get("schema_version", -1)) != PathMetadata.schema_version:
            raise ValueError(f"Unsupported path metadata schema in {path}")
        if int(data.get("index_base", -1)) != 0:
            raise ValueError(f"Path metadata must use zero-based indices: {path}")
        if data.get("path_order_version") != PathMetadata.path_order_version:
            raise ValueError(f"Unsupported path ordering in {path}")
        return data

    @staticmethod
    def _parse_paths(data: object) -> PathFamily:
        if not isinstance(data, list):
            raise TypeError("Path metadata paths must use compact nested-list storage")
        return _parse_compact_paths(data)


def _parse_compact_paths(data: list[object]) -> PathFamily:
    paths: PathFamily = {}
    global_id = 0
    for s, by_m in enumerate(data):
        if s == 0 or not by_m:
            continue
        if not isinstance(by_m, list):
            raise TypeError(f"paths[{s}] must be a list")
        paths[s] = {}
        for m, by_m1 in enumerate(by_m):
            if m == 0 or not by_m1:
                continue
            if not isinstance(by_m1, list):
                raise TypeError(f"paths[{s}][{m}] must be a list")
            paths[s][m] = {}
            for m1, by_m2 in enumerate(by_m1):
                if m1 == 0 or not by_m2:
                    continue
                if not isinstance(by_m2, list):
                    raise TypeError(f"paths[{s}][{m}][{m1}] must be a list")
                paths[s][m][m1] = {}
                for m2, block in enumerate(by_m2):
                    if m2 == 0 or block is None:
                        continue
                    if not isinstance(block, list):
                        raise TypeError(f"paths[{s}][{m}][{m1}][{m2}] must be a list")
                    parsed = []
                    for local_id, item in enumerate(block):
                        if not (
                            isinstance(item, list)
                            and len(item) == 3
                            and all(isinstance(component, list) for component in item)
                        ):
                            raise TypeError("compact path entries must be [tau, tau1, tau2]")
                        path = VirtualPath(
                            s=s,
                            m=m,
                            m1=m1,
                            m2=m2,
                            local_id=local_id,
                            global_id=global_id,
                            tau=tuple(int(value) for value in item[0]),
                            tau1=tuple(int(value) for value in item[1]),
                            tau2=tuple(int(value) for value in item[2]),
                        )
                        validate_virtual_path(path, max_virtual_order=s)
                        parsed.append(path)
                        global_id += 1
                    paths[s][m][m1][m2] = parsed
    return paths


def load_default_path_metadata(output_embedding: OutputEmbedding) -> PathMetadata:
    """Load saved project path metadata for an output embedding.

    Parameters
    ----------
    output_embedding : {"canonical", "full"}
        Saved path family to load.

    Returns
    -------
    PathMetadata
        Metadata loaded from ``spenn/cache``.
    """

    if output_embedding not in DEFAULT_PATH_FILES:
        raise ValueError(f"Unsupported output_embedding {output_embedding!r}")
    return PathMetadata.load(DEFAULT_PATH_FILES[output_embedding])


def generate_virtual_paths(
    *,
    max_order: int,
    max_virtual_order: int,
    output_embedding: OutputEmbedding,
) -> PathFamily:
    """Generate deterministic virtual-support paths.

    Parameters
    ----------
    max_order : int
        Maximum input/output body order.
    max_virtual_order : int
        Maximum hidden virtual support order.
    output_embedding : {"canonical", "full"}
        Whether the output map ``tau`` is fixed to ``(0, ..., m - 1)`` or all
        injective output maps are used.
    """

    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    if max_virtual_order <= 0:
        raise ValueError(f"max_virtual_order must be positive, got {max_virtual_order}")
    if output_embedding not in {"canonical", "full"}:
        raise ValueError(f"Unsupported output_embedding {output_embedding!r}")

    paths: PathFamily = {}
    global_id = 0
    for s in range(1, max_virtual_order + 1):
        paths[s] = {}
        for m in range(1, min(max_order, s) + 1):
            paths[s][m] = {}
            # Canonical output embeddings fix tau = (0, ..., m - 1).
            # This is a gauge choice for the output injection, not independent
            # canonicalization of all injections. Relative input injections
            # tau1/tau2 remain part of the path data and carry the interaction
            # degrees of freedom.
            output_maps = [tuple(range(m))] if output_embedding == "canonical" else ordered_tuples(s, m)
            for m1 in range(1, min(max_order, s) + 1):
                paths[s][m][m1] = {}
                # Injections from [m1] into the virtual support
                left_maps = ordered_tuples(s, m1)
                for m2 in range(1, min(max_order, s) + 1):
                    # Injections from [m2] into the virtual support
                    right_maps = ordered_tuples(s, m2)
                    block: list[VirtualPath] = []
                    local_id = 0
                    # Iterate over all injections
                    for tau in output_maps:
                        for tau1 in left_maps:
                            for tau2 in right_maps:
                                path = VirtualPath(
                                    s=s,
                                    m=m,
                                    m1=m1,
                                    m2=m2,
                                    local_id=local_id,
                                    global_id=global_id,
                                    tau=tuple(tau),
                                    tau1=tuple(tau1),
                                    tau2=tuple(tau2),
                                )
                                if path.input_support != set(range(s)):
                                    continue
                                validate_virtual_path(path, max_virtual_order=max_virtual_order)
                                block.append(path)
                                local_id += 1
                                global_id += 1
                    paths[s][m][m1][m2] = sorted(
                        block,
                        key=lambda path: (path.s, path.m, path.m1, path.m2, path.tau, path.tau1, path.tau2),
                    )
    return paths


def iter_path_blocks(paths: PathFamily) -> Iterator[tuple[tuple[int, int, int, int], list[VirtualPath]]]:
    """Yield ``((s, m, m1, m2), paths)`` blocks in deterministic order."""

    for s in sorted(paths):
        for m in sorted(paths[s]):
            for m1 in sorted(paths[s][m]):
                for m2 in sorted(paths[s][m][m1]):
                    yield (s, m, m1, m2), paths[s][m][m1][m2]


def validate_virtual_path(path: VirtualPath, *, max_order: int | None = None, max_virtual_order: int | None = None) -> None:
    """Validate one virtual-support path."""

    limit = max_virtual_order if max_virtual_order is not None else max_order
    if limit is not None and path.s > limit:
        raise ValueError(f"Virtual support order {path.s} exceeds max_virtual_order {limit}")
    if path.m <= 0 or path.m1 <= 0 or path.m2 <= 0:
        raise ValueError("Path orders must be positive")
    if path.m > path.s:
        raise ValueError("m must be <= s")
    for name, order, injection in (
        ("tau", path.m, path.tau),
        ("tau1", path.m1, path.tau1),
        ("tau2", path.m2, path.tau2),
    ):
        if len(injection) != order:
            raise ValueError(f"{name} length must match its order")
        if len(set(injection)) != len(injection):
            raise ValueError(f"{name} must be injective")
        if any(label < 0 or label >= path.s for label in injection):
            raise ValueError(f"{name} labels must land in the virtual support")
    if path.input_support != set(range(path.s)):
        raise ValueError("left and right injections must cover the virtual support")


def _serialize_paths(paths: PathFamily) -> list[object]:
    serialized: list[object] = []
    for (s, m, m1, m2), block in iter_path_blocks(paths):
        by_m = _ensure_list_index(serialized, s)
        by_m1 = _ensure_list_index(by_m, m)
        by_m2 = _ensure_list_index(by_m1, m1)
        _ensure_list_index(by_m2, m2)
        by_m2[m2] = [[list(path.tau), list(path.tau1), list(path.tau2)] for path in block]
    return serialized


def _ensure_list_index(items: list[object], index: int) -> list[object]:
    while len(items) <= index:
        items.append([])
    value = items[index]
    if not isinstance(value, list):
        raise TypeError(f"Expected list at compact path index {index}")
    return value


__all__ = [
    "PathFamily",
    "PathMetadata",
    "VirtualPath",
    "generate_virtual_paths",
    "iter_path_blocks",
    "load_default_path_metadata",
    "validate_virtual_path",
]
