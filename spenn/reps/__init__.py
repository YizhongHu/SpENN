"""Representation-theoretic machinery namespace."""

from spenn.reps.paths import (
    PathMetadata,
    VirtualPath,
    generate_virtual_paths,
    load_default_path_metadata,
    validate_virtual_path,
)

__all__ = [
    "PathMetadata",
    "VirtualPath",
    "generate_virtual_paths",
    "load_default_path_metadata",
    "validate_virtual_path",
]
