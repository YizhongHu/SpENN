"""Stable config hashing for checkpoint manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf

RESOLVED_CONFIG_HASH_EXCLUSIONS = (
    ("run", "run_id"),
    ("run", "dir"),
)


def stable_config_hash(config: Any) -> str:
    """Return a sha256 hash of a JSON-safe, resolved config container.

    The hash uses canonical JSON with sorted keys. Unsupported objects fail
    loudly instead of being coerced with ``default=str``.
    """

    safe = _json_safe(to_plain_container(config))
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolved_config_hash(config: Any) -> str:
    """Hash the resolved run config after documented run-specific exclusions."""

    plain = to_plain_container(config)
    if isinstance(plain, Mapping):
        plain = copy.deepcopy(plain)
        for path in RESOLVED_CONFIG_HASH_EXCLUSIONS:
            _delete_path(plain, path)
    return stable_config_hash(plain)


def component_config_hash(config: Any, section: str) -> str | None:
    """Hash one top-level config section, returning ``None`` when absent."""

    plain = to_plain_container(config)
    if not isinstance(plain, Mapping) or section not in plain:
        return None
    return stable_config_hash(plain[section])


def checkpoint_hashes(config: Any) -> dict[str, str | None]:
    """Return the standard checkpoint manifest hashes for `config`."""

    return {
        "resolved_config": resolved_config_hash(config),
        "model_config": component_config_hash(config, "model"),
        "optimizer_config": component_config_hash(config, "optimizer"),
        "trainer_config": component_config_hash(config, "trainer"),
        "sampler_config": component_config_hash(config, "sampler"),
        "hamiltonian_config": component_config_hash(config, "hamiltonian_terms"),
    }


def to_plain_container(config: Any) -> Any:
    """Resolve OmegaConf containers and return plain Python containers."""

    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite float is not JSON-safe for hashing: {value!r}")
        return value
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"config hash mapping keys must be strings, got {type(key).__name__}")
            safe[key] = _json_safe(item)
        return safe
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"config value is not JSON/YAML-safe for hashing: {type(value).__name__}")


def _delete_path(container: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = container
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)
