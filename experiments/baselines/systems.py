"""Read access to the comparison-system registry in ``systems.yaml``.

The registry is the single place that says which physical systems this program
compares and what reference energy each one is measured against. This module
owns reading it, so that callers that only need to *check membership* -- the
collector, most importantly -- do not each grow their own YAML load and their
own idea of where the file lives.

Why membership checking matters: a ``system_id`` is the join key between a run's
record and its reference energy. A record naming an id that is not in the
registry has nothing to be compared against, and until this module existed such
a record validated cleanly and reached ``results.jsonl``, where it looked like a
result rather than a dangling pointer.

Examples
--------
::

    from experiments.baselines.systems import known_system_ids

    if record.system_id not in known_system_ids():
        ...
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "systems.yaml"


class RegistryError(RuntimeError):
    """The registry could not be read, or is not shaped like a registry.

    Raised rather than falling back to an empty registry: an empty set of known
    ids would make every membership check fail, or -- worse, depending on how a
    caller spells the check -- succeed vacuously.
    """


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Return the parsed registry document.

    Parameters
    ----------
    path : pathlib.Path, optional
        Registry file to read. Defaults to the in-repo :data:`REGISTRY_PATH`.

    Returns
    -------
    dict
        The parsed document, guaranteed to carry a non-empty ``systems`` list
        of mappings.

    Raises
    ------
    RegistryError
        If the file is missing, unparseable, or not shaped like a registry.
    """

    registry_path = REGISTRY_PATH if path is None else path
    try:
        text = registry_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RegistryError(f"cannot read system registry at {registry_path}: {error}") from error
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RegistryError(f"system registry at {registry_path} is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        raise RegistryError(f"system registry at {registry_path} must be a mapping")
    systems = document.get("systems")
    if not isinstance(systems, list) or not systems:
        raise RegistryError(f"system registry at {registry_path} must carry a non-empty 'systems' list")
    for entry in systems:
        if not isinstance(entry, dict):
            raise RegistryError(f"system registry at {registry_path} has a non-mapping entry")
    return document


def system_ids(path: Path | None = None) -> frozenset[str]:
    """Return every declared system id.

    Parameters
    ----------
    path : pathlib.Path, optional
        Registry file to read. Defaults to the in-repo :data:`REGISTRY_PATH`.

    Returns
    -------
    frozenset of str
        Declared ids. Entries without a usable string id are rejected rather
        than skipped, because a skipped entry silently narrows the set that
        membership is checked against.

    Raises
    ------
    RegistryError
        If any entry lacks a non-empty string ``id``.
    """

    ids: set[str] = set()
    for entry in load_registry(path)["systems"]:
        system_id = entry.get("id")
        if not isinstance(system_id, str) or not system_id.strip():
            raise RegistryError("every registry entry needs a non-empty string id")
        ids.add(system_id)
    return frozenset(ids)


@lru_cache(maxsize=1)
def known_system_ids() -> frozenset[str]:
    """Return the in-repo registry's system ids, read once per process.

    Returns
    -------
    frozenset of str
        Ids declared by :data:`REGISTRY_PATH`.

    Raises
    ------
    RegistryError
        If the in-repo registry cannot be read.
    """

    return system_ids()


__all__ = ["REGISTRY_PATH", "RegistryError", "known_system_ids", "load_registry", "system_ids"]
