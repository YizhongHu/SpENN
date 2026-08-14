"""Schema tests for the baseline comparison-system registry.

These tests are deliberately about *shape and evidence discipline*, not physics.
They guard the two failure modes that would silently corrupt the baseline
comparison: a malformed entry, and a reference energy that is present without a
citation (or absent while still claiming to be transcribed).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "systems.yaml"

# Keys every registry entry must carry. Missing any of these is a hard failure:
# downstream collection joins on `id` and reports against
# `reference_energy_hartree`, so a partial entry is worse than no entry.
REQUIRED_KEYS = frozenset(
    {
        "id",
        "n_up",
        "n_down",
        "spatial_dim",
        "hamiltonian",
        "reference_energy_hartree",
        "reference_source",
        "confidence",
    }
)

ALLOWED_CONFIDENCE = frozenset({"transcribed", "unverified"})


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """Return the parsed registry document.

    Returns
    -------
    dict
        The full ``systems.yaml`` document.
    """

    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict), "systems.yaml must parse to a mapping"
    return document


@pytest.fixture(scope="module")
def entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the registry's system entries.

    Returns
    -------
    list of dict
        One mapping per comparison system.
    """

    systems = registry.get("systems")
    assert isinstance(systems, list) and systems, "systems must be a non-empty list"
    for entry in systems:
        assert isinstance(entry, dict), "each system entry must be a mapping"
    return systems


def _walk_floats(value: Any) -> list[float]:
    """Return every float reachable from ``value``.

    Parameters
    ----------
    value : Any
        Parsed YAML fragment: mapping, sequence, or scalar.

    Returns
    -------
    list of float
        All floats found, in document order.
    """

    if isinstance(value, dict):
        return [found for item in value.values() for found in _walk_floats(item)]
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in _walk_floats(item)]
    # bool is a subclass of int but never a physical quantity here.
    if isinstance(value, float):
        return [value]
    return []


def test_document_header(registry: dict[str, Any]) -> None:
    """The registry declares its schema version and unit convention."""

    assert registry.get("schema_version") == 1
    units = registry.get("units")
    assert units == {"energy": "hartree", "length": "bohr"}


def test_required_keys_present(entries: list[dict[str, Any]]) -> None:
    """Every entry carries the full required key set."""

    for entry in entries:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"{entry.get('id', '<no id>')} is missing keys: {sorted(missing)}"


def test_ids_are_unique_non_empty_strings(entries: list[dict[str, Any]]) -> None:
    """Entry ids are usable as join keys."""

    ids = [entry["id"] for entry in entries]
    for system_id in ids:
        assert isinstance(system_id, str) and system_id.strip(), "id must be a non-empty string"
    assert len(set(ids)) == len(ids), "system ids must be unique"


def test_no_nan_or_infinite_values(registry: dict[str, Any]) -> None:
    """No NaN or infinity anywhere in the document.

    YAML happily parses ``.nan`` and ``.inf``; either would poison an energy
    comparison without raising, so reject them at load time.
    """

    for value in _walk_floats(registry):
        assert math.isfinite(value), f"non-finite value in registry: {value!r}"


def test_spin_counts_are_positive_integers(entries: list[dict[str, Any]]) -> None:
    """Spin sectors are integer counts with at least one spin-up electron.

    The majority sector is required to be positive; ``n_down`` may be zero to
    allow a fully spin-polarized system.
    """

    for entry in entries:
        n_up, n_down = entry["n_up"], entry["n_down"]
        for name, count in (("n_up", n_up), ("n_down", n_down)):
            assert isinstance(count, int) and not isinstance(count, bool), (
                f"{entry['id']}.{name} must be an int"
            )
        assert n_up >= 1, f"{entry['id']}.n_up must be positive"
        assert n_down >= 0, f"{entry['id']}.n_down must be non-negative"
        assert n_up >= n_down, f"{entry['id']} must put the majority sector in n_up"


def test_spatial_dim_is_positive_integer(entries: list[dict[str, Any]]) -> None:
    """Spatial dimension is a positive integer."""

    for entry in entries:
        spatial_dim = entry["spatial_dim"]
        assert isinstance(spatial_dim, int) and not isinstance(spatial_dim, bool)
        assert spatial_dim >= 1, f"{entry['id']}.spatial_dim must be positive"


def test_hamiltonian_terms_are_named(entries: list[dict[str, Any]]) -> None:
    """Each entry lists at least one named Hamiltonian term."""

    for entry in entries:
        hamiltonian = entry["hamiltonian"]
        assert isinstance(hamiltonian, dict), f"{entry['id']}.hamiltonian must be a mapping"
        terms = hamiltonian.get("terms")
        assert isinstance(terms, list) and terms, f"{entry['id']} must declare Hamiltonian terms"
        for term in terms:
            assert isinstance(term, dict), f"{entry['id']} term entries must be mappings"
            name = term.get("term")
            assert isinstance(name, str) and name.strip(), f"{entry['id']} term needs a name"


def test_reference_energy_is_null_or_finite_float(entries: list[dict[str, Any]]) -> None:
    """Reference energies are either absent or real numbers."""

    for entry in entries:
        energy = entry["reference_energy_hartree"]
        if energy is None:
            continue
        assert isinstance(energy, (int, float)) and not isinstance(energy, bool), (
            f"{entry['id']}.reference_energy_hartree must be numeric or null"
        )
        assert math.isfinite(float(energy)), f"{entry['id']} reference energy must be finite"


def test_confidence_matches_evidence(entries: list[dict[str, Any]]) -> None:
    """Confidence labels agree with what the entry actually carries.

    The evidence rule for this program: a number may only be present when it was
    transcribed from a cited source, and a missing number may never be labelled
    as transcribed.
    """

    for entry in entries:
        confidence = entry["confidence"]
        assert confidence in ALLOWED_CONFIDENCE, (
            f"{entry['id']}.confidence must be one of {sorted(ALLOWED_CONFIDENCE)}"
        )
        energy = entry["reference_energy_hartree"]
        source = entry["reference_source"]
        if energy is None:
            assert confidence == "unverified", (
                f"{entry['id']} has no reference energy and cannot be transcribed"
            )
        else:
            assert confidence == "transcribed", (
                f"{entry['id']} carries a reference energy so it must be transcribed"
            )
            assert isinstance(source, str) and source.strip(), (
                f"{entry['id']} carries a reference energy without a citation"
            )


def test_no_facility_absolute_paths() -> None:
    """The registry never hard-codes a facility filesystem path."""

    text = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "/n/netscratch" not in text
    assert "/n/holy" not in text
