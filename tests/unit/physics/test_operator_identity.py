"""Every shipped Hamiltonian term must declare which operator it computes.

The discovery helper below enumerates terms by looking for a ``local_energy``
attribute. That is deliberately **test-only** and is the thing this slice
exists to make unnecessary in production code: TPEN terms are duck-typed
against the ``HamiltonianTerm`` protocol and there is not yet a registry to
enumerate, so a test that wants to catch "someone added a term and forgot to
declare its identity" has nothing else to key on. It cannot key on
``operator_id`` itself, which would be circular — the terms it must catch are
exactly the ones missing that attribute.

Once the lowering registry lands, this helper should be replaced by iterating
the registry. Production dispatch must never enumerate this way.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import tpen.physics
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.operators import (
    ELECTRON_ELECTRON_COULOMB,
    ELECTRON_NUCLEUS_COULOMB,
    HARMONIC_TRAP,
    KINETIC_ENERGY,
    NUCLEUS_NUCLEUS_COULOMB,
    OperatorId,
)
from tpen.physics.potential import (
    ElectronNucleusInteraction,
    ElectronNucleusPotential,
    NucleusNucleusInteraction,
    NucleusNucleusPotential,
)


def _shipped_terms() -> list[type]:
    """Discover every Hamiltonian term class shipped under ``tpen.physics``."""

    discovered: list[type] = []
    for info in pkgutil.iter_modules(tpen.physics.__path__, f"{tpen.physics.__name__}."):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Only classes defined in this module, so a term is not counted
            # once per module that imports it.
            if obj.__module__ != info.name:
                continue
            # Protocols and result containers are not terms.
            if getattr(obj, "_is_protocol", False):
                continue
            if callable(getattr(obj, "local_energy", None)):
                discovered.append(obj)
    return discovered


def test_discovery_finds_the_known_terms() -> None:
    """Guard the guard: a discovery helper that finds nothing proves nothing.

    Without this, a helper broken by a refactor (renamed module, changed
    package layout) would silently yield an empty list and every
    parametrized test below would vacuously pass.
    """
    discovered = {cls.__name__ for cls in _shipped_terms()}

    assert "KineticEnergy" in discovered
    assert "ElectronNucleusPotential" in discovered
    # Seven terms ship today. This is a floor, not an equality: adding a term
    # should not fail this test, only failing to declare its identity should.
    assert len(discovered) >= 7


@pytest.mark.parametrize("term_type", _shipped_terms(), ids=lambda cls: cls.__name__)
def test_every_shipped_term_declares_a_typed_operator_id(term_type: type) -> None:
    """A term declares its operator; nothing infers it from name or type."""

    operator_id = getattr(term_type, "operator_id", None)

    assert operator_id is not None, (
        f"{term_type.__name__} does not declare operator_id. Add one from "
        "tpen.physics.operators, or mint a new OperatorId there if it "
        "computes an operator TPEN does not yet name."
    )
    # A bare string would satisfy "declares something" while reintroducing
    # identity-by-string-literal at every consumer.
    assert isinstance(operator_id, OperatorId), (
        f"{term_type.__name__}.operator_id must be an OperatorId, got "
        f"{type(operator_id).__name__}"
    )


def test_same_physics_from_different_classes_shares_one_identity() -> None:
    """Identity tracks the operator, not the implementing class.

    Both pairs below compute identical physics and differ only in where
    nuclear geometry comes from. The lowering registry keys dispatch on the
    class; the planner's exactly-once manifest keys on the identity. Collapsing
    the two concepts would make one of those two jobs impossible.
    """
    assert ElectronNucleusInteraction.operator_id == ELECTRON_NUCLEUS_COULOMB
    assert ElectronNucleusPotential.operator_id == ELECTRON_NUCLEUS_COULOMB
    assert NucleusNucleusInteraction.operator_id == NUCLEUS_NUCLEUS_COULOMB
    assert NucleusNucleusPotential.operator_id == NUCLEUS_NUCLEUS_COULOMB


def test_distinct_physics_have_distinct_identities() -> None:
    """Guards against a copy-paste that gives two operators one identity."""

    canonical = [
        KINETIC_ENERGY,
        ELECTRON_NUCLEUS_COULOMB,
        ELECTRON_ELECTRON_COULOMB,
        NUCLEUS_NUCLEUS_COULOMB,
        HARMONIC_TRAP,
    ]

    assert len(set(canonical)) == len(canonical)
    assert KineticEnergy.operator_id == KINETIC_ENERGY


def test_operator_id_is_hashable_and_compares_by_value() -> None:
    """The exactly-once manifest will use these as mapping keys."""

    first = OperatorId("vendor.pkg", "custom_operator")
    second = OperatorId("vendor.pkg", "custom_operator")

    assert first == second
    assert first is not second
    assert len({first, second}) == 1
    assert {first: "claimed"}[second] == "claimed"


def test_namespacing_separates_third_party_operators() -> None:
    """A vendor may reuse a name without colliding with the canonical set."""

    vendor_kinetic = OperatorId("vendor.pkg", "kinetic_energy")

    assert vendor_kinetic != KINETIC_ENERGY
    assert vendor_kinetic.name == KINETIC_ENERGY.name


@pytest.mark.parametrize(
    "namespace, name",
    [("", "kinetic"), ("   ", "kinetic"), ("tpen.physics", ""), ("tpen.physics", "   ")],
)
def test_empty_identity_components_are_rejected(namespace: str, name: str) -> None:
    """Empty components would make unrelated operators compare equal.

    The manifest would then read two distinct operators as one double claim,
    so this has to fail at construction rather than at plan time.
    """
    with pytest.raises(ValueError, match="non-empty string"):
        OperatorId(namespace, name)


def test_operator_id_is_immutable() -> None:
    """A mutable identity could be changed after the manifest recorded it."""

    operator_id = OperatorId("vendor.pkg", "custom_operator")

    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        operator_id.name = "mutated"  # type: ignore[misc]


def test_str_renders_the_namespaced_identity() -> None:
    """Plan diagnostics name operators; the rendering must stay stable."""

    assert str(KINETIC_ENERGY) == "tpen.physics:kinetic_energy"
