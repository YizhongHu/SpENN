"""Every registered Hamiltonian term must declare its operator identity."""

from __future__ import annotations

import pytest

import tpen.physics.operators as operators_module
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.operators import (
    ELECTRON_ELECTRON_COULOMB,
    ELECTRON_NUCLEUS_COULOMB,
    HARMONIC_TRAP,
    KINETIC_ENERGY,
    NUCLEUS_NUCLEUS_COULOMB,
    OperatorId,
    register_operator,
    registered_operators,
)
from tpen.physics.potential import (
    ElectronNucleusInteraction,
    ElectronNucleusPotential,
    NucleusNucleusInteraction,
    NucleusNucleusPotential,
)


def test_discovery_finds_the_known_terms() -> None:
    """Guard the guard: a discovery helper that finds nothing proves nothing.

    Without this, a helper broken by a refactor (renamed module, changed
    package layout) would silently yield an empty list and every
    parametrized test below would vacuously pass.
    """
    discovered = {cls.__name__ for cls in registered_operators()}

    assert "KineticEnergy" in discovered
    assert "ElectronNucleusPotential" in discovered
    # Seven terms ship today. This is a floor, not an equality: adding a term
    # should not fail this test, only failing to declare its identity should.
    assert len(discovered) >= 7


@pytest.mark.parametrize("term_type", registered_operators(), ids=lambda cls: cls.__name__)
def test_every_shipped_term_declares_a_typed_operator_id(term_type: type) -> None:
    """A term declares its operator; nothing infers it from name or type."""

    operator_id = term_type.operator_id
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


@pytest.fixture
def isolated_registry():
    """Restore the module-global registry after a test registers into it.

    ``register_operator`` mutates process-wide state. Without this, a test
    that registers a throwaway class leaves it in the registry for every
    later test in the session, so ``registered_operators()`` no longer
    describes what the package actually ships and the leak surfaces as an
    unrelated failure elsewhere.
    """
    snapshot = dict(operators_module._OPERATOR_REGISTRY)
    try:
        yield
    finally:
        operators_module._OPERATOR_REGISTRY.clear()
        operators_module._OPERATOR_REGISTRY.update(snapshot)


def test_registration_rejects_duplicate_class(isolated_registry) -> None:
    """One class cannot claim two identities.

    Silently accepting the second registration would let the registry and the
    class's own ``operator_id`` disagree about the same term.
    """

    class NotAnOperator:
        pass

    register_operator(OperatorId("test", "one"))(NotAnOperator)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already registered"):
        register_operator(OperatorId("test", "two"))(NotAnOperator)  # type: ignore[arg-type]


def test_registration_is_observable_through_the_registry(isolated_registry) -> None:
    """A registered class becomes enumerable, which is the point of the decorator.

    This does NOT prove the ``isolated_registry`` fixture restores state: the
    fixture's teardown runs after this body, so nothing here can observe it.
    Leakage is instead caught by
    ``test_shipped_registry_holds_no_test_only_classes`` below.
    """

    before = set(registered_operators())

    class Throwaway:
        pass

    register_operator(OperatorId("test", "throwaway"))(Throwaway)  # type: ignore[arg-type]

    assert Throwaway in set(registered_operators())
    assert before < set(registered_operators())


def test_registration_rejects_malformed_identity() -> None:
    """The explicit check, not typeguard, is what enforces the contract.

    ``register_operator`` annotates its parameter ``object`` deliberately so
    that this error comes from the module's own rule and stays identical
    whether or not typeguard instruments the call.
    """
    with pytest.raises(TypeError, match="must be an OperatorId"):
        register_operator("not-an-operator-id")  # type: ignore[arg-type]


def test_shipped_registry_holds_no_test_only_classes() -> None:
    """Detects registry leakage from the registration tests above.

    ``register_operator`` mutates process-wide state, so a test that registers
    a throwaway class without isolation would corrupt what
    ``registered_operators()`` reports for the rest of the session. Every
    identity minted by a test in this module uses the ``test`` namespace, so
    finding one here means an isolation fixture was dropped.

    This is order-sensitive by nature: it can only observe leaks from tests
    that already ran. Within this module pytest runs top-to-bottom, and the
    registering tests are above it.
    """
    leaked = [
        cls.__name__
        for cls in registered_operators()
        if cls.operator_id.namespace == "test"
    ]

    assert leaked == [], f"test-only classes leaked into the shipped registry: {leaked}"
