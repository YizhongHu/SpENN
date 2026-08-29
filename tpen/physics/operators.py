"""Typed identity for local-energy operators.

This module owns the *identity* of a physical operator, separately from the
class that computes it. The distinction is load-bearing for the lowering
registry: a registry keys dispatch on the operator's **type**, while a plan's
exactly-once manifest keys on the operator's **identity**. Two different
classes may compute the same physics — `ElectronNucleusInteraction` and
`ElectronNucleusPotential` differ only in where nuclear geometry comes from —
and must therefore share an `OperatorId` while remaining separate types.

Why an explicit declaration exists at all
-----------------------------------------
Without it, code that needs to know what a term *is* has no option but to
interrogate the object: `isinstance` scanning, matching on a `name` string,
`getattr` capability sniffing, or structural protocol checks. Those are all
the same act wearing different spellings, and each one erases the semantics it
is trying to recover. An operator declares its identity; nothing infers it.

Notes
-----
`OperatorId` is deliberately an open constructor rather than an enumeration.
The canonical TPEN identities are defined below so the shipped vocabulary is
readable in one place, but a third-party operator can mint its own id without
editing this module — an enum would force a central edit for every addition,
which is precisely the coupling the lowering registry exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, final


@final
@dataclass(frozen=True)
class OperatorId:
    """Immutable, hashable identity of a physical operator.

    Parameters
    ----------
    namespace : str
        Owning package or vendor, e.g. ``"tpen.physics"``. Namespacing keeps
        third-party operators from colliding with the canonical set.
    name : str
        Operator name, unique within ``namespace``.

    Notes
    -----
    This is a typed value rather than a bare :class:`str` on purpose. A string
    identity invites comparison against literals written at the call site
    (``term.name == "kinetic"``), where a typo silently fails to match and the
    operator is quietly skipped. Consumers compare against the module-level
    constants below, so a typo is an :class:`ImportError` at import time
    instead of a wrong answer at run time.

    Frozen and hashable because the planner's exactly-once manifest uses
    operator identities as mapping keys.
    """

    namespace: str
    name: str

    def __post_init__(self) -> None:
        # Empty components would make two unrelated operators compare equal,
        # which the exactly-once manifest would then read as a double claim.
        if not self.namespace or not self.namespace.strip():
            raise ValueError("OperatorId.namespace must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError("OperatorId.name must be a non-empty string")

    def __str__(self) -> str:
        return f"{self.namespace}:{self.name}"


class LocalEnergyOperator(Protocol):
    """Protocol for a term that declares which physical operator it computes.

    Attributes
    ----------
    operator_id : OperatorId
        The operator this term contributes. Declared, never inferred.

    Notes
    -----
    Intentionally *not* ``@runtime_checkable``. A structural check would only
    confirm that the attribute exists, so callers would still need a follow-up
    check on its type — reintroducing the probing this declaration removes.
    Static checking is the point; where a runtime narrowing is genuinely
    needed, use ``isinstance`` against a concrete class as a type guard, never
    as the mechanism that selects which physics runs.
    """

    operator_id: OperatorId


_OPERATOR_REGISTRY: dict[type[LocalEnergyOperator], OperatorId] = {}


def register_operator[T](operator_id: object) -> Callable[[type[T]], type[T]]:
    """Declare and register a local-energy operator class in one act.

    Declaring identity and being enumerable are deliberately a single act. If
    a term set ``operator_id`` and separately registered itself, a reviewer
    could update one and forget the other, and the registry would disagree
    with the class it describes.

    Notes
    -----
    ``operator_id`` is annotated :class:`object` rather than
    :class:`OperatorId` on purpose, and the ``isinstance`` check below is the
    real contract. This project runs ``typeguard``, which enforces annotations
    at run time: with the precise annotation, passing a malformed identity
    raised typeguard's own error before this function could raise its own, so
    the failure surfaced as an instrumentation detail rather than as this
    module's stated rule. Validating explicitly keeps the error identical
    whether or not typeguard is installed. The ``isinstance`` here is a guard
    that decides only *which error is raised*, never which physics runs.
    """

    if not isinstance(operator_id, OperatorId):
        raise TypeError("operator_id must be an OperatorId")

    def decorate(term_type: type[T]) -> type[T]:
        if term_type in _OPERATOR_REGISTRY:
            raise ValueError(f"{term_type.__name__} is already registered")
        term_type.operator_id = operator_id  # type: ignore[attr-defined]
        _OPERATOR_REGISTRY[term_type] = operator_id
        return term_type

    return decorate


def registered_operators() -> tuple[type[LocalEnergyOperator], ...]:
    """Return all operator classes declared through :func:`register_operator`."""

    return tuple(_OPERATOR_REGISTRY)


_TPEN = "tpen.physics"

#: Kinetic energy, :math:`-\\tfrac{1}{2}\\sum_i \\nabla_i^2`.
KINETIC_ENERGY = OperatorId(_TPEN, "kinetic_energy")

#: Electron-nucleus Coulomb attraction,
#: :math:`-\\sum_{i,A} Z_A / |r_i - R_A|`.
ELECTRON_NUCLEUS_COULOMB = OperatorId(_TPEN, "electron_nucleus_coulomb")

#: Electron-electron Coulomb repulsion,
#: :math:`\\sum_{i<j} 1 / |r_i - r_j|`.
ELECTRON_ELECTRON_COULOMB = OperatorId(_TPEN, "electron_electron_coulomb")

#: Born-Oppenheimer nuclear repulsion,
#: :math:`\\sum_{A<B} Z_A Z_B / |R_A - R_B|`.
NUCLEUS_NUCLEUS_COULOMB = OperatorId(_TPEN, "nucleus_nucleus_coulomb")

#: External harmonic confinement.
HARMONIC_TRAP = OperatorId(_TPEN, "harmonic_trap")


__all__ = [
    "ELECTRON_ELECTRON_COULOMB",
    "ELECTRON_NUCLEUS_COULOMB",
    "HARMONIC_TRAP",
    "KINETIC_ENERGY",
    "NUCLEUS_NUCLEUS_COULOMB",
    "LocalEnergyOperator",
    "OperatorId",
    "register_operator",
    "registered_operators",
]
