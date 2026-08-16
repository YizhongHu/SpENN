"""Canonical short-range Kato cusp factors for electron-electron and electron-nucleus pairs.

Both cusp factors here are `tpen.nn.factor.LogAmplitudeFactor` instances and
compose directly through `TPENWaveFunction.factors`; `ElectronElectronCusp`
additionally remains a `tpen.nn.envelope.Envelope` so it stays usable in the
legacy `AdditiveEnvelope` compatibility stack. Electron-nucleus cusp laws
separate the charge-fixed Kato first radial slope (`LinearElectronNucleusCuspLaw`,
the exact compatibility default) from an optional trainable regular curvature
term (`TrainableCurvatureElectronNucleusCuspLaw`) that contributes only at
second order, so it can never perturb the enforced ``u'(0+) = -Z`` slope. That
curvature term carries the electron-nucleus counterpart of
`ElectronElectronCusp`'s trainable ``range_parameter``; its outer-tail
consequence is stated on the class and is executable through
`TrainableCurvatureElectronNucleusCuspLaw.outer_tail_slope`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, electron_nuclear_displacements, pairwise_distances
from tpen.dependencies import require_torch, require_torch_functional, require_torch_nn
from tpen.nn.envelope import Envelope
from tpen.nn.factor import LogAmplitudeFactor, _inverse_softplus

torch = require_torch(feature="TPEN cusp modules")
nn = require_torch_nn(feature="TPEN cusp modules")
F = require_torch_functional(feature="TPEN cusp modules")


def rational_pair_cusp(
    distance: torch.Tensor,
    coefficient: torch.Tensor | float,
    range_parameter: torch.Tensor | float,
) -> torch.Tensor:
    """Return the rational pair cusp term.

    Parameters
    ----------
    distance : torch.Tensor
        Pair distances.
    coefficient : torch.Tensor or float
        Short-range cusp coefficient ``a``.
    range_parameter : torch.Tensor or float
        Positive range parameter ``b``.

    Returns
    -------
    torch.Tensor
        Values of ``a r / (1 + b r)`` with the same shape as `distance`.
    """

    output = coefficient * distance / (1.0 + range_parameter * distance)
    assert output.shape == distance.shape
    return output


class ElectronNucleusCuspLaw(ABC):
    """Typed radial law for a generic electron-nucleus cusp factor.

    A law computes the additive pairwise log-amplitude value for one
    electron-nucleus pair from raw (unclamped) distances and nuclear charges.
    Concrete laws are independently tested against the Kato cusp condition
    (the required ``d/dr`` slope at coalescence); this base class enforces no
    formula, only the typed value contract.

    Every law must fix its first-order Kato slope at ``r = 0`` by nuclear
    charge alone (``d/dr v_A(0) = -Z_A``, matching `LinearElectronNucleusCuspLaw`
    below). A law may separately expose an *optional* trainable regular radial
    component ``w_A(r)`` contributing only to second-order (and higher)
    curvature -- i.e. satisfying ``w_A(0) = 0`` and ``d/dr w_A(0) = 0`` -- so
    curvature can be learned without perturbing the enforced cusp condition.
    `TrainableCurvatureElectronNucleusCuspLaw` implements this ``w_A`` term.
    """

    @abstractmethod
    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return the pairwise cusp value with the same shape as `distance`.

        Parameters
        ----------
        distance : torch.Tensor
            Raw, unclamped electron-nucleus pair distances.
        charges : torch.Tensor
            Nuclear charges, broadcastable against `distance`.

        Returns
        -------
        torch.Tensor
            Cusp value with the same shape as `distance`.
        """


class LinearElectronNucleusCuspLaw(ElectronNucleusCuspLaw):
    """Compatibility law reproducing the existing He linear cusp ``-Z r``.

    This is the He linear cusp formerly hard-coded by the retired
    ``NuclearConfinement`` envelope, preserved here as a
    `ElectronNucleusCuspLaw` so existing systems have a like-for-like generic
    counterpart.
    """

    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return ``-Z r`` broadcast to the shape of `distance`."""

        return -charges * distance


class TrainableCurvatureElectronNucleusCuspLaw(nn.Module, ElectronNucleusCuspLaw):
    """Linear Kato slope plus an optional trainable second-order curvature term.

    Computes ``v_A(r) = -Z_A r + w_A(r)`` with ``w_A(r) = c r^2 / (1 + d r)``,
    where ``d`` is the strictly positive, optionally trainable RANGE parameter
    of the electron-nucleus cusp -- the counterpart of `ElectronElectronCusp`'s
    ``range_parameter`` -- and ``c`` is an unconstrained-sign curvature
    coefficient. ``w_A`` satisfies ``w_A(0) = 0`` and ``d/dr w_A(0) = 0`` for
    any ``c`` and positive ``d``: the curvature term contributes only from
    second order, so the charge-fixed Kato slope ``d/dr v_A(0) = -Z_A`` stays
    exact regardless of ``c`` or ``d``, before and after any optimizer step.

    Functional-form decision, recorded here because it moves outer-tail
    diagnostics by construction. Kato constrains only the ``r -> 0`` slope, so
    a saturating Pade law ``-Z r / (1 + a r)`` satisfies it too; but that law
    tends to the constant ``-Z / a``, i.e. ``log |psi|`` stops decreasing and
    the factor stops confining. This law is deliberately NON-saturating:
    ``w_A(r) -> (c / d) r`` at large ``r``, so ``v_A`` keeps growing linearly
    and its outer-tail slope is ``-Z_A + c / d`` (see `outer_tail_slope`). Two
    consequences bind consumers:

    - At ``c = 0`` the law is exactly ``-Z_A r`` and the outer-tail slope is
      exactly ``-Z_A``. At ``c != 0`` the tail slope is SHIFTED by ``c / d``,
      so outer-tail tolerances calibrated against the pure ``-Z r`` law must
      not be applied unchanged to a trained curvature law.
    - A decaying (normalizable) tail needs a negative slope, i.e.
      ``c / d < Z_A``. This class does not enforce that bound; a caller that
      trains ``c`` and ``d`` owns choosing an initialization, and if needed a
      constraint, that keeps ``c / d`` well below the nuclear charge.

    Gradient reachability, which matters when enabling training from a config:
    ``d`` enters the value only through ``w_A``, so at exactly ``c = 0`` the
    gradient with respect to ``d`` is identically zero. ``c`` itself still
    receives gradient there and moves off zero, which unlocks ``d`` from the
    next step; a configuration that wants the range trained from the first step
    should initialize `curvature_coefficient` nonzero.

    Parameters
    ----------
    curvature_coefficient : float, optional
        Initial value of the (unconstrained-sign) curvature coefficient ``c``.
    curvature_range : float, optional
        Positive initial value of the range parameter ``d``.
    trainable : bool, optional
        Whether ``c`` and ``d`` are optimized. When ``False`` they are fixed
        non-persistent buffers at the constructor values, so the module
        contributes no checkpoint state; when ``True`` it contributes
        ``raw_curvature_coefficient`` and ``raw_curvature_range`` to the state
        dict, which a ``strict=True`` restore will not accept against a
        differently configured law.
    eps : float, optional
        Positivity offset for the trainable range parameter.
    """

    def __init__(
        self,
        curvature_coefficient: float = 0.0,
        curvature_range: float = 1.0,
        trainable: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if curvature_range <= 0.0:
            raise ValueError(f"curvature_range must be positive, got {curvature_range}")
        self.trainable = bool(trainable)
        self.eps = eps
        if self.trainable:
            self.raw_curvature_coefficient = nn.Parameter(
                torch.tensor(float(curvature_coefficient), dtype=torch.float64)
            )
            self.raw_curvature_range = nn.Parameter(_inverse_softplus(float(curvature_range) - eps))
        else:
            self.register_buffer(
                "_curvature_coefficient",
                torch.tensor(float(curvature_coefficient), dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_curvature_range",
                torch.tensor(float(curvature_range), dtype=torch.float64),
                persistent=False,
            )

    @property
    def curvature_coefficient(self) -> torch.Tensor:
        """Return the (unconstrained-sign) curvature coefficient ``c``."""

        if self.trainable:
            return self.raw_curvature_coefficient
        return self._curvature_coefficient

    @property
    def curvature_range(self) -> torch.Tensor:
        """Return the positive curvature range parameter ``d``."""

        if self.trainable:
            return F.softplus(self.raw_curvature_range) + self.eps
        return self._curvature_range

    def outer_tail_slope(self, charges: torch.Tensor | float) -> torch.Tensor:
        """Return the large-``r`` limit of ``d/dr v_A``, namely ``-Z + c / d``.

        This makes the recorded outer-tail consequence of the chosen functional
        form executable: a tail diagnostic calibrated against the pure ``-Z r``
        law expects ``-Z``, while this law asymptotes to ``-Z + c / d``.

        Parameters
        ----------
        charges : torch.Tensor or float
            Nuclear charges ``Z``.

        Returns
        -------
        torch.Tensor
            Asymptotic radial slope with the shape of `charges`.
        """

        charge_tensor = torch.as_tensor(charges)
        if not torch.is_floating_point(charge_tensor):
            charge_tensor = charge_tensor.to(dtype=torch.get_default_dtype())
        coefficient = self.curvature_coefficient.to(device=charge_tensor.device, dtype=charge_tensor.dtype)
        range_parameter = self.curvature_range.to(device=charge_tensor.device, dtype=charge_tensor.dtype)
        return -charge_tensor + coefficient / range_parameter

    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return ``-Z r + c r^2 / (1 + d r)`` broadcast to the shape of `distance`."""

        linear = -charges * distance
        coefficient = self.curvature_coefficient.to(device=distance.device, dtype=distance.dtype)
        range_parameter = self.curvature_range.to(device=distance.device, dtype=distance.dtype)
        curvature = coefficient * distance.square() / (1.0 + range_parameter * distance)
        return linear + curvature


class ElectronNucleusCusp(LogAmplitudeFactor):
    """Generic electron-nucleus Kato cusp factor for arbitrary nuclei.

    This factor is constructed directly from an `AtomicConfiguration` -- the
    sole authority for nuclear geometry -- and never infers nuclear context
    from a batch. It uses raw, unclamped pair distances so the cusp condition
    is observed exactly at coalescence.

    Parameters
    ----------
    atoms : AtomicConfiguration
        Constructor-owned fixed nuclear geometry authority.
    law : ElectronNucleusCuspLaw or None, optional
        Cusp radial law. Defaults to `LinearElectronNucleusCuspLaw`, the He
        linear-cusp compatibility law.
    """

    def __init__(self, atoms: object, law: object = None) -> None:
        super().__init__()
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(f"{type(self).__name__} requires an AtomicConfiguration, got {type(atoms).__name__}")
        self.atoms = atoms
        resolved_law = law if law is not None else LinearElectronNucleusCuspLaw()
        if not isinstance(resolved_law, ElectronNucleusCuspLaw):
            raise TypeError(
                f"{type(self).__name__} law must be an ElectronNucleusCuspLaw, got {type(resolved_law).__name__}"
            )
        self.law = resolved_law

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the summed electron-nucleus cusp contribution."""

        atoms = self.atoms.to(device=batch.device, dtype=batch.dtype)
        distance = electron_nuclear_displacements(batch, nuclear_positions=atoms.positions).norm(dim=-1)
        charges = atoms.charges.reshape(1, 1, -1)
        value = self.law.value(distance, charges)
        if value.shape != distance.shape:
            raise ValueError(
                f"{type(self.law).__name__}.value must have shape {tuple(distance.shape)}, got {tuple(value.shape)}"
            )
        return value.sum(dim=(1, 2))


class ElectronElectronCusp(Envelope, LogAmplitudeFactor):
    """Spin-aware analytic electron-electron cusp envelope.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    same_spin_coefficient : float, optional
        Short-range coefficient for equal-spin electron pairs.
    opposite_spin_coefficient : float, optional
        Short-range coefficient for opposite-spin electron pairs.
    spinless_coefficient : float or None, optional
        Coefficient used when `ElectronBatch.spins` is absent. If ``None``,
        `same_spin_coefficient` is used.
    range_parameter : float, optional
        Default positive range parameter.
    same_range_parameter : float or None, optional
        Equal-spin range parameter. If ``None``, `range_parameter` is used.
    opposite_range_parameter : float or None, optional
        Opposite-spin range parameter. If ``None``, `range_parameter` is used.
    trainable_range : bool, optional
        Whether to optimize the range parameters through a softplus
        parametrization.
    eps : float, optional
        Numerical distance floor and positivity offset.
    """

    def __init__(
        self,
        enabled: bool = True,
        same_spin_coefficient: float = 0.25,
        opposite_spin_coefficient: float = 0.5,
        spinless_coefficient: float | None = None,
        range_parameter: float = 1.0,
        same_range_parameter: float | None = None,
        opposite_range_parameter: float | None = None,
        trainable_range: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(enabled=enabled)
        self.same_spin_coefficient = float(same_spin_coefficient)
        self.opposite_spin_coefficient = float(opposite_spin_coefficient)
        if spinless_coefficient is None:
            spinless_coefficient = same_spin_coefficient
        self.spinless_coefficient = float(spinless_coefficient)
        self.trainable_range = bool(trainable_range)
        self.eps = eps
        same_range = range_parameter if same_range_parameter is None else same_range_parameter
        opposite_range = range_parameter if opposite_range_parameter is None else opposite_range_parameter
        if self.trainable_range:
            self.raw_same_range = nn.Parameter(_inverse_softplus(float(same_range) - eps))
            self.raw_opposite_range = nn.Parameter(_inverse_softplus(float(opposite_range) - eps))
        else:
            self.register_buffer("same_range", torch.tensor(float(same_range)), persistent=False)
            self.register_buffer("opposite_range", torch.tensor(float(opposite_range)), persistent=False)

    @property
    def same_range_parameter(self) -> torch.Tensor:
        """Return the positive same-spin range parameter."""

        if self.trainable_range:
            return F.softplus(self.raw_same_range) + self.eps
        return self.same_range

    @property
    def opposite_range_parameter(self) -> torch.Tensor:
        """Return the positive opposite-spin range parameter."""

        if self.trainable_range:
            return F.softplus(self.raw_opposite_range) + self.eps
        return self.opposite_range

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the electron-electron cusp contribution."""

        distances = pairwise_distances(batch.positions, eps=self.eps).squeeze(-1)
        assert distances.shape == (batch.batch_size, batch.n_electrons, batch.n_electrons)
        tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        if batch.spins is None:
            contribution = rational_pair_cusp(distances, self.spinless_coefficient, self.same_range_parameter)
        else:
            spins = batch.spins.to(device=batch.device, dtype=batch.dtype)
            same_spin = spins.unsqueeze(2) == spins.unsqueeze(1)
            coefficients = torch.where(
                same_spin,
                torch.full_like(distances, self.same_spin_coefficient),
                torch.full_like(distances, self.opposite_spin_coefficient),
            )
            ranges = torch.where(
                same_spin,
                self.same_range_parameter,
                self.opposite_range_parameter,
            )
            contribution = rational_pair_cusp(distances, coefficients, ranges)
        output = contribution.masked_fill(~tri, 0.0).sum(dim=(1, 2))
        assert output.shape == (batch.batch_size,)
        return output


__all__ = [
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspLaw",
    "LinearElectronNucleusCuspLaw",
    "TrainableCurvatureElectronNucleusCuspLaw",
    "rational_pair_cusp",
]
