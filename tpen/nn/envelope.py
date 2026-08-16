"""Additive envelope factors for wavefunction log-amplitudes.

This module keeps two API generations side by side. `Envelope` and
`AdditiveEnvelope` are a supported minor-release compatibility surface: their
constructor, forward behavior, Hydra target, and `ModuleList` state-dict keys
must not change. `LogAmplitudeFactor`, `AdditiveCusp`, `ElectronNucleusCusp`,
and `AsymptoticDecay` are the new generic, atom-system-facing types (see
`main.typ`, "Electron-nucleus cusp (deferred)"); they compose independently
and do not replace the legacy envelope stack. `TPENWaveFunction` sums both
generations in one post-readout factor pipeline (see
`tpen/nn/spenn_wave_function.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, electron_nuclear_displacements, pairwise_distances
from tpen.dependencies import require_torch, require_torch_functional, require_torch_nn

torch = require_torch(feature="TPEN envelope modules")
nn = require_torch_nn(feature="TPEN envelope modules")
F = require_torch_functional(feature="TPEN envelope modules")


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


class Envelope(nn.Module):
    """Template for additive log-amplitude envelope factors.

    An envelope accepts an :class:`ElectronBatch` and returns a scalar
    contribution to ``log |psi|`` for each flattened configuration. Smooth
    confinement tails and short-range cusp factors both use this interface.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch envelope contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Envelope contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        if not self.enabled:
            return torch.zeros(flat_batch.batch_size, device=flat_batch.device, dtype=flat_batch.dtype)
        output = self.envelope_value(flat_batch)
        _check_envelope_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the enabled envelope contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Envelope contribution with shape ``[batch]``.
        """

        raise NotImplementedError("Envelope.envelope_value must be implemented by subclasses")


class AdditiveEnvelope(Envelope):
    """Envelope that sums a sequence of envelope components.

    Parameters
    ----------
    envelopes : iterable of torch.nn.Module, optional
        Envelope modules whose outputs are added. Each component must accept an
        :class:`ElectronBatch` and return a tensor of shape ``[batch]``.
    enabled : bool, optional
        Whether this envelope contributes to the output.
    """

    def __init__(self, envelopes: Iterable[nn.Module] = (), enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.envelopes = nn.ModuleList(tuple(envelopes))

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the sum of all component envelope contributions."""

        total = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        for index, envelope in enumerate(self.envelopes):
            value = envelope(batch)
            _check_envelope_tensor(value, batch, name=f"envelopes[{index}]")
            total = total + value
        return total


class GaussianConfinement(Envelope):
    """Smooth Gaussian envelope for harmonically trapped systems.

    This contributes

    ``log |psi| <- log |psi| - coefficient * sum_i |r_i|^2``.

    For a Hooke or harmonic-oscillator tail with frequency ``omega``, the fixed
    ground-state Gaussian coefficient is ``omega / 2``.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    coefficient : float, optional
        Nonnegative coefficient multiplying ``sum_i |r_i|^2``.
    trainable : bool, optional
        Whether to optimize the coefficient through a softplus
        parametrization. A trainable coefficient is strictly positive.
    """

    def __init__(
        self,
        enabled: bool = True,
        coefficient: float = 0.0,
        trainable: bool = False,
    ) -> None:
        super().__init__(enabled=enabled)
        if coefficient < 0.0:
            raise ValueError(f"coefficient must be nonnegative, got {coefficient}")
        self.trainable = bool(trainable)
        if self.trainable:
            self.raw_coefficient = nn.Parameter(_inverse_softplus(float(coefficient)))
        else:
            self.register_buffer(
                "_coefficient",
                torch.tensor(float(coefficient), dtype=torch.float64),
                persistent=False,
            )

    @property
    def coefficient(self) -> torch.Tensor:
        """Return the nonnegative harmonic-confinement coefficient."""

        if self.trainable:
            return F.softplus(self.raw_coefficient)
        return self._coefficient

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the smooth harmonic envelope contribution."""

        radius_squared = batch.positions.square().sum(dim=(1, 2))
        output = -self.coefficient.to(device=batch.device, dtype=batch.dtype) * radius_squared
        assert output.shape == (batch.batch_size,)
        return output


class HookeGaussianConfinement(GaussianConfinement):
    """Gaussian ground-state envelope for the Hooke / harmonic oscillator.

    This is :class:`GaussianConfinement` parametrized by the oscillator
    frequency ``omega`` instead of a raw coefficient. The fixed ground-state
    Gaussian uses ``coefficient = omega / 2``, contributing

    ``log |psi| <- log |psi| - (omega / 2) * sum_i |r_i|^2``.

    It supplies the common output-side asymptotic prior shared by every main
    architecture choice in the pair-stability study.

    Parameters
    ----------
    omega : float
        Positive oscillator frequency.
    enabled : bool, optional
        Whether this envelope contributes to the output.
    trainable : bool, optional
        Whether to optimize the coefficient through a softplus parametrization.
    """

    def __init__(self, *, omega: float, enabled: bool = True, trainable: bool = False) -> None:
        if omega <= 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        super().__init__(enabled=enabled, coefficient=float(omega) / 2.0, trainable=trainable)
        self.omega = float(omega)


class LogAmplitudeFactor(nn.Module):
    """Template for generic additive post-readout log-amplitude factors.

    This is the new-generation counterpart to `Envelope`: a factor accepts an
    `ElectronBatch` and returns one scalar contribution to `log |psi|` per
    flattened configuration, with a value-only forward (no auxiliary radial
    derivative structure). It is deliberately separate from `Envelope` so the
    legacy compatibility surface never has to change shape to accommodate
    generic atom-system consumers.
    """

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch factor contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Factor contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        output = self.factor_value(flat_batch)
        _check_envelope_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the factor contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Factor contribution with shape ``[batch]``.
        """

        raise NotImplementedError("LogAmplitudeFactor.factor_value must be implemented by subclasses")


class AdditiveCusp(LogAmplitudeFactor):
    """Generic composition summing typed `LogAmplitudeFactor` components.

    Parameters
    ----------
    factors : iterable of LogAmplitudeFactor, optional
        Cusp (or other additive) factors whose outputs are summed. Each
        component must be a `LogAmplitudeFactor`; this is a typed-interface
        check, not container traversal or class-name matching.
    """

    def __init__(self, factors: Iterable["LogAmplitudeFactor"] = ()) -> None:
        super().__init__()
        factors = tuple(factors)
        for factor in factors:
            if not isinstance(factor, LogAmplitudeFactor):
                raise TypeError(
                    f"AdditiveCusp components must be LogAmplitudeFactor, got {type(factor).__name__}"
                )
        self.factors = nn.ModuleList(factors)

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the sum of all component factor contributions."""

        total = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        for index, factor in enumerate(self.factors):
            value = factor(batch)
            _check_envelope_tensor(value, batch, name=f"factors[{index}]")
            total = total + value
        return total


class ElectronNucleusCuspLaw(ABC):
    """Typed radial law for a generic electron-nucleus cusp factor.

    A law computes the additive pairwise log-amplitude value for one
    electron-nucleus pair from raw (unclamped) distances and nuclear charges.
    Concrete laws are independently tested against the Kato cusp condition
    (the required ``d/dr`` slope at coalescence); this base class enforces no
    formula, only the typed value contract.
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


class AsymptoticDecay(nn.Module):
    """Template for an optional long-range log-amplitude decay factor.

    This is a separate, optional capability from cusp factors
    (`LogAmplitudeFactor`/`AdditiveCusp`) and from legacy feature envelopes
    (`Envelope`): it exists so a consumer that needs asymptotic decay can
    require this interface explicitly and fail loudly when it is absent,
    instead of a decay term being inferred or silently substituted.
    """

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch decay contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Decay contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        output = self.decay_value(flat_batch)
        _check_envelope_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def decay_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the decay contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Decay contribution with shape ``[batch]``.
        """

        raise NotImplementedError("AsymptoticDecay.decay_value must be implemented by subclasses")


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


def _check_envelope_tensor(value: object, batch: ElectronBatch, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    expected = (batch.batch_size,)
    if value.shape != expected:
        raise ValueError(f"{name} output must have shape {expected}, got {tuple(value.shape)}")


def _inverse_softplus(value: float) -> torch.Tensor:
    value = max(value, 1e-12)
    tensor = torch.tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(tensor))


__all__ = [
    "AdditiveCusp",
    "AdditiveEnvelope",
    "AsymptoticDecay",
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspLaw",
    "Envelope",
    "GaussianConfinement",
    "HookeGaussianConfinement",
    "LinearElectronNucleusCuspLaw",
    "LogAmplitudeFactor",
    "rational_pair_cusp",
]
