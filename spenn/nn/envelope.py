"""Additive envelope factors for wavefunction log-amplitudes."""

from __future__ import annotations

from collections.abc import Iterable

from spenn.data.batch import ElectronBatch, pairwise_distances
from spenn.dependencies import require_torch, require_torch_functional, require_torch_nn

torch = require_torch(feature="SpENN envelope modules")
nn = require_torch_nn(feature="SpENN envelope modules")
F = require_torch_functional(feature="SpENN envelope modules")


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


class HarmonicConfinement(Envelope):
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


class HookeGaussianEnvelope(HarmonicConfinement):
    """Gaussian ground-state envelope for the Hooke / harmonic oscillator.

    This is :class:`HarmonicConfinement` parametrized by the oscillator
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


class ElectronElectronCusp(Envelope):
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
    "AdditiveEnvelope",
    "ElectronElectronCusp",
    "Envelope",
    "HarmonicConfinement",
    "HookeGaussianEnvelope",
    "rational_pair_cusp",
]
