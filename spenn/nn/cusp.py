"""Trainable cusp factors for wavefunction log-amplitudes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spenn.data.batch import ElectronBatch
from spenn.utils.tensor_utils import pairwise_distances


def rational_pair_cusp(distance: torch.Tensor, coefficient: torch.Tensor | float, range_parameter: torch.Tensor | float) -> torch.Tensor:
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


def rational_nuclear_cusp(distance: torch.Tensor, charge: torch.Tensor, range_parameter: torch.Tensor | float) -> torch.Tensor:
    """Return the rational electron-nucleus cusp term.

    Parameters
    ----------
    distance : torch.Tensor
        Electron-nucleus distances.
    charge : torch.Tensor
        Nuclear charges broadcastable to `distance`.
    range_parameter : torch.Tensor or float
        Positive range parameter ``b``.

    Returns
    -------
    torch.Tensor
        Values of ``-Z r / (1 + b r)`` with the same shape as `distance`.
    """

    output = -charge * distance / (1.0 + range_parameter * distance)
    assert output.shape == distance.shape
    return output


class Cusp(nn.Module):
    """Template for additive log-amplitude cusp factors.

    Parameters
    ----------
    enabled : bool, optional
        Whether this cusp contributes to the output.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, enabled: bool = True, **_: object) -> None:
        super().__init__()
        self.enabled = enabled

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch cusp contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Cusp contribution with shape ``[batch]`` after sample flattening.
        """

        flat_batch = batch.flatten_samples()
        if not self.enabled:
            return torch.zeros(flat_batch.batch_size, device=flat_batch.device, dtype=flat_batch.dtype)
        output = self.cusp_value(flat_batch)
        assert output.shape == (flat_batch.batch_size,)
        return output

    def cusp_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the enabled cusp contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Cusp contribution with shape ``[batch]``.
        """

        raise NotImplementedError("Cusp.cusp_value must be implemented by subclasses")


class ElectronElectronCusp(Cusp):
    """Spin-aware analytic electron-electron cusp.

    Parameters
    ----------
    enabled : bool, optional
        Whether this cusp contributes to the output.
    same_spin_coefficient : float, optional
        Short-range coefficient for equal-spin electron pairs.
    opposite_spin_coefficient : float, optional
        Short-range coefficient for opposite-spin electron pairs.
    spinless_coefficient : float or None, optional
        Coefficient used when `ElectronBatch.spins` is absent. If ``None``,
        `coefficient` is used when supplied, otherwise
        `same_spin_coefficient`.
    coefficient : float or None, optional
        Backward-compatible alias for the spinless coefficient.
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
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        enabled: bool = True,
        same_spin_coefficient: float = 0.25,
        opposite_spin_coefficient: float = 0.5,
        spinless_coefficient: float | None = None,
        coefficient: float | None = None,
        range_parameter: float = 1.0,
        same_range_parameter: float | None = None,
        opposite_range_parameter: float | None = None,
        trainable_range: bool = False,
        eps: float = 1e-12,
        **_: object,
    ) -> None:
        super().__init__(enabled=enabled)
        self.same_spin_coefficient = float(same_spin_coefficient)
        self.opposite_spin_coefficient = float(opposite_spin_coefficient)
        if spinless_coefficient is None:
            spinless_coefficient = coefficient if coefficient is not None else same_spin_coefficient
        self.spinless_coefficient = float(spinless_coefficient)
        self.trainable_range = trainable_range
        self.eps = eps
        same_range = range_parameter if same_range_parameter is None else same_range_parameter
        opposite_range = range_parameter if opposite_range_parameter is None else opposite_range_parameter
        if trainable_range:
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

    def cusp_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the electron-electron cusp contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Pair cusp sum with shape ``[batch]``.
        """

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
                self.same_range_parameter.to(device=batch.device, dtype=batch.dtype),
                self.opposite_range_parameter.to(device=batch.device, dtype=batch.dtype),
            )
            contribution = rational_pair_cusp(distances, coefficients, ranges)
        output = contribution.masked_fill(~tri, 0.0).sum(dim=(1, 2))
        assert output.shape == (batch.batch_size,)
        return output


class NuclearCusp(Cusp):
    """Analytic electron-nucleus cusp with a global range parameter.

    Parameters
    ----------
    enabled : bool, optional
        Whether this cusp contributes to the output.
    nuclear_positions : torch.Tensor or None, optional
        Constructor-owned nuclear positions with shape ``[n_nuclei, dim]``.
        If absent, batch or system nuclear positions are used.
    nuclear_charges : torch.Tensor or None, optional
        Constructor-owned nuclear charges with shape ``[n_nuclei]``. If
        absent, batch or system nuclear charges are used.
    range_parameter : float, optional
        Positive global range parameter.
    trainable_range : bool, optional
        Whether to optimize the range parameter through a softplus
        parametrization.
    eps : float, optional
        Numerical distance floor and positivity offset.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        enabled: bool = True,
        nuclear_positions: torch.Tensor | None = None,
        nuclear_charges: torch.Tensor | None = None,
        range_parameter: float = 1.0,
        trainable_range: bool = False,
        eps: float = 1e-12,
        **_: object,
    ) -> None:
        super().__init__(enabled=enabled)
        self.eps = eps
        self.trainable_range = trainable_range
        self.register_buffer(
            "nuclear_positions",
            None if nuclear_positions is None else torch.as_tensor(nuclear_positions, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "nuclear_charges",
            None if nuclear_charges is None else torch.as_tensor(nuclear_charges, dtype=torch.float64),
            persistent=False,
        )
        if trainable_range:
            self.raw_range = nn.Parameter(_inverse_softplus(float(range_parameter) - eps))
        else:
            self.register_buffer("range", torch.tensor(float(range_parameter)), persistent=False)

    @property
    def range_parameter(self) -> torch.Tensor:
        """Return the positive global nuclear-cusp range parameter."""

        if self.trainable_range:
            return F.softplus(self.raw_range) + self.eps
        return self.range

    def cusp_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the electron-nucleus cusp contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch with constructor, batch, or system
            nuclear metadata.

        Returns
        -------
        torch.Tensor
            Electron-nucleus cusp sum with shape ``[batch]``.
        """

        nuclear_positions, nuclear_charges = self._nuclear_data(batch)
        nuclear_positions = nuclear_positions.to(device=batch.device, dtype=batch.dtype)
        nuclear_charges = nuclear_charges.to(device=batch.device, dtype=batch.dtype)
        if nuclear_positions.ndim == 2:
            nuclear_positions = nuclear_positions.unsqueeze(0).expand(batch.batch_size, -1, -1)
        if nuclear_charges.ndim == 1:
            nuclear_charges = nuclear_charges.unsqueeze(0).expand(batch.batch_size, -1)
        assert nuclear_positions.shape[0] == batch.batch_size
        assert nuclear_positions.shape[-1] == batch.spatial_dim
        assert nuclear_charges.shape == nuclear_positions.shape[:2]
        distances = torch.linalg.norm(batch.positions.unsqueeze(2) - nuclear_positions.unsqueeze(1), dim=-1).clamp_min(self.eps)
        contribution = rational_nuclear_cusp(
            distances,
            nuclear_charges.unsqueeze(1),
            self.range_parameter.to(device=batch.device, dtype=batch.dtype),
        )
        output = contribution.sum(dim=(1, 2))
        assert output.shape == (batch.batch_size,)
        return output

    def _nuclear_data(self, batch: ElectronBatch) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.nuclear_positions
        charges = self.nuclear_charges
        if positions is None:
            positions = batch.nuclear_positions
        if charges is None:
            charges = batch.nuclear_charges
        system = batch.system
        if positions is None and system is not None:
            positions = getattr(system, "nuclear_positions", None)
        if charges is None and system is not None:
            charges = getattr(system, "nuclear_charges", None)
        if positions is None or charges is None:
            raise ValueError("NuclearCusp requires nuclear positions and charges")
        return positions, charges


class NuclearFeatureCusp(Cusp):
    """Placeholder for cusp factors using learned nuclear features.

    Notes
    -----
    This module reserves the public API for a future model that receives
    nuclear positions and learned atomic encodings alongside electrons.
    """

    def cusp_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Raise until nuclear-feature cusp inputs are designed.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Raises
        ------
        NotImplementedError
            Always raised in this scaffold.
        """

        raise NotImplementedError("NuclearFeatureCusp is reserved for models with nuclear feature inputs")


def _inverse_softplus(value: float) -> torch.Tensor:
    value = max(value, 1e-12)
    tensor = torch.tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(tensor))


__all__ = [
    "Cusp",
    "ElectronElectronCusp",
    "NuclearCusp",
    "NuclearFeatureCusp",
    "rational_nuclear_cusp",
    "rational_pair_cusp",
]
