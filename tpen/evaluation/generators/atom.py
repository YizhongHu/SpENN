"""Atom-owned deterministic evaluation geometries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext


class HeliumRadialGridGenerator:
    """Generate explicit positive-radius electron-nucleus profiles for He.

    The generated grid contains both electron labels for each direction and
    radius. One electron approaches or recedes from the sole nucleus while the
    other remains at a fixed, nonzero spectator radius on the opposite ray.
    This isolates the electron-nucleus radial derivative without introducing
    either exact nuclear or electron-electron coalescence.

    Parameters
    ----------
    cusp_radii : sequence of float
        Strictly positive, increasing radii used for the one-sided ``r -> 0+``
        cusp fit.
    tail_radii : sequence of float
        Strictly positive, increasing radii beyond the cusp grid.
    spectator_radius : float
        Fixed positive radius of the non-probed electron.
    nuclear_positions : sequence
        Exactly one three-dimensional helium nuclear position.
    nuclear_charges : sequence
        Exactly one charge, equal to ``2``.
    n_directions : int, optional
        Number of Cartesian axes, at most three. Both antipodal rays are
        generated for each axis while every radius remains strictly positive.
    """

    name = "helium_radial_grid"

    def __init__(
        self,
        *,
        cusp_radii: Sequence[float],
        tail_radii: Sequence[float],
        spectator_radius: float,
        nuclear_positions: Sequence[Sequence[float]] | torch.Tensor,
        nuclear_charges: Sequence[float] | torch.Tensor,
        n_directions: int = 3,
    ) -> None:
        self.cusp_radii = _validate_radii(cusp_radii, name="cusp_radii")
        self.tail_radii = _validate_radii(tail_radii, name="tail_radii")
        if self.cusp_radii[-1] >= self.tail_radii[0]:
            raise ValueError("HeliumRadialGridGenerator requires cusp radii below tail radii")
        self.spectator_radius = float(spectator_radius)
        if not math.isfinite(self.spectator_radius) or self.spectator_radius <= 0.0:
            raise ValueError("HeliumRadialGridGenerator requires spectator_radius > 0")
        self.n_directions = int(n_directions)
        if not 1 <= self.n_directions <= 3:
            raise ValueError("HeliumRadialGridGenerator requires 1 <= n_directions <= 3")
        nuclei = torch.as_tensor(nuclear_positions, dtype=torch.float64)
        charges = torch.as_tensor(nuclear_charges, dtype=torch.float64)
        if tuple(nuclei.shape) != (1, 3):
            raise ValueError("HeliumRadialGridGenerator requires one three-dimensional nucleus")
        if tuple(charges.shape) != (1,) or not torch.equal(charges, torch.tensor([2.0], dtype=charges.dtype)):
            raise ValueError("HeliumRadialGridGenerator requires one Z=2 helium nucleus")
        if not torch.isfinite(nuclei).all():
            raise ValueError("HeliumRadialGridGenerator nuclear position must be finite")
        self.nuclear_positions = nuclei
        self.nuclear_charges = charges

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return He profile configurations with explicit nuclear context."""

        del model
        device = context.device or torch.device("cpu")
        dtype = context.dtype or torch.float64
        nuclei = self.nuclear_positions.to(device=device, dtype=dtype)
        charges = self.nuclear_charges.to(device=device, dtype=dtype)
        nucleus = nuclei[0]
        directions = torch.eye(3, device=device, dtype=dtype)[: self.n_directions]

        positions: list[torch.Tensor] = []
        regions: list[str] = []
        radii: list[float] = []
        direction_ids: list[int] = []
        direction_signs: list[int] = []
        probe_electrons: list[int] = []
        for region, region_radii in (("cusp", self.cusp_radii), ("tail", self.tail_radii)):
            for direction_id, axis in enumerate(directions):
                for direction_sign in (-1, 1):
                    direction = float(direction_sign) * axis
                    spectator = nucleus - self.spectator_radius * direction
                    for radius in region_radii:
                        probe_position = nucleus + radius * direction
                        for probe_electron in range(2):
                            configuration = torch.empty(2, 3, device=device, dtype=dtype)
                            configuration[probe_electron] = probe_position
                            configuration[1 - probe_electron] = spectator
                            positions.append(configuration)
                            regions.append(region)
                            radii.append(radius)
                            direction_ids.append(direction_id)
                            direction_signs.append(direction_sign)
                            probe_electrons.append(probe_electron)

        stacked = torch.stack(positions)
        spins = torch.tensor([[1.0, -1.0]], device=device, dtype=dtype).expand(stacked.shape[0], -1).clone()
        batch = ElectronBatch(
            positions=stacked,
            nuclear_positions=nuclei,
            nuclear_charges=charges,
            spins=spins,
            aux={},
        ).validate()
        sample_count = int(stacked.shape[0])
        return GeneratedConfigurations(
            batch=batch,
            metadata={
                "sample_index": torch.arange(sample_count, device=device),
                "profile_region": tuple(regions),
                "radius": torch.tensor(radii, device=device, dtype=dtype),
                "direction_id": torch.tensor(direction_ids, device=device, dtype=torch.long),
                "direction_sign": torch.tensor(direction_signs, device=device, dtype=torch.long),
                "probe_electron": torch.tensor(probe_electrons, device=device, dtype=torch.long),
                "nucleus_index": torch.zeros(sample_count, device=device, dtype=torch.long),
            },
        )


def _validate_radii(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    radii = tuple(float(value) for value in values)
    if not radii:
        raise ValueError(f"HeliumRadialGridGenerator requires non-empty {name}")
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError(f"HeliumRadialGridGenerator {name} must be finite and strictly positive")
    if any(right <= left for left, right in zip(radii, radii[1:])):
        raise ValueError(f"HeliumRadialGridGenerator {name} must be strictly increasing")
    return radii


__all__ = ["HeliumRadialGridGenerator"]
