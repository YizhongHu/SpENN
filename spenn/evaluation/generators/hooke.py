"""Deterministic Hooke-pair evaluation generators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from spenn.data.batch import ElectronBatch
from spenn.evaluation.bundle import GeneratedConfigurations
from spenn.evaluation.protocols import EvaluationContext


class CuspGridGenerator:
    """Generate paired near-coalescence two-electron Hooke configurations."""

    name = "cusp_grid"

    def __init__(
        self,
        *,
        n_points: int,
        r12_min: float,
        r12_max: float,
        n_directions: int,
        center_of_mass_radii: Sequence[float],
        paired_directions: bool = True,
        spin_pair: Literal["opposite", "same"] = "opposite",
        seed: int | None = None,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        spatial_dim: int = 3,
    ) -> None:
        self.n_points = int(n_points)
        self.r12_min = float(r12_min)
        self.r12_max = float(r12_max)
        self.n_directions = int(n_directions)
        self.center_of_mass_radii = tuple(float(value) for value in center_of_mass_radii)
        self.paired_directions = bool(paired_directions)
        self.spin_pair = spin_pair
        self.seed = seed
        self.dtype = _dtype(dtype)
        self.device = None if device is None else torch.device(device)
        self.spatial_dim = int(spatial_dim)

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return a log-spaced pair-distance cusp grid."""

        del model
        device = self.device or context.device or torch.device("cpu")
        dtype = self.dtype or context.dtype or torch.float64
        directions = _directions(self.spatial_dim, self.n_directions, device=device, dtype=dtype, seed=self.seed)
        distances = torch.logspace(
            math.log10(self.r12_min),
            math.log10(self.r12_max),
            self.n_points,
            device=device,
            dtype=dtype,
        )

        positions: list[torch.Tensor] = []
        r12_values: list[torch.Tensor] = []
        direction_ids: list[int] = []
        pair_ids: list[int] = []
        direction_signs: list[int] = []
        center_radii: list[float] = []
        center_ids: list[int] = []
        sample_index = 0
        for center_id, center_radius in enumerate(self.center_of_mass_radii):
            for direction_id, direction in enumerate(directions):
                center_direction = directions[(direction_id + 1) % len(directions)]
                center = float(center_radius) * center_direction
                signs = (1, -1) if self.paired_directions else (1,)
                for distance in distances:
                    for sign in signs:
                        signed_direction = float(sign) * direction
                        positions.append(
                            torch.stack(
                                [
                                    center + 0.5 * distance * signed_direction,
                                    center - 0.5 * distance * signed_direction,
                                ]
                            )
                        )
                        r12_values.append(distance)
                        direction_ids.append(direction_id)
                        pair_ids.append(center_id * len(directions) + direction_id)
                        direction_signs.append(sign)
                        center_radii.append(float(center_radius))
                        center_ids.append(center_id)
                    sample_index += len(signs)

        stacked = torch.stack(positions) if positions else torch.empty(0, 2, self.spatial_dim, device=device, dtype=dtype)
        metadata = {
            "r12": torch.stack(r12_values) if r12_values else torch.empty(0, device=device, dtype=dtype),
            "direction_id": torch.tensor(direction_ids, device=device, dtype=torch.long),
            "antipodal_pair_id": torch.tensor(pair_ids, device=device, dtype=torch.long),
            "direction_sign": torch.tensor(direction_signs, device=device, dtype=torch.long),
            "center_of_mass_radius": torch.tensor(center_radii, device=device, dtype=dtype),
            "center_of_mass_id": torch.tensor(center_ids, device=device, dtype=torch.long),
            "spin_pair": self.spin_pair,
            "sample_index": torch.arange(stacked.shape[0], device=device),
        }
        return GeneratedConfigurations(batch=_batch_from_positions(stacked, spin_pair=self.spin_pair), metadata=metadata)


class TailGridGenerator:
    """Move a fixed two-electron relative displacement outward in radius."""

    name = "tail_grid"

    def __init__(
        self,
        *,
        radius_min: float,
        radius_max: float,
        n_points: int,
        pair_distance: float,
        n_directions: int,
        seed: int | None = None,
        spacing: Literal["linear", "log"] = "linear",
        spatial_dim: int = 3,
    ) -> None:
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.n_points = int(n_points)
        self.pair_distance = float(pair_distance)
        self.n_directions = int(n_directions)
        self.seed = seed
        self.spacing = spacing
        if self.spacing not in ("linear", "log"):
            raise ValueError(f"unsupported spacing {self.spacing!r}")
        if self.spacing == "log" and self.radius_min <= 0.0:
            raise ValueError("TailGridGenerator requires radius_min > 0 for log spacing")
        self.spatial_dim = int(spatial_dim)

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return a center-of-mass tail grid."""

        del model
        device = context.device or torch.device("cpu")
        dtype = context.dtype or torch.float64
        directions = _directions(self.spatial_dim, self.n_directions, device=device, dtype=dtype, seed=self.seed)
        radii = _grid(self.radius_min, self.radius_max, self.n_points, spacing=self.spacing, device=device, dtype=dtype)
        pair = torch.tensor(self.pair_distance, device=device, dtype=dtype)

        positions: list[torch.Tensor] = []
        radius_values: list[torch.Tensor] = []
        direction_ids: list[int] = []
        relative_direction_ids: list[int] = []
        for direction_id, direction in enumerate(directions):
            relative_direction_id = (direction_id + 1) % len(directions)
            relative_direction = directions[relative_direction_id]
            for radius in radii:
                center = radius * direction
                displacement = pair * relative_direction
                positions.append(torch.stack([center + 0.5 * displacement, center - 0.5 * displacement]))
                radius_values.append(radius)
                direction_ids.append(direction_id)
                relative_direction_ids.append(relative_direction_id)

        stacked = torch.stack(positions) if positions else torch.empty(0, 2, self.spatial_dim, device=device, dtype=dtype)
        metadata = {
            "radius": torch.stack(radius_values) if radius_values else torch.empty(0, device=device, dtype=dtype),
            "direction_id": torch.tensor(direction_ids, device=device, dtype=torch.long),
            "relative_direction_id": torch.tensor(relative_direction_ids, device=device, dtype=torch.long),
            "pair_distance": torch.full((stacked.shape[0],), self.pair_distance, device=device, dtype=dtype),
            "sample_index": torch.arange(stacked.shape[0], device=device),
        }
        return GeneratedConfigurations(batch=_batch_from_positions(stacked), metadata=metadata)


class StratifiedGeometryGenerator:
    """Generate deterministic pseudo-random configurations from geometry strata."""

    name = "stratified_geometry"

    def __init__(
        self,
        *,
        n_samples: int,
        strata: Mapping[str, float],
        seed: int,
        bounds: Mapping[str, object],
        spatial_dim: int = 3,
    ) -> None:
        self.n_samples = int(n_samples)
        self.strata = dict(strata)
        self.seed = int(seed)
        self.bounds = dict(bounds)
        self.spatial_dim = int(spatial_dim)

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return seeded random configurations with stratum bookkeeping."""

        del model
        device = context.device or torch.device("cpu")
        dtype = context.dtype or torch.float64
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        names, weights = _normalized_weights(self.strata)
        choices = torch.multinomial(torch.tensor(weights, dtype=torch.float64), self.n_samples, replacement=True, generator=generator)

        positions: list[torch.Tensor] = []
        strata_names: list[str] = []
        r12_values: list[float] = []
        radius_values: list[float] = []
        for choice in choices.tolist():
            stratum = names[choice]
            r12, radius = _sample_geometry_for_stratum(stratum, self.bounds, generator=generator)
            direction = _random_unit(self.spatial_dim, generator=generator, device=device, dtype=dtype)
            center_direction = _random_unit(self.spatial_dim, generator=generator, device=device, dtype=dtype)
            center = torch.tensor(radius, device=device, dtype=dtype) * center_direction
            displacement = torch.tensor(r12, device=device, dtype=dtype) * direction
            positions.append(torch.stack([center + 0.5 * displacement, center - 0.5 * displacement]))
            strata_names.append(stratum)
            r12_values.append(float(r12))
            radius_values.append(float(radius))

        stacked = torch.stack(positions) if positions else torch.empty(0, 2, self.spatial_dim, device=device, dtype=dtype)
        metadata = {
            "stratum": tuple(strata_names),
            "sample_index": torch.arange(stacked.shape[0], device=device),
            "r12": torch.tensor(r12_values, device=device, dtype=dtype),
            "radius": torch.tensor(radius_values, device=device, dtype=dtype),
        }
        return GeneratedConfigurations(batch=_batch_from_positions(stacked), metadata=metadata)


class HookeOrbitalGenerator:
    """Generate Hooke-envelope-style validation configurations."""

    name = "hooke_orbital"

    def __init__(
        self,
        *,
        n_samples: int,
        omega: float = 0.5,
        seed: int,
        envelope_scale: float | None = None,
        spatial_dim: int = 3,
    ) -> None:
        self.n_samples = int(n_samples)
        self.omega = float(omega)
        self.seed = int(seed)
        self.envelope_scale = envelope_scale
        self.spatial_dim = int(spatial_dim)

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return seeded samples from a Hooke-inspired Gaussian envelope."""

        del model
        device = context.device or torch.device("cpu")
        dtype = context.dtype or torch.float64
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        scale = float(self.envelope_scale) if self.envelope_scale is not None else 1.0 / math.sqrt(max(self.omega, 1.0e-12))
        positions = scale * torch.randn(self.n_samples, 2, self.spatial_dim, generator=generator, dtype=dtype).to(device=device)
        r12 = torch.linalg.norm(positions[:, 0, :] - positions[:, 1, :], dim=-1)
        radius = torch.linalg.norm(positions.mean(dim=1), dim=-1)
        metadata = {
            "sample_index": torch.arange(positions.shape[0], device=device),
            "radius": radius,
            "r12": r12,
            "proposal_family": "hooke_orbital",
        }
        return GeneratedConfigurations(batch=_batch_from_positions(positions), metadata=metadata)


def _batch_from_positions(positions: torch.Tensor, *, spin_pair: str = "opposite") -> ElectronBatch:
    if spin_pair == "same":
        spins = torch.ones(positions.shape[:2], device=positions.device, dtype=positions.dtype)
    else:
        spins = torch.tensor([[1.0, -1.0]], device=positions.device, dtype=positions.dtype).repeat(positions.shape[0], 1)
    return ElectronBatch(positions=positions, spins=spins, aux={})


def _directions(
    spatial_dim: int,
    n_directions: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> tuple[torch.Tensor, ...]:
    base: list[torch.Tensor] = [torch.eye(spatial_dim, device=device, dtype=dtype)[idx] for idx in range(spatial_dim)]
    if spatial_dim >= 2:
        vec = torch.zeros(spatial_dim, device=device, dtype=dtype)
        vec[:2] = 1.0
        base.append(vec / torch.linalg.norm(vec))
    if spatial_dim >= 3:
        vec = torch.zeros(spatial_dim, device=device, dtype=dtype)
        vec[:3] = 1.0
        base.append(vec / torch.linalg.norm(vec))
    n = max(1, int(n_directions))
    if seed is None:
        return tuple(base[index % len(base)] for index in range(n))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return tuple(_random_unit(spatial_dim, generator=generator, device=device, dtype=dtype) for _ in range(n))


def _grid(
    minimum: float,
    maximum: float,
    n_points: int,
    *,
    spacing: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = max(1, int(n_points))
    if spacing == "log":
        return torch.logspace(math.log10(minimum), math.log10(maximum), n, device=device, dtype=dtype)
    return torch.linspace(minimum, maximum, n, device=device, dtype=dtype)


def _random_unit(
    spatial_dim: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.randn(spatial_dim, generator=generator, dtype=dtype).to(device=device)
    return value / torch.linalg.norm(value).clamp_min(1.0e-12)


def _normalized_weights(strata: Mapping[str, float]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not strata:
        raise ValueError("StratifiedGeometryGenerator requires at least one stratum")
    names = tuple(str(name) for name in strata)
    weights = tuple(max(0.0, float(strata[name])) for name in strata)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("strata weights must have positive total mass")
    return names, tuple(weight / total for weight in weights)


def _sample_geometry_for_stratum(
    stratum: str,
    bounds: Mapping[str, object],
    *,
    generator: torch.Generator,
) -> tuple[float, float]:
    default = bounds.get("default", {}) if isinstance(bounds.get("default", {}), Mapping) else {}
    config = bounds.get(stratum, default)
    if not isinstance(config, Mapping):
        config = default
    r12_min, r12_max = _range(config, default, "r12", fallback=(0.2, 3.0))
    radius_min, radius_max = _range(config, default, "radius", fallback=(0.0, 4.0))
    r12 = _uniform(r12_min, r12_max, generator=generator)
    radius = _uniform(radius_min, radius_max, generator=generator)
    return r12, radius


def _range(config: Mapping[str, object], default: Mapping[str, object], name: str, *, fallback: tuple[float, float]) -> tuple[float, float]:
    minimum = config.get(f"{name}_min", default.get(f"{name}_min", fallback[0]))
    maximum = config.get(f"{name}_max", default.get(f"{name}_max", fallback[1]))
    return float(minimum), float(maximum)


def _uniform(minimum: float, maximum: float, *, generator: torch.Generator) -> float:
    value = torch.rand((), generator=generator, dtype=torch.float64).item()
    return float(minimum + (maximum - minimum) * value)


def _dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    dtype = getattr(torch, str(value))
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype {value!r}")
    return dtype


__all__ = [
    "CuspGridGenerator",
    "HookeOrbitalGenerator",
    "StratifiedGeometryGenerator",
    "TailGridGenerator",
]
