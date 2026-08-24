"""Deterministic helium singular-limit and tail atlas geometries."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext


@dataclass(frozen=True)
class _AtlasRow:
    """One generated configuration and its explicit path metadata."""

    positions: torch.Tensor
    tangent: torch.Tensor
    requested_coordinate: torch.Tensor
    realized_coordinate: torch.Tensor
    sample_kind: str
    direction_id: int
    ray_id: int
    refinement_index: int
    probe_electron: int
    geometry_kind: str
    coordinate_kind: str
    is_coordinate_representability_boundary: bool = False
    is_exact_zero_sentinel: bool = False


class _HeliumAtlasGenerator:
    """Shared validation and batch assembly for helium atlas generators."""

    def __init__(
        self,
        *,
        atoms: AtomicConfiguration,
        directions: Sequence[Sequence[float]] | torch.Tensor,
    ) -> None:
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(
                f"{type(self).__name__} requires an AtomicConfiguration, "
                f"got {type(atoms).__name__}"
            )
        atoms.validate()
        if tuple(atoms.positions.shape) != (1, 3):
            raise ValueError(f"{type(self).__name__} requires one three-dimensional nucleus")
        if not torch.equal(
            atoms.charges.detach().cpu().to(dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        ):
            raise ValueError(f"{type(self).__name__} requires one Z=2 helium nucleus")
        self.atoms = atoms
        self.directions = _validated_directions(directions, owner=type(self).__name__)

    def _materialize(
        self,
        rows: Sequence[_AtlasRow],
        *,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        if not rows:
            raise ValueError(f"{type(self).__name__} generated no atlas rows")
        device = context.device or torch.device("cpu")
        dtype = context.dtype or torch.float64
        atoms = self.atoms.to(device=device, dtype=dtype)
        positions = torch.stack([row.positions for row in rows]).to(device=device, dtype=dtype)
        tangents = torch.stack([row.tangent for row in rows]).to(device=device, dtype=dtype)
        spins = torch.tensor([1.0, -1.0], device=device, dtype=dtype).expand(len(rows), -1).clone()
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=atoms.positions,
            nuclear_charges=atoms.charges,
            atomic_configuration=atoms,
            spins=spins,
            aux={},
        ).validate()
        return GeneratedConfigurations(
            batch=batch,
            metadata={
                "atlas_coordinate_kind": tuple(row.coordinate_kind for row in rows),
                "atlas_geometry_kind": tuple(row.geometry_kind for row in rows),
                "atlas_sample_kind": tuple(row.sample_kind for row in rows),
                "requested_coordinate": torch.stack(
                    [row.requested_coordinate for row in rows]
                ).to(device=device, dtype=dtype),
                "generated_realized_coordinate": torch.stack(
                    [row.realized_coordinate for row in rows]
                ).to(device=device, dtype=dtype),
                "coordinate_tangent": tangents,
                "direction_id": torch.tensor(
                    [row.direction_id for row in rows], device=device, dtype=torch.long
                ),
                "ray_id": torch.tensor([row.ray_id for row in rows], device=device, dtype=torch.long),
                "refinement_index": torch.tensor(
                    [row.refinement_index for row in rows], device=device, dtype=torch.long
                ),
                "probe_electron": torch.tensor(
                    [row.probe_electron for row in rows], device=device, dtype=torch.long
                ),
                "is_coordinate_representability_boundary": torch.tensor(
                    [row.is_coordinate_representability_boundary for row in rows],
                    device=device,
                    dtype=torch.bool,
                ),
                "is_exact_zero_sentinel": torch.tensor(
                    [row.is_exact_zero_sentinel for row in rows],
                    device=device,
                    dtype=torch.bool,
                ),
                "atlas_seed": int(context.seed),
                "atlas_boundary_dtype": "float64",
                "atlas_boundary_device": "cpu",
                "atlas_evaluation_dtype": str(batch.dtype).removeprefix("torch."),
                "atlas_evaluation_device": str(batch.device),
            },
        )


class HeliumElectronNucleusApproachGenerator(_HeliumAtlasGenerator):
    """Refine electron-nucleus rays to their numerical representation boundary.

    Parameters
    ----------
    atoms : AtomicConfiguration
        Fixed helium nuclear geometry.
    directions : sequence
        Explicit nonzero three-dimensional ray directions.
    spectator_position : sequence
        Explicit absolute position of the non-probed electron.
    start_radius : float
        First strictly positive requested radius.
    refinement_ratio : float
        Geometric ratio in ``(0, 1)``.
    probe_electrons : sequence of int, optional
        Electron labels to move along every configured ray.
    max_refinement_steps : int, optional
        Fail-closed bound; exhaustion before a numerical boundary is an error.
    """

    name = "helium_electron_nucleus_approach"

    def __init__(
        self,
        *,
        atoms: AtomicConfiguration,
        directions: Sequence[Sequence[float]] | torch.Tensor,
        spectator_position: Sequence[float] | torch.Tensor,
        start_radius: float,
        refinement_ratio: float,
        probe_electrons: Sequence[int] = (0, 1),
        max_refinement_steps: int = 4096,
    ) -> None:
        super().__init__(atoms=atoms, directions=directions)
        self.spectator_position = _validated_vector(
            spectator_position, name="spectator_position", owner=type(self).__name__
        )
        self.start_radius, self.refinement_ratio, self.max_refinement_steps = (
            _validated_refinement(
                start_radius=start_radius,
                refinement_ratio=refinement_ratio,
                max_refinement_steps=max_refinement_steps,
                owner=type(self).__name__,
            )
        )
        probes = tuple(int(value) for value in probe_electrons)
        if not probes or len(set(probes)) != len(probes) or any(value not in (0, 1) for value in probes):
            raise ValueError(
                "HeliumElectronNucleusApproachGenerator probe_electrons must be a "
                "non-empty unique subset of (0, 1)"
            )
        self.probe_electrons = probes

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return spectator-held e-n rays through the numerical boundary."""

        del model
        # Numerical boundaries are generated on the provenance-pinned CPU
        # float64 reference, then materialized on the evaluation device.
        device = torch.device("cpu")
        dtype = torch.float64
        atoms = self.atoms.to(device=device, dtype=dtype)
        nucleus = atoms.positions[0]
        spectator = self.spectator_position.to(device=device, dtype=dtype)
        directions = self.directions.to(device=device, dtype=dtype)
        rows: list[_AtlasRow] = []
        ray_id = 0
        for direction_id, direction in enumerate(directions):
            for probe_electron in self.probe_electrons:

                def build(radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    positions = torch.empty((2, 3), device=device, dtype=dtype)
                    positions[probe_electron] = nucleus + radius * direction
                    positions[1 - probe_electron] = spectator
                    tangent = torch.zeros_like(positions)
                    tangent[probe_electron] = direction
                    realized = torch.linalg.vector_norm(positions[probe_electron] - nucleus)
                    return positions, tangent, realized

                rows.extend(
                    _refine_ray(
                        build=build,
                        start_radius=self.start_radius,
                        refinement_ratio=self.refinement_ratio,
                        max_refinement_steps=self.max_refinement_steps,
                        direction_id=direction_id,
                        ray_id=ray_id,
                        probe_electron=probe_electron,
                        geometry_kind="configured_spectator",
                        coordinate_kind="electron_nucleus_distance",
                        device=device,
                        dtype=dtype,
                    )
                )
                ray_id += 1
        return self._materialize(rows, context=context)


class HeliumElectronElectronApproachGenerator(_HeliumAtlasGenerator):
    """Refine e-e separation at an explicitly configured center of mass."""

    name = "helium_electron_electron_approach"

    def __init__(
        self,
        *,
        atoms: AtomicConfiguration,
        directions: Sequence[Sequence[float]] | torch.Tensor,
        center_of_mass: Sequence[float] | torch.Tensor,
        start_radius: float,
        refinement_ratio: float,
        max_refinement_steps: int = 4096,
    ) -> None:
        super().__init__(atoms=atoms, directions=directions)
        self.center_of_mass = _validated_vector(
            center_of_mass, name="center_of_mass", owner=type(self).__name__
        )
        self.start_radius, self.refinement_ratio, self.max_refinement_steps = (
            _validated_refinement(
                start_radius=start_radius,
                refinement_ratio=refinement_ratio,
                max_refinement_steps=max_refinement_steps,
                owner=type(self).__name__,
            )
        )

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return COM-held e-e rays through the numerical boundary."""

        del model
        device = torch.device("cpu")
        dtype = torch.float64
        center = self.center_of_mass.to(device=device, dtype=dtype)
        directions = self.directions.to(device=device, dtype=dtype)
        rows: list[_AtlasRow] = []
        for direction_id, direction in enumerate(directions):

            def build(radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                half = 0.5 * radius * direction
                positions = torch.stack((center + half, center - half))
                tangent = torch.stack((0.5 * direction, -0.5 * direction))
                realized = torch.linalg.vector_norm(positions[0] - positions[1])
                return positions, tangent, realized

            rows.extend(
                _refine_ray(
                    build=build,
                    start_radius=self.start_radius,
                    refinement_ratio=self.refinement_ratio,
                    max_refinement_steps=self.max_refinement_steps,
                    direction_id=direction_id,
                    ray_id=direction_id,
                    probe_electron=-1,
                    geometry_kind="configured_center_of_mass",
                    coordinate_kind="electron_electron_distance",
                    device=device,
                    dtype=dtype,
                )
            )
        return self._materialize(rows, context=context)


class HeliumOneElectronEscapeGenerator(_HeliumAtlasGenerator):
    """Move one electron along configured rays with an explicit spectator."""

    name = "helium_one_electron_escape"

    def __init__(
        self,
        *,
        atoms: AtomicConfiguration,
        directions: Sequence[Sequence[float]] | torch.Tensor,
        radii: Sequence[float],
        spectator_position: Sequence[float] | torch.Tensor,
        probe_electron: int = 0,
    ) -> None:
        super().__init__(atoms=atoms, directions=directions)
        self.radii = _validated_radii(radii, owner=type(self).__name__)
        self.spectator_position = _validated_vector(
            spectator_position, name="spectator_position", owner=type(self).__name__
        )
        self.probe_electron = int(probe_electron)
        if self.probe_electron not in (0, 1):
            raise ValueError("HeliumOneElectronEscapeGenerator probe_electron must be 0 or 1")

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return configured spectator-held escape geometries."""

        del model
        device = torch.device("cpu")
        dtype = torch.float64
        atoms = self.atoms.to(device=device, dtype=dtype)
        nucleus = atoms.positions[0]
        spectator = self.spectator_position.to(device=device, dtype=dtype)
        directions = self.directions.to(device=device, dtype=dtype)
        rows: list[_AtlasRow] = []
        for direction_id, direction in enumerate(directions):
            for index, radius_value in enumerate(self.radii):
                radius = torch.tensor(radius_value, device=device, dtype=dtype)
                positions = torch.empty((2, 3), device=device, dtype=dtype)
                positions[self.probe_electron] = nucleus + radius * direction
                positions[1 - self.probe_electron] = spectator
                tangent = torch.zeros_like(positions)
                tangent[self.probe_electron] = direction
                realized = torch.linalg.vector_norm(positions[self.probe_electron] - nucleus)
                rows.append(
                    _AtlasRow(
                        positions=positions,
                        tangent=tangent,
                        requested_coordinate=radius,
                        realized_coordinate=realized,
                        sample_kind="positive_nonzero",
                        direction_id=direction_id,
                        ray_id=direction_id,
                        refinement_index=index,
                        probe_electron=self.probe_electron,
                        geometry_kind="configured_spectator",
                        coordinate_kind="one_electron_escape_radius",
                    )
                )
        return self._materialize(rows, context=context)


class HeliumCenterOfMassEscapeGenerator(_HeliumAtlasGenerator):
    """Translate a fixed two-electron shape along configured COM rays."""

    name = "helium_center_of_mass_escape"

    def __init__(
        self,
        *,
        atoms: AtomicConfiguration,
        directions: Sequence[Sequence[float]] | torch.Tensor,
        radii: Sequence[float],
        relative_positions: Sequence[Sequence[float]] | torch.Tensor,
    ) -> None:
        super().__init__(atoms=atoms, directions=directions)
        self.radii = _validated_radii(radii, owner=type(self).__name__)
        relative = torch.as_tensor(relative_positions, dtype=torch.float64)
        if tuple(relative.shape) != (2, 3) or not torch.isfinite(relative).all():
            raise ValueError(
                "HeliumCenterOfMassEscapeGenerator relative_positions must be finite with shape (2, 3)"
            )
        if not torch.equal(relative.mean(dim=0), torch.zeros(3, dtype=relative.dtype)):
            raise ValueError(
                "HeliumCenterOfMassEscapeGenerator relative_positions must have exact zero center of mass"
            )
        self.relative_positions = relative

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return translated fixed-shape COM escape geometries."""

        del model
        device = torch.device("cpu")
        dtype = torch.float64
        nucleus = self.atoms.to(device=device, dtype=dtype).positions[0]
        relative = self.relative_positions.to(device=device, dtype=dtype)
        directions = self.directions.to(device=device, dtype=dtype)
        rows: list[_AtlasRow] = []
        for direction_id, direction in enumerate(directions):
            for index, radius_value in enumerate(self.radii):
                radius = torch.tensor(radius_value, device=device, dtype=dtype)
                center = nucleus + radius * direction
                positions = center.unsqueeze(0) + relative
                tangent = direction.expand_as(positions).clone()
                realized = torch.linalg.vector_norm(positions.mean(dim=0) - nucleus)
                rows.append(
                    _AtlasRow(
                        positions=positions,
                        tangent=tangent,
                        requested_coordinate=radius,
                        realized_coordinate=realized,
                        sample_kind="positive_nonzero",
                        direction_id=direction_id,
                        ray_id=direction_id,
                        refinement_index=index,
                        probe_electron=-1,
                        geometry_kind="configured_center_of_mass",
                        coordinate_kind="center_of_mass_escape_radius",
                    )
                )
        return self._materialize(rows, context=context)


class HeliumAngularShellGenerator(HeliumOneElectronEscapeGenerator):
    """Place one electron on configured angular directions and shell radii."""

    name = "helium_angular_shell"

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Return explicit angular-shell rows with spectator geometry."""

        generated = super().generate(model=model, context=context)
        metadata = dict(generated.metadata)
        metadata["atlas_coordinate_kind"] = tuple(
            "angular_shell_radius" for _ in range(generated.batch.batch_size)
        )
        metadata["atlas_geometry_kind"] = tuple(
            "configured_angular_shell_with_spectator"
            for _ in range(generated.batch.batch_size)
        )
        return GeneratedConfigurations(batch=generated.batch, metadata=metadata)


def _refine_ray(
    *,
    build: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    start_radius: float,
    refinement_ratio: float,
    max_refinement_steps: int,
    direction_id: int,
    ray_id: int,
    probe_electron: int,
    geometry_kind: str,
    coordinate_kind: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[_AtlasRow]:
    """Return a requested-nonzero ray, its realized boundary, and zero sentinel."""

    rows: list[_AtlasRow] = []
    requested = torch.tensor(start_radius, device=device, dtype=dtype)
    boundary_reached = False
    for index in range(max_refinement_steps):
        positions, tangent, realized = build(requested)
        if not torch.isfinite(realized) or realized <= 0:
            rows.append(
                _AtlasRow(
                    positions=positions,
                    tangent=tangent,
                    requested_coordinate=requested,
                    realized_coordinate=realized,
                    sample_kind="nonzero_requested_coordinate_representation_failure",
                    direction_id=direction_id,
                    ray_id=ray_id,
                    refinement_index=index,
                    probe_electron=probe_electron,
                    geometry_kind=geometry_kind,
                    coordinate_kind=coordinate_kind,
                    is_coordinate_representability_boundary=True,
                )
            )
            boundary_reached = True
            break
        rows.append(
            _AtlasRow(
                positions=positions,
                tangent=tangent,
                requested_coordinate=requested,
                realized_coordinate=realized,
                sample_kind="positive_nonzero",
                direction_id=direction_id,
                ray_id=ray_id,
                refinement_index=index,
                probe_electron=probe_electron,
                geometry_kind=geometry_kind,
                coordinate_kind=coordinate_kind,
            )
        )
        next_requested = requested * refinement_ratio
        requested = next_requested
    if not boundary_reached:
        raise ValueError(
            "geometric refinement exhausted max_refinement_steps before reaching a numerical boundary"
        )

    zero = torch.zeros((), device=device, dtype=dtype)
    positions, tangent, realized = build(zero)
    rows.append(
        _AtlasRow(
            positions=positions,
            tangent=tangent,
            requested_coordinate=zero,
            realized_coordinate=realized,
            sample_kind="exact_zero_sentinel",
            direction_id=direction_id,
            ray_id=ray_id,
            refinement_index=len(rows),
            probe_electron=probe_electron,
            geometry_kind=geometry_kind,
            coordinate_kind=coordinate_kind,
            is_exact_zero_sentinel=True,
        )
    )
    return rows


def _validated_directions(
    values: Sequence[Sequence[float]] | torch.Tensor,
    *,
    owner: str,
) -> torch.Tensor:
    directions = torch.as_tensor(values, dtype=torch.float64)
    if directions.ndim != 2 or directions.shape[1:] != (3,) or directions.shape[0] == 0:
        raise ValueError(f"{owner} directions must have shape (n_directions, 3)")
    if not torch.isfinite(directions).all():
        raise ValueError(f"{owner} directions must be finite")
    norms = torch.linalg.vector_norm(directions, dim=-1)
    if torch.any(norms <= 0):
        raise ValueError(f"{owner} directions must be nonzero")
    return directions / norms.unsqueeze(-1)


def _validated_vector(values: Sequence[float] | torch.Tensor, *, name: str, owner: str) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float64)
    if tuple(vector.shape) != (3,) or not torch.isfinite(vector).all():
        raise ValueError(f"{owner} {name} must be a finite three-dimensional vector")
    return vector


def _validated_refinement(
    *,
    start_radius: float,
    refinement_ratio: float,
    max_refinement_steps: int,
    owner: str,
) -> tuple[float, float, int]:
    start = float(start_radius)
    ratio = float(refinement_ratio)
    steps = int(max_refinement_steps)
    if not math.isfinite(start) or start <= 0:
        raise ValueError(f"{owner} start_radius must be finite and strictly positive")
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError(f"{owner} refinement_ratio must lie strictly between zero and one")
    if steps <= 0:
        raise ValueError(f"{owner} max_refinement_steps must be positive")
    return start, ratio, steps


def _validated_radii(values: Sequence[float], *, owner: str) -> tuple[float, ...]:
    radii = tuple(float(value) for value in values)
    if not radii:
        raise ValueError(f"{owner} requires at least one radius")
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError(f"{owner} radii must be finite and strictly positive")
    if any(right <= left for left, right in zip(radii, radii[1:])):
        raise ValueError(f"{owner} radii must be strictly increasing")
    return radii


__all__ = [
    "HeliumAngularShellGenerator",
    "HeliumCenterOfMassEscapeGenerator",
    "HeliumElectronElectronApproachGenerator",
    "HeliumElectronNucleusApproachGenerator",
    "HeliumOneElectronEscapeGenerator",
]
