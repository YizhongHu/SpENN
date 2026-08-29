"""Geometry helpers for electron batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tpen.data.batch.electron_batch import ElectronBatch


@dataclass(frozen=True)
class TwoElectronAtomicGeometry:
    """Permutation-invariant geometry for two electrons and one nucleus.

    Every field has shape ``[batch]`` after sample axes are flattened.  The
    angular coordinate is ``cos(theta_12)``.  It is non-finite, with
    ``angle_defined=False``, when an electron lies exactly on the nucleus
    because its direction is undefined.

    Parameters
    ----------
    minimum_electron_nuclear_radius : torch.Tensor
        ``min(r_1N, r_2N)`` for each configuration.
    electron_electron_distance : torch.Tensor
        Inter-electron distance ``r_12``.
    maximum_electron_nuclear_radius : torch.Tensor
        ``max(r_1N, r_2N)`` for each configuration.
    hyperradius : torch.Tensor
        ``sqrt(r_1N**2 + r_2N**2)``.
    cos_theta12 : torch.Tensor
        Cosine of the electron-nucleus-electron angle.
    angle_defined : torch.Tensor
        Boolean domain mask; false at coalescence or for non-finite geometry.
    angle_undefined_at_coalescence : torch.Tensor
        Boolean mask distinguishing finite electron-nucleus coalescence from
        other undefined cases such as non-finite input coordinates.
    """

    minimum_electron_nuclear_radius: torch.Tensor
    electron_electron_distance: torch.Tensor
    maximum_electron_nuclear_radius: torch.Tensor
    hyperradius: torch.Tensor
    cos_theta12: torch.Tensor
    angle_defined: torch.Tensor
    angle_undefined_at_coalescence: torch.Tensor

    def validate(self) -> "TwoElectronAtomicGeometry":
        """Validate the shared one-dimensional batch contract."""

        fields = (
            self.minimum_electron_nuclear_radius,
            self.electron_electron_distance,
            self.maximum_electron_nuclear_radius,
            self.hyperradius,
            self.cos_theta12,
        )
        expected = fields[0].shape
        if len(expected) != 1:
            raise ValueError("two-electron atomic geometry fields must have shape [batch]")
        if any(value.shape != expected for value in fields[1:]):
            raise ValueError("two-electron atomic geometry fields must share one batch shape")
        if any(value.device != fields[0].device for value in fields[1:]):
            raise ValueError("two-electron atomic geometry fields must share one device")
        if any(value.dtype != fields[0].dtype for value in fields[1:]):
            raise ValueError("two-electron atomic geometry fields must share one dtype")
        if self.angle_defined.shape != expected or self.angle_defined.device != fields[0].device:
            raise ValueError("two-electron angle domain status must match the batch shape/device")
        if self.angle_defined.dtype != torch.bool:
            raise ValueError("two-electron angle domain status must be boolean")
        if not torch.equal(self.angle_defined, torch.isfinite(self.cos_theta12)):
            raise ValueError("angle_defined must equal isfinite(cos_theta12)")
        if (
            self.angle_undefined_at_coalescence.shape != expected
            or self.angle_undefined_at_coalescence.device != fields[0].device
            or self.angle_undefined_at_coalescence.dtype != torch.bool
        ):
            raise ValueError("coalescence angle-domain status must match the batch contract")
        if torch.any(self.angle_defined & self.angle_undefined_at_coalescence):
            raise ValueError("a defined angle cannot also be undefined at coalescence")
        return self


def pairwise_displacements(positions: torch.Tensor) -> torch.Tensor:
    """Return pairwise displacement vectors ``r_i - r_j``.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor with shape ``[batch, n_electrons, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_electrons, spatial_dim]``.
    """

    _validate_positions(positions)
    return positions.unsqueeze(2) - positions.unsqueeze(1)


def pairwise_distances(positions: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return pairwise distances with a differentiable numerical floor.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor with shape ``[batch, n_electrons, spatial_dim]``.
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent with cusp factors and Coulomb
        potentials, yielding a hybrid Hamiltonian: a clipped potential
        evaluated with the boundary condition of an unclipped Coulomb
        potential. Inspect and validate before use; finite-eps
        electron-electron behaviour is UNMEASURED.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_electrons, 1]``.
    """

    displacement = pairwise_displacements(positions)
    squared = displacement.square().sum(dim=-1, keepdim=True)
    if eps:
        squared = squared + float(eps) ** 2
    return squared.sqrt()


def electron_nuclear_displacements(
    batch: ElectronBatch,
    nuclear_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return electron-nuclear displacement vectors ``r_i - R_A``.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions and nuclear coordinates. Nuclear
        coordinates may be shared across all samples or sampled with the same
        leading shape as the electron positions.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates overriding `batch.nuclear_positions`, with shape
        ``[n_nuclei, spatial_dim]`` or ``[batch, n_nuclei, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_nuclei, spatial_dim]`` after
        flattening sample axes.

    Raises
    ------
    ValueError
        If nuclear positions are absent.
    """

    flat = batch.flatten_samples()
    nuclei = nuclear_positions if nuclear_positions is not None else flat.nuclear_positions
    if nuclei is None:
        raise ValueError("electron-nuclear displacements require nuclear positions")
    nuclei = nuclei.to(device=flat.device, dtype=flat.dtype)
    if nuclei.ndim == 2:
        if nuclei.shape[-1] != flat.spatial_dim:
            raise ValueError("nuclear positions must match batch spatial dimension")
        return flat.positions.unsqueeze(-2) - nuclei.reshape(1, 1, *nuclei.shape)
    if nuclei.ndim != 3:
        raise ValueError("nuclear positions must have shape [n_nuclei, dim] or [batch, n_nuclei, dim]")
    if nuclei.shape[0] != flat.batch_size or nuclei.shape[-1] != flat.spatial_dim:
        raise ValueError("batched nuclear positions must match batch size and spatial dimension")
    return flat.positions.unsqueeze(-2) - nuclei.unsqueeze(-3)


def electron_nuclear_distances(
    batch: ElectronBatch,
    eps: float = 0.0,
    nuclear_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return electron-nuclear distances with a numerical floor.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions and nuclear coordinates.
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent with the cusp factor and Coulomb
        potential, yielding a hybrid Hamiltonian: a clipped potential evaluated
        with the boundary condition of an unclipped Coulomb potential. Inspect
        and validate before use; finite-eps electron-electron behaviour is
        UNMEASURED.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates overriding `batch.nuclear_positions`.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons, n_nuclei]`` after flattening
        sample axes.

    Raises
    ------
    ValueError
        If nuclear positions are absent.
    """

    return electron_nuclear_displacements(batch, nuclear_positions=nuclear_positions).norm(dim=-1).clamp_min(eps)


def nuclear_potential(batch: ElectronBatch, eps: float = 0.0) -> torch.Tensor:
    """Return ``sum_A Z_A / |r_i - R_A|`` for each electron.

    Parameters
    ----------
    batch : ElectronBatch
        Electron batch containing positions, nuclear coordinates, and nuclear
        charges. Nuclear data may be shared or sampled with the electron
        positions.
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent: the cusp factor and Coulomb
        potential do not necessarily apply the same floor, yielding a hybrid
        Hamiltonian: a clipped potential evaluated with the boundary condition
        of an unclipped Coulomb potential. Inspect and validate before use.
        Measurement found
        ``E(r; eps) = Z/r - Z/eps - Z^2/2`` for ``0 < r < eps`` across three
        eps scales and three directions, with normalized error <= ``1.11e-16``;
        electron-electron finite-eps behaviour is UNMEASURED.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons]`` after flattening sample
        axes.

    Raises
    ------
    ValueError
        If nuclear positions or nuclear charges are absent.
    """

    if batch.nuclear_charges is None:
        raise ValueError("nuclear potential requires nuclear charges")
    flat = batch.flatten_samples()
    distances = electron_nuclear_distances(flat, eps=eps)
    charges = flat.nuclear_charges
    if charges is None:
        raise ValueError("nuclear potential requires nuclear charges")
    if charges.ndim == 1:
        charge_view = charges.reshape(1, 1, charges.shape[-1])
    else:
        charge_view = charges.unsqueeze(-2)
    return (charge_view / distances).sum(dim=-1)


def two_electron_atomic_geometry(batch: ElectronBatch) -> TwoElectronAtomicGeometry:
    """Return the invariant geometry of a two-electron, one-nucleus batch.

    This is the geometry contract used by range-conditioned helium
    diagnostics.  It deliberately accepts an :class:`ElectronBatch` rather
    than probing an arbitrary coordinate container, and it fails when the
    electron or nucleus cardinality does not define ``r_12`` and
    ``theta_12`` unambiguously.  This vectorized implementation has a slow
    readable oracle in :func:`two_electron_atomic_geometry_reference`.

    Parameters
    ----------
    batch : ElectronBatch
        Batch containing exactly two electrons and one typed nuclear position.

    Returns
    -------
    TwoElectronAtomicGeometry
        Invariant radii, separation, hyperradius, and angle cosine with shape
        ``[batch]``.

    Raises
    ------
    ValueError
        If the batch does not contain exactly two electrons and one nucleus.
    """

    flat = batch.flatten_samples()
    if flat.n_electrons != 2:
        raise ValueError(
            "two-electron atomic geometry requires exactly two electrons, "
            f"got {flat.n_electrons}"
        )
    displacements = electron_nuclear_displacements(flat)
    if displacements.shape[2] != 1:
        raise ValueError(
            "two-electron atomic geometry requires exactly one nucleus, "
            f"got {displacements.shape[2]}"
        )

    vectors = displacements[:, :, 0, :]
    radii = vectors.norm(dim=-1)
    minimum_radius = radii.min(dim=1).values
    maximum_radius = radii.max(dim=1).values
    electron_distance = (vectors[:, 0] - vectors[:, 1]).norm(dim=-1)
    hyperradius = radii.square().sum(dim=1).sqrt()

    radius_product = radii[:, 0] * radii[:, 1]
    dot_product = (vectors[:, 0] * vectors[:, 1]).sum(dim=-1)
    finite_geometry = torch.isfinite(radii).all(dim=1) & torch.isfinite(dot_product)
    angle_undefined_at_coalescence = finite_geometry & (radii == 0).any(dim=1)
    angle_defined = finite_geometry & (radius_product > 0)
    cosine = dot_product / radius_product
    # A zero radius has no direction.  Keep an explicit domain status and a
    # non-finite value rather than making coalescence look collinear.
    cosine = torch.where(
        angle_defined,
        cosine,
        torch.full_like(cosine, float("nan")),
    )
    return TwoElectronAtomicGeometry(
        minimum_electron_nuclear_radius=minimum_radius,
        electron_electron_distance=electron_distance,
        maximum_electron_nuclear_radius=maximum_radius,
        hyperradius=hyperradius,
        cos_theta12=cosine,
        angle_defined=angle_defined,
        angle_undefined_at_coalescence=angle_undefined_at_coalescence,
    ).validate()


def two_electron_atomic_geometry_reference(batch: ElectronBatch) -> TwoElectronAtomicGeometry:
    """Return the slow row-wise definition of two-electron atomic geometry.

    The production helper is vectorized.  This reference intentionally spells
    out one configuration at a time so tests can compare the optimized tensor
    expressions against an independently readable definition.

    Parameters
    ----------
    batch : ElectronBatch
        Batch containing exactly two electrons and one typed nuclear position.

    Returns
    -------
    TwoElectronAtomicGeometry
        Row-wise reference geometry with the same semantic fields as
        :func:`two_electron_atomic_geometry`.
    """

    flat = batch.flatten_samples()
    if flat.n_electrons != 2:
        raise ValueError(
            "two-electron atomic geometry requires exactly two electrons, "
            f"got {flat.n_electrons}"
        )
    displacements = electron_nuclear_displacements(flat)
    if displacements.shape[2] != 1:
        raise ValueError(
            "two-electron atomic geometry requires exactly one nucleus, "
            f"got {displacements.shape[2]}"
        )

    minimum_radii: list[torch.Tensor] = []
    electron_distances: list[torch.Tensor] = []
    maximum_radii: list[torch.Tensor] = []
    hyperradii: list[torch.Tensor] = []
    cosines: list[torch.Tensor] = []
    defined: list[torch.Tensor] = []
    coalescence: list[torch.Tensor] = []
    for row in range(flat.batch_size):
        first = displacements[row, 0, 0]
        second = displacements[row, 1, 0]
        first_radius = first.norm()
        second_radius = second.norm()
        radius_product = first_radius * second_radius
        dot_product = first.dot(second)
        finite_geometry = (
            torch.isfinite(first_radius)
            & torch.isfinite(second_radius)
            & torch.isfinite(dot_product)
        )
        is_coalescence = finite_geometry & (
            (first_radius == 0) | (second_radius == 0)
        )
        is_defined = finite_geometry & (radius_product > 0)
        cosine = torch.where(
            is_defined,
            dot_product / radius_product,
            torch.full_like(radius_product, float("nan")),
        )
        minimum_radii.append(torch.minimum(first_radius, second_radius))
        electron_distances.append((first - second).norm())
        maximum_radii.append(torch.maximum(first_radius, second_radius))
        hyperradii.append((first_radius.square() + second_radius.square()).sqrt())
        cosines.append(cosine)
        defined.append(is_defined)
        coalescence.append(is_coalescence)
    return TwoElectronAtomicGeometry(
        minimum_electron_nuclear_radius=torch.stack(minimum_radii),
        electron_electron_distance=torch.stack(electron_distances),
        maximum_electron_nuclear_radius=torch.stack(maximum_radii),
        hyperradius=torch.stack(hyperradii),
        cos_theta12=torch.stack(cosines),
        angle_defined=torch.stack(defined),
        angle_undefined_at_coalescence=torch.stack(coalescence),
    ).validate()


def _validate_positions(positions: torch.Tensor) -> None:
    if positions.ndim != 3:
        raise ValueError(
            "positions must have shape [batch, n_electrons, spatial_dim], "
            f"got {tuple(positions.shape)}"
        )


__all__ = [
    "TwoElectronAtomicGeometry",
    "electron_nuclear_displacements",
    "electron_nuclear_distances",
    "nuclear_potential",
    "pairwise_displacements",
    "pairwise_distances",
    "two_electron_atomic_geometry",
    "two_electron_atomic_geometry_reference",
]
