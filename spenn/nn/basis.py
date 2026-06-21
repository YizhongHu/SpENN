"""Model-side equivariant input bases for electron configurations.

An :class:`ElectronBasis` is the first equivariant map in the SpENN model
pipeline. It featurizes a raw :class:`ElectronBatch` into a typed
:class:`ElectronBasisFeatures` object consumed by the embedding::

    ElectronBatch -> ElectronBasis -> ElectronBasisFeatures -> Embedding -> ...

The basis only computes per-particle (and optional per-pair) features; it does
not control feature scale. Feature-scale control lives in
:mod:`spenn.nn.normalization`. Each concrete basis owns its own physics
hyperparameters and exposes ``out_features`` so the embedding input width can be
derived from the selected basis (see :func:`spenn.config.register_resolvers`).

The active particle-permutation convention follows
:mod:`spenn.data.equivariant_state`:
``(pi x)[i] = x[pi^{-1} i]``. Every basis here featurizes each electron
independently of the others, so the maps are particle-permutation equivariant by
construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from spenn.data.batch import ElectronBatch
from spenn.data.equivariant_state import JsonScalar, compare_tensor_blocks
from spenn.data.indices import permute_particle_axis
from spenn.data.permutation import Permutation
from spenn.dependencies import require_torch
from spenn.equivariance import EquivariantMap

torch = require_torch(feature="SpENN basis modules")


@dataclass(frozen=True)
class ElectronBasisFeatures:
    """Typed output of an :class:`ElectronBasis`.

    This is intentionally minimal: it holds only what the embedding consumes.
    It is **not** an :class:`ElectronBatch`; the raw physical configuration is
    kept separate so the readout and envelope still see true coordinates.

    Parameters
    ----------
    one_body : torch.Tensor
        Per-particle feature vectors with shape ``[*sample_shape, n_electrons,
        features]``. This vector replaces the raw coordinate vector as the
        per-particle input to the embedding.
    pair : torch.Tensor or None, optional
        Optional per-pair feature tensor with shape ``[*sample_shape,
        n_electrons, n_electrons, pair_features]``. Reserved for future
        pair-feature augmentation; unused by the current embedding.
    metadata : mapping of str to JSON scalar, optional
        Free-form provenance describing how the features were produced.
    """

    one_body: torch.Tensor
    pair: torch.Tensor | None = None
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    @property
    def n_electrons(self) -> int:
        """Return the electron count read from the one-body axis."""

        return int(self.one_body.shape[-2])

    @property
    def n_features(self) -> int:
        """Return the per-particle one-body feature width."""

        return int(self.one_body.shape[-1])

    def permute(self, permutation: Permutation) -> "ElectronBasisFeatures":
        """Return a copy transformed by an active particle permutation."""

        one_body = permute_particle_axis(self.one_body, permutation, axis=-2)
        pair = self.pair
        if pair is not None:
            # Both electron axes of the pair tensor permute together.
            pair = permute_particle_axis(pair, permutation, axis=-3)
            pair = permute_particle_axis(pair, permutation, axis=-2)
        return type(self)(one_body=one_body, pair=pair, metadata=dict(self.metadata))

    def compare(
        self,
        other: "ElectronBasisFeatures",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, JsonScalar]]:
        """Compare one-body and pair tensors; return ``(is_close, metrics)``."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        left = [self.one_body] if self.pair is None else [self.one_body, self.pair]
        right = [other.one_body] if other.pair is None else [other.one_body, other.pair]
        return compare_tensor_blocks(left, right, atol=atol, rtol=rtol)


class ElectronBasis(EquivariantMap):
    """Equivariant featurization map from electron configurations to features.

    Concrete subclasses implement :meth:`forward_impl`, returning an
    :class:`ElectronBasisFeatures`. The base class records the per-particle
    feature width as ``out_features`` so downstream modules (and config
    resolvers) can size the embedding input.

    Parameters
    ----------
    spatial_dim : int
        Coordinate dimension of each electron.
    include_spin : bool, optional
        If ``True``, append the electron spin as a final one-body channel.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, *, spatial_dim: int, include_spin: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if spatial_dim <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        self.spatial_dim = int(spatial_dim)
        self.include_spin = bool(include_spin)

    @property
    def coordinate_features(self) -> int:
        """Return the number of one-body channels before the optional spin."""

        raise NotImplementedError(f"{type(self).__name__}.coordinate_features is not implemented")

    @property
    def out_features(self) -> int:
        """Return the total per-particle one-body feature width."""

        return self.coordinate_features + (1 if self.include_spin else 0)

    def _one_body(self, coordinate_features: torch.Tensor, batch: ElectronBatch) -> torch.Tensor:
        """Append the optional spin channel to coordinate features."""

        if not self.include_spin:
            return coordinate_features
        if batch.spins is None:
            raise ValueError(f"{type(self).__name__} include_spin=True requires ElectronBatch.spins")
        spins = batch.spins.unsqueeze(-1).to(device=coordinate_features.device, dtype=coordinate_features.dtype)
        return torch.cat([coordinate_features, spins], dim=-1)

    def _features(self, coordinate_features: torch.Tensor, batch: ElectronBatch, *, name: str) -> ElectronBasisFeatures:
        """Assemble the typed features and record provenance metadata."""

        one_body = self._one_body(coordinate_features, batch)
        metadata: dict[str, JsonScalar] = {
            "basis": name,
            "spatial_dim": self.spatial_dim,
            "include_spin": self.include_spin,
            "out_features": self.out_features,
        }
        features = ElectronBasisFeatures(one_body=one_body, metadata=metadata)
        self.trace("features", features)
        return features


class RawCoordinateBasis(ElectronBasis):
    """Compatibility baseline that passes raw coordinates through unchanged.

    With ``include_spin=True`` the per-particle vector is ``(r_i, s_i)``, which
    reproduces the historical embedding input. In the pair-stability scan raw
    coordinates are used only with a Gaussian envelope.

    Parameters
    ----------
    spatial_dim : int
        Coordinate dimension of each electron.
    include_spin : bool, optional
        If ``True``, append spin as a final one-body channel.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    @property
    def coordinate_features(self) -> int:
        """Return the raw coordinate width."""

        return self.spatial_dim

    def forward_impl(self, batch: ElectronBatch) -> ElectronBasisFeatures:
        """Return raw coordinates (and optional spin) as one-body features."""

        if batch.spatial_dim != self.spatial_dim:
            raise ValueError(
                f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with "
                f"{type(self).__name__} spatial_dim={self.spatial_dim}"
            )
        return self._features(batch.positions, batch, name="raw")


class HookeHermiteBasis(ElectronBasis):
    """Hermite / oscillator-polynomial features without the Gaussian factor.

    For each electron and each spatial component ``x`` the basis evaluates the
    physicists' Hermite polynomials ``H_0(xi), ..., H_{max_order}(xi)`` of the
    scaled coordinate ``xi = x / length_scale``. This is the clean polynomial
    match for models that already apply an output Gaussian envelope, so the
    decay is supplied once on the output side rather than baked into the inputs.

    Parameters
    ----------
    omega : float
        Oscillator frequency. The default oscillator length is ``1 / sqrt(omega)``.
    max_order : int
        Highest Hermite polynomial order. ``max_order + 1`` polynomials are
        produced per spatial component.
    length_scale : float or None, optional
        Coordinate scale ``L`` in ``xi = x / L``. If ``None``, the oscillator
        length ``1 / sqrt(omega)`` is used.
    spatial_dim : int
        Coordinate dimension of each electron.
    include_spin : bool, optional
        If ``True``, append spin as a final one-body channel.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        omega: float,
        max_order: int,
        length_scale: float | None = None,
        spatial_dim: int,
        include_spin: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(spatial_dim=spatial_dim, include_spin=include_spin, **kwargs)
        if omega <= 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        if max_order < 0:
            raise ValueError(f"max_order must be nonnegative, got {max_order}")
        self.omega = float(omega)
        self.max_order = int(max_order)
        self.length_scale = float(length_scale) if length_scale is not None else self.omega ** -0.5
        if self.length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {self.length_scale}")

    @property
    def coordinate_features(self) -> int:
        """Return ``spatial_dim * (max_order + 1)`` polynomial channels."""

        return self.spatial_dim * (self.max_order + 1)

    def forward_impl(self, batch: ElectronBatch) -> ElectronBasisFeatures:
        """Return Hermite polynomial features (no Gaussian factor)."""

        if batch.spatial_dim != self.spatial_dim:
            raise ValueError(
                f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with "
                f"{type(self).__name__} spatial_dim={self.spatial_dim}"
            )
        features = _hermite_features(
            batch.positions,
            max_order=self.max_order,
            length_scale=self.length_scale,
            gaussian=False,
        )
        return self._features(features, batch, name="hermite")


class HookeOrbitalBasis(ElectronBasis):
    """Hooke / quantum-harmonic-oscillator orbital-shaped features.

    For each electron and each spatial component ``x`` the basis evaluates
    ``H_n(xi) * g(xi)`` for ``n = 0, ..., max_shell`` where ``xi = sqrt(omega) x``
    and ``g(xi) = exp(-xi^2 / 2)`` when ``include_gaussian_factor`` is set. With
    the factor enabled these are the 1D oscillator eigenfunction shapes.

    Even though this basis already carries oscillator-decay structure, the
    pair-stability scan still applies the output Gaussian envelope to every main
    variant so the asymptotic output prior is consistent across architecture
    choices. This may be mildly double-normalized for orbital inputs, but it is
    intentional for a common-envelope comparison.

    Parameters
    ----------
    omega : float
        Oscillator frequency setting the scaled coordinate ``xi = sqrt(omega) x``.
    max_shell : int
        Highest oscillator shell index. ``max_shell + 1`` functions are produced
        per spatial component.
    include_gaussian_factor : bool, optional
        If ``True``, multiply each polynomial by ``exp(-xi^2 / 2)``.
    spatial_dim : int
        Coordinate dimension of each electron.
    include_spin : bool, optional
        If ``True``, append spin as a final one-body channel.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        omega: float,
        max_shell: int,
        include_gaussian_factor: bool = True,
        spatial_dim: int,
        include_spin: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(spatial_dim=spatial_dim, include_spin=include_spin, **kwargs)
        if omega <= 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        if max_shell < 0:
            raise ValueError(f"max_shell must be nonnegative, got {max_shell}")
        self.omega = float(omega)
        self.max_shell = int(max_shell)
        self.include_gaussian_factor = bool(include_gaussian_factor)
        # xi = sqrt(omega) * x, i.e. an oscillator length of 1 / sqrt(omega).
        self.length_scale = self.omega ** -0.5

    @property
    def coordinate_features(self) -> int:
        """Return ``spatial_dim * (max_shell + 1)`` orbital channels."""

        return self.spatial_dim * (self.max_shell + 1)

    def forward_impl(self, batch: ElectronBatch) -> ElectronBasisFeatures:
        """Return oscillator orbital-shaped features."""

        if batch.spatial_dim != self.spatial_dim:
            raise ValueError(
                f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with "
                f"{type(self).__name__} spatial_dim={self.spatial_dim}"
            )
        features = _hermite_features(
            batch.positions,
            max_order=self.max_shell,
            length_scale=self.length_scale,
            gaussian=self.include_gaussian_factor,
        )
        return self._features(features, batch, name="orbital")


def _hermite_features(
    positions: torch.Tensor,
    *,
    max_order: int,
    length_scale: float,
    gaussian: bool,
) -> torch.Tensor:
    """Return per-particle Hermite features flattened over component and order.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape ``[*sample_shape, n_electrons,
        spatial_dim]``.
    max_order : int
        Highest physicists' Hermite polynomial order.
    length_scale : float
        Coordinate scale ``L`` in ``xi = x / L``.
    gaussian : bool
        If ``True``, multiply each polynomial by ``exp(-xi^2 / 2)``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[*sample_shape, n_electrons, spatial_dim *
        (max_order + 1)]``. Polynomials vary fastest within each spatial
        component.
    """

    xi = positions / length_scale
    # Physicists' Hermite recurrence: H_0 = 1, H_1 = 2 xi, H_{n+1} = 2 xi H_n - 2 n H_{n-1}.
    polynomials = [torch.ones_like(xi)]
    if max_order >= 1:
        polynomials.append(2.0 * xi)
    for order in range(1, max_order):
        polynomials.append(2.0 * xi * polynomials[order] - 2.0 * order * polynomials[order - 1])
    stacked = torch.stack(polynomials, dim=-1)  # [*sample, n, spatial_dim, max_order + 1]
    if gaussian:
        stacked = stacked * torch.exp(-0.5 * xi.square()).unsqueeze(-1)
    sample_shape = positions.shape[:-1]
    return stacked.reshape(*sample_shape, positions.shape[-1] * (max_order + 1))


__all__ = [
    "ElectronBasis",
    "ElectronBasisFeatures",
    "HookeHermiteBasis",
    "HookeOrbitalBasis",
    "RawCoordinateBasis",
]
