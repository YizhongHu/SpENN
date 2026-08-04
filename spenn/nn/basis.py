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

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from spenn.data.batch import ElectronBatch
from spenn.data.equivariant_state import JsonScalar, compare_tensor_blocks
from spenn.data.indices import permute_particle_axis
from spenn.data.permutation import Permutation
from spenn.dependencies import require_torch
from spenn.equivariance import EquivariantMap

torch = require_torch(feature="SpENN basis modules")


HookeBasisSemantics = Literal["axiswise_v1", "product_v2"]
HookeProductTruncation = Literal["total_shell", "cartesian_box"]

# Changing this order changes the meaning of every product-basis channel. Keep
# the version in both the implementation and feature metadata explicit.
_MULTI_INDEX_ORDER_V1 = "shell-major-coordinate-priority-v1"


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

    def _features(
        self,
        coordinate_features: torch.Tensor,
        batch: ElectronBatch,
        *,
        name: str,
        provenance: Mapping[str, JsonScalar] | None = None,
    ) -> ElectronBasisFeatures:
        """Assemble the typed features and record provenance metadata."""

        one_body = self._one_body(coordinate_features, batch)
        metadata: dict[str, JsonScalar] = {
            "basis": name,
            "spatial_dim": self.spatial_dim,
            "include_spin": self.include_spin,
            "out_features": self.out_features,
        }
        if provenance is not None:
            metadata.update(provenance)
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
    """Versioned Hooke / harmonic-oscillator one-body basis.

    ``axiswise_v1`` is the historical coordinatewise feature map. It is
    deprecated but retained as an explicit mode (decisions D10/D11 in the
    TPEN migration): selecting it emits a :class:`FutureWarning` and
    keeps the frozen output contract, so it stays available as a smoke-run
    knob. An omitted ``basis_semantics`` selects ``product_v2`` — the flip is
    loud, not silent, because legacy ``max_shell`` arguments fail product-v2
    validation instead of producing different numbers.

    ``product_v2`` is the multidimensional orbital basis. Each spatial channel
    has a configured multi-index ``n`` and evaluates
    ``prod_k H_(n_k)(sqrt(omega) * x_k)``. Its truncation and Gaussian choice
    are explicit so channel meanings cannot be inferred from a shared width.

    Parameters
    ----------
    omega : float
        Oscillator frequency setting ``xi = sqrt(omega) * x``.
    spatial_dim : int
        Coordinate dimension of each electron.
    basis_semantics : {"axiswise_v1", "product_v2"} or None, optional
        Versioned channel contract. ``None`` selects ``product_v2``.
        ``"axiswise_v1"`` is deprecated (warns) but fully supported.
    max_shell : int or None, optional
        Legacy highest per-coordinate Hermite order. Valid only for
        ``axiswise_v1`` and required by that contract.
    truncation : {"total_shell", "cartesian_box"} or None, optional
        Product-v2 multi-index admission rule.
    max_total_shell : int or None, optional
        Product-v2 total-shell bound. Valid only with ``total_shell``.
    box_size : int or None, optional
        Product-v2 number of admitted orders per coordinate. Valid only with
        ``cartesian_box``.
    include_gaussian_factor : bool or None, optional
        Multiply channels by their oscillator Gaussian. Legacy V1 defaults to
        ``True`` for compatibility; V2 requires this choice explicitly.
    include_spin : bool, optional
        If ``True``, append spin as a final one-body channel.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        omega: float,
        spatial_dim: int,
        basis_semantics: str | None = None,
        max_shell: int | None = None,
        truncation: str | None = None,
        max_total_shell: int | None = None,
        box_size: int | None = None,
        include_gaussian_factor: bool | None = None,
        include_spin: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(spatial_dim=spatial_dim, include_spin=include_spin, **kwargs)
        if omega <= 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        self.omega = float(omega)
        # xi = sqrt(omega) * x, i.e. an oscillator length of 1 / sqrt(omega).
        self.length_scale = self.omega ** -0.5
        self._multi_indices: tuple[tuple[int, ...], ...] = ()
        self._product_max_order: int | None = None
        self._provenance: Mapping[str, JsonScalar] = {}

        if basis_semantics == "axiswise_v1":
            # Deprecated but retained as an explicit smoke knob (D10/D11).
            # FutureWarning, not DeprecationWarning (D15d): the latter is
            # suppressed by default outside __main__/pytest, so a production
            # smoke run selecting the legacy basis would print nothing.
            warnings.warn(
                "HookeOrbitalBasis basis_semantics='axiswise_v1' is deprecated; "
                "product_v2 is the default. axiswise_v1 remains fully supported "
                "as an explicit legacy mode with no silent fallback.",
                FutureWarning,
                stacklevel=2,
            )
            self._initialize_axiswise_v1(
                max_shell=max_shell,
                truncation=truncation,
                max_total_shell=max_total_shell,
                box_size=box_size,
                include_gaussian_factor=include_gaussian_factor,
            )
        elif basis_semantics is None or basis_semantics == "product_v2":
            self._initialize_product_v2(
                max_shell=max_shell,
                truncation=truncation,
                max_total_shell=max_total_shell,
                box_size=box_size,
                include_gaussian_factor=include_gaussian_factor,
            )
        else:
            raise ValueError(f"Unsupported Hooke basis_semantics {basis_semantics!r}")

    def _initialize_axiswise_v1(
        self,
        *,
        max_shell: int | None,
        truncation: str | None,
        max_total_shell: int | None,
        box_size: int | None,
        include_gaussian_factor: bool | None,
    ) -> None:
        """Initialize frozen axiswise V1 behavior for historical configs."""

        if truncation is not None or max_total_shell is not None or box_size is not None:
            raise ValueError("axiswise_v1 HookeOrbitalBasis does not accept product-v2 arguments")
        if max_shell is None:
            raise ValueError("axiswise_v1 HookeOrbitalBasis requires max_shell")
        if max_shell < 0:
            raise ValueError(f"max_shell must be nonnegative, got {max_shell}")

        self.basis_semantics: HookeBasisSemantics = "axiswise_v1"
        self.max_shell: int | None = int(max_shell)
        self.truncation: HookeProductTruncation | None = None
        self.max_total_shell: int | None = None
        self.box_size: int | None = None
        self.truncation_bound: int | None = None
        self._product_max_order = None
        self.include_gaussian_factor = (
            True if include_gaussian_factor is None else bool(include_gaussian_factor)
        )
        self._provenance = {
            "basis_semantics": "axiswise_v1",
            "legacy": True,
            "max_shell": self.max_shell,
            "include_gaussian_factor": self.include_gaussian_factor,
        }
        self.register_buffer(
            "_multi_index_tensor",
            torch.empty((0, self.spatial_dim), dtype=torch.long),
            persistent=False,
        )

    def _initialize_product_v2(
        self,
        *,
        max_shell: int | None,
        truncation: str | None,
        max_total_shell: int | None,
        box_size: int | None,
        include_gaussian_factor: bool | None,
    ) -> None:
        """Initialize one explicit, multidimensional product-basis contract."""

        if max_shell is not None:
            raise ValueError(
                "product_v2 HookeOrbitalBasis does not accept legacy max_shell "
                "(basis_semantics now defaults to product_v2; pass "
                "basis_semantics='axiswise_v1' explicitly for the legacy per-axis basis)"
            )
        if include_gaussian_factor is None:
            raise ValueError("product_v2 HookeOrbitalBasis requires include_gaussian_factor")
        if truncation not in {"total_shell", "cartesian_box"}:
            raise ValueError(
                "product_v2 HookeOrbitalBasis truncation must be 'total_shell' or 'cartesian_box'"
            )

        self.basis_semantics = "product_v2"
        self.max_shell = None
        self.include_gaussian_factor = bool(include_gaussian_factor)
        self.truncation = truncation

        if truncation == "total_shell":
            if max_total_shell is None:
                raise ValueError("total_shell product_v2 HookeOrbitalBasis requires max_total_shell")
            if box_size is not None:
                raise ValueError("total_shell product_v2 HookeOrbitalBasis does not accept box_size")
            if max_total_shell < 0:
                raise ValueError(f"max_total_shell must be nonnegative, got {max_total_shell}")
            self.max_total_shell = int(max_total_shell)
            self.box_size = None
            self.truncation_bound = self.max_total_shell
            self._multi_indices = _total_shell_multi_indices(self.spatial_dim, self.max_total_shell)
        else:
            if box_size is None:
                raise ValueError("cartesian_box product_v2 HookeOrbitalBasis requires box_size")
            if max_total_shell is not None:
                raise ValueError("cartesian_box product_v2 HookeOrbitalBasis does not accept max_total_shell")
            if box_size <= 0:
                raise ValueError(f"box_size must be positive, got {box_size}")
            self.max_total_shell = None
            self.box_size = int(box_size)
            self.truncation_bound = self.box_size
            self._multi_indices = _cartesian_box_multi_indices(self.spatial_dim, self.box_size)

        self._product_max_order = _max_multi_index_order(self._multi_indices)
        self._provenance = {
            "basis_semantics": "product_v2",
            "truncation": self.truncation,
            "truncation_bound": self.truncation_bound,
            "multi_index_order": _MULTI_INDEX_ORDER_V1,
            "include_gaussian_factor": self.include_gaussian_factor,
        }
        self.register_buffer(
            "_multi_index_tensor",
            torch.tensor(self._multi_indices, dtype=torch.long),
            persistent=False,
        )

    @property
    def multi_indices(self) -> tuple[tuple[int, ...], ...]:
        """Return immutable product-v2 channel indices in canonical order.

        Legacy V1 has no multidimensional channel meaning, so it returns an
        empty tuple rather than an inferred or misleading product index set.
        """

        return self._multi_indices

    @property
    def coordinate_features(self) -> int:
        """Return configured spatial-channel width without the spin channel."""

        if self.basis_semantics == "axiswise_v1":
            if self.max_shell is None:  # Defensive invariant for static config state.
                raise RuntimeError("axiswise_v1 HookeOrbitalBasis is missing max_shell")
            return self.spatial_dim * (self.max_shell + 1)
        return len(self._multi_indices)

    def forward_impl(self, batch: ElectronBatch) -> ElectronBasisFeatures:
        """Return the configured legacy or product oscillator features."""

        if batch.spatial_dim != self.spatial_dim:
            raise ValueError(
                f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with "
                f"{type(self).__name__} spatial_dim={self.spatial_dim}"
            )
        if self.basis_semantics == "axiswise_v1":
            if self.max_shell is None:  # Defensive invariant for static config state.
                raise RuntimeError("axiswise_v1 HookeOrbitalBasis is missing max_shell")
            features = _hermite_features(
                batch.positions,
                max_order=self.max_shell,
                length_scale=self.length_scale,
                gaussian=self.include_gaussian_factor,
            )
        else:
            if self._product_max_order is None:
                raise RuntimeError("product_v2 HookeOrbitalBasis is missing maximum Hermite order")
            features = _product_hermite_features(
                batch.positions,
                max_order=self._product_max_order,
                length_scale=self.length_scale,
                multi_index_tensor=self._multi_index_tensor,
                gaussian=self.include_gaussian_factor,
            )
        return self._features(features, batch, name="orbital", provenance=self._provenance)


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


def _total_shell_multi_indices(
    spatial_dim: int,
    max_total_shell: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate ``|n| <= max_total_shell`` in the canonical V1 order."""

    return tuple(
        multi_index
        for shell in range(max_total_shell + 1)
        for multi_index in _multi_indices_in_shell(spatial_dim, shell)
    )


def _cartesian_box_multi_indices(
    spatial_dim: int,
    box_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate ``0 <= n_k < box_size`` in the canonical V1 order."""

    return tuple(
        multi_index
        for shell in range(spatial_dim * (box_size - 1) + 1)
        for multi_index in _multi_indices_in_shell(spatial_dim, shell)
        if all(order < box_size for order in multi_index)
    )


def _multi_indices_in_shell(spatial_dim: int, shell: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate one shell with earlier coordinates descending first.

    This recursion is over configured integer dimensions at initialization, not
    over model data. For example, its three-dimensional shell-two order is
    ``(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)``.
    """

    if spatial_dim == 1:
        return ((shell,),)
    return tuple(
        (first_order, *suffix)
        for first_order in range(shell, -1, -1)
        for suffix in _multi_indices_in_shell(spatial_dim - 1, shell - first_order)
    )


def _max_multi_index_order(multi_indices: tuple[tuple[int, ...], ...]) -> int:
    """Return highest one-dimensional Hermite order admitted by a product basis."""

    if not multi_indices:
        raise ValueError("product basis must contain at least one multi-index")
    return max(max(multi_index) for multi_index in multi_indices)


def _product_hermite_features(
    positions: torch.Tensor,
    *,
    max_order: int,
    length_scale: float,
    multi_index_tensor: torch.Tensor,
    gaussian: bool,
) -> torch.Tensor:
    """Evaluate configured multidimensional Hermite products.

    ``multi_index_tensor`` has shape ``[channels, spatial_dim]`` and is
    prepared at basis initialization. Evaluation remains vectorized over every
    sample and electron axis; only the short Hermite recurrence loops over
    configured polynomial order.
    """

    xi = positions / length_scale
    polynomials = [torch.ones_like(xi)]
    if max_order >= 1:
        polynomials.append(2.0 * xi)
    for order in range(1, max_order):
        polynomials.append(2.0 * xi * polynomials[order] - 2.0 * order * polynomials[order - 1])
    stacked = torch.stack(polynomials, dim=-1)  # [*sample, n, spatial_dim, max_order + 1]

    indices = multi_index_tensor
    if indices.device != positions.device:
        # Normal module ``.to(...)`` movement keeps this allocation-free. This
        # fallback makes a stateless basis safe when callers move only a batch.
        indices = indices.to(device=positions.device)
    coordinate_orders = indices.transpose(0, 1)
    gather_indices = coordinate_orders.reshape(
        *((1,) * (stacked.ndim - 2)),
        *coordinate_orders.shape,
    ).expand(*stacked.shape[:-1], coordinate_orders.shape[-1])
    features = torch.gather(stacked, dim=-1, index=gather_indices).prod(dim=-2)
    if gaussian:
        features = features * torch.exp(-0.5 * xi.square().sum(dim=-1, keepdim=True))
    return features


__all__ = [
    "ElectronBasis",
    "ElectronBasisFeatures",
    "HookeHermiteBasis",
    "HookeOrbitalBasis",
    "RawCoordinateBasis",
]
