"""H2-only bounded-cusp and smooth-tail nuclear confinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tpen.data.batch import ElectronBatch
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import permutation_index, permute_particle_axis
from tpen.data.permutation import Permutation
from tpen.dependencies import require_torch, require_torch_nn
from tpen.nn.envelope import Envelope, _check_envelope_tensor

torch = require_torch(feature="H2 nuclear confinement")
nn = require_torch_nn(feature="H2 nuclear confinement")


@dataclass(frozen=True)
class H2NuclearConfinementEvaluation:
    """Typed bounded-cusp data and separate smooth H2 tail.

    The pair fields have shape ``[batch, n_electrons, 2]``. ``value`` is only
    the bounded cusp contribution and is therefore the part intended for a
    future separated Coulomb-cancellation evaluator. ``smooth_tail_logabs`` has
    shape ``[batch, n_electrons]`` and remains a regular log-amplitude term.
    """

    distance: torch.Tensor
    value: torch.Tensor
    radial_first_derivative: torch.Tensor
    radial_second_derivative: torch.Tensor
    origin_radial_derivative: torch.Tensor
    cusp_residual: torch.Tensor
    smooth_tail_logabs: torch.Tensor

    def bounded_cusp_logabs(self) -> torch.Tensor:
        """Reduce identical-H pair values without label-order roundoff."""

        return _sum_identical_h_pair_values(self.value)

    def validate(self, batch: ElectronBatch) -> "H2NuclearConfinementEvaluation":
        """Validate the explicit H2 shapes, devices, and finite limits."""

        flat, _, charges = _validated_h2_geometry(batch)
        pair_shape = (flat.batch_size, flat.n_electrons, 2)
        for name, value in (
            ("distance", self.distance),
            ("value", self.value),
            ("radial_first_derivative", self.radial_first_derivative),
            ("radial_second_derivative", self.radial_second_derivative),
            ("cusp_residual", self.cusp_residual),
        ):
            if value.shape != pair_shape:
                raise ValueError(f"H2NuclearConfinementEvaluation.{name} must have shape {pair_shape}")
            if value.device != flat.device or value.dtype != flat.dtype or not torch.isfinite(value).all():
                raise ValueError(
                    f"H2NuclearConfinementEvaluation.{name} must be finite and match batch dtype/device"
                )
        origin_shape = (flat.batch_size, 2)
        if self.origin_radial_derivative.shape != origin_shape:
            raise ValueError(
                "H2NuclearConfinementEvaluation.origin_radial_derivative "
                f"must have shape {origin_shape}"
            )
        if (
            self.origin_radial_derivative.device != flat.device
            or self.origin_radial_derivative.dtype != flat.dtype
            or not torch.isfinite(self.origin_radial_derivative).all()
        ):
            raise ValueError(
                "H2NuclearConfinementEvaluation.origin_radial_derivative "
                "must be finite and match batch dtype/device"
            )
        tail_shape = (flat.batch_size, flat.n_electrons)
        if self.smooth_tail_logabs.shape != tail_shape:
            raise ValueError(
                f"H2NuclearConfinementEvaluation.smooth_tail_logabs must have shape {tail_shape}"
            )
        if (
            self.smooth_tail_logabs.device != flat.device
            or self.smooth_tail_logabs.dtype != flat.dtype
            or not torch.isfinite(self.smooth_tail_logabs).all()
        ):
            raise ValueError(
                "H2NuclearConfinementEvaluation.smooth_tail_logabs "
                "must be finite and match batch dtype/device"
            )
        if torch.any(self.distance < 0):
            raise ValueError("H2NuclearConfinementEvaluation.distance must be nonnegative")
        if not torch.equal(self.origin_radial_derivative, -charges):
            raise ValueError(
                "H2NuclearConfinementEvaluation.origin_radial_derivative "
                "must equal -batch.nuclear_charges"
            )
        return self

    def require_separated_local_energy_domain(self) -> "H2NuclearConfinementEvaluation":
        """Reject exact coalescence for separated local-energy consumption."""

        if torch.any(self.distance == 0):
            raise ValueError(
                "separated H2 local-energy evaluation is undefined at exact "
                "electron-nucleus coalescence"
            )
        return self

    def permute(self, permutation: Permutation) -> "H2NuclearConfinementEvaluation":
        """Apply an electron permutation while leaving H labels fixed."""

        return type(self)(
            distance=permute_particle_axis(self.distance, permutation, axis=1),
            value=permute_particle_axis(self.value, permutation, axis=1),
            radial_first_derivative=permute_particle_axis(
                self.radial_first_derivative, permutation, axis=1
            ),
            radial_second_derivative=permute_particle_axis(
                self.radial_second_derivative, permutation, axis=1
            ),
            origin_radial_derivative=self.origin_radial_derivative.clone(),
            cusp_residual=permute_particle_axis(self.cusp_residual, permutation, axis=1),
            smooth_tail_logabs=permute_particle_axis(
                self.smooth_tail_logabs, permutation, axis=1
            ),
        )

    def permute_nuclei(self, permutation: Permutation) -> "H2NuclearConfinementEvaluation":
        """Relabel the two identical H nuclei in every nucleus-indexed field."""

        if len(permutation) != 2:
            raise ValueError(f"H2 nuclear permutations must have size 2, got {len(permutation)}")
        index = permutation_index(permutation, device=self.distance.device)
        return type(self)(
            distance=self.distance.index_select(2, index),
            value=self.value.index_select(2, index),
            radial_first_derivative=self.radial_first_derivative.index_select(2, index),
            radial_second_derivative=self.radial_second_derivative.index_select(2, index),
            origin_radial_derivative=self.origin_radial_derivative.index_select(1, index),
            cusp_residual=self.cusp_residual.index_select(2, index),
            smooth_tail_logabs=self.smooth_tail_logabs.clone(),
        )

    def compare(
        self,
        other: "H2NuclearConfinementEvaluation",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare every cusp and smooth-tail field in explicit order."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(
            [
                self.distance,
                self.value,
                self.radial_first_derivative,
                self.radial_second_derivative,
                self.origin_radial_derivative,
                self.cusp_residual,
                self.smooth_tail_logabs,
            ],
            [
                other.distance,
                other.value,
                other.radial_first_derivative,
                other.radial_second_derivative,
                other.origin_radial_derivative,
                other.cusp_residual,
                other.smooth_tail_logabs,
            ],
            atol=atol,
            rtol=rtol,
        )


class H2NuclearConfinement(Envelope):
    """H2-only bounded cusp with a separate smooth molecular tail.

    Parameters
    ----------
    beta_H : float
        Positive shared range of both H cusp factors.
    a : float
        Positive shared smoothing scale of the molecular tail.
    kappa : float
        Positive asymptotic decay rate of the molecular tail.

    Notes
    -----
    The three scalars are persistent buffers so their checkpoint ownership is
    explicit. They are never indexed by nucleus or inferred from charge or bond
    length.
    """

    def __init__(self, *, beta_H: float, a: float, kappa: float) -> None:
        super().__init__(enabled=True)
        self.register_buffer("beta_H", _positive_scalar(beta_H, "beta_H"))
        self.register_buffer("a", _positive_scalar(a, "a"))
        self.register_buffer("kappa", _positive_scalar(kappa, "kappa"))

    def evaluate_reference(self, batch: ElectronBatch) -> H2NuclearConfinementEvaluation:
        """Evaluate the H2 formula with explicit slow scalar loops.

        This is the readable reference implementation against which the normal
        vectorized path is tested. It deliberately loops over configurations,
        electrons, and the two nuclei.
        """

        flat, nuclei, charges = _validated_h2_geometry(batch)
        beta_H, a, kappa = self._scalars_for(flat)
        distance_batches = []
        value_batches = []
        first_batches = []
        second_batches = []
        residual_batches = []
        tail_batches = []
        for batch_index in range(flat.batch_size):
            distance_electrons = []
            value_electrons = []
            first_electrons = []
            second_electrons = []
            residual_electrons = []
            tail_electrons = []
            for electron_index in range(flat.n_electrons):
                distances = []
                values = []
                first_derivatives = []
                second_derivatives = []
                residuals = []
                for nucleus_index in range(2):
                    distance = torch.linalg.vector_norm(
                        flat.positions[batch_index, electron_index]
                        - nuclei[batch_index, nucleus_index]
                    )
                    charge = charges[batch_index, nucleus_index]
                    exponential = torch.exp(-beta_H * distance)
                    distances.append(distance)
                    values.append((charge / beta_H) * torch.expm1(-beta_H * distance))
                    first_derivatives.append(-charge * exponential)
                    second_derivatives.append(charge * beta_H * exponential)
                    safe_distance = torch.where(
                        distance == 0, torch.ones_like(distance), distance
                    )
                    residual = charge * (-torch.expm1(-beta_H * distance)) / safe_distance
                    residuals.append(torch.where(distance == 0, charge * beta_H, residual))
                distance_pair = torch.stack(distances)
                radius_squared = distance_pair.square()
                product = radius_squared[0] * radius_squared[1]
                fraction = product / (radius_squared[0] + radius_squared[1])
                root = torch.sqrt(a.square() + 2.0 * fraction)
                smooth_tail = -kappa * (2.0 * fraction / (root + a))
                distance_electrons.append(distance_pair)
                value_electrons.append(torch.stack(values))
                first_electrons.append(torch.stack(first_derivatives))
                second_electrons.append(torch.stack(second_derivatives))
                residual_electrons.append(torch.stack(residuals))
                tail_electrons.append(smooth_tail)
            distance_batches.append(torch.stack(distance_electrons))
            value_batches.append(torch.stack(value_electrons))
            first_batches.append(torch.stack(first_electrons))
            second_batches.append(torch.stack(second_electrons))
            residual_batches.append(torch.stack(residual_electrons))
            tail_batches.append(torch.stack(tail_electrons))
        evaluation = H2NuclearConfinementEvaluation(
            distance=torch.stack(distance_batches),
            value=torch.stack(value_batches),
            radial_first_derivative=torch.stack(first_batches),
            radial_second_derivative=torch.stack(second_batches),
            origin_radial_derivative=-charges,
            cusp_residual=torch.stack(residual_batches),
            smooth_tail_logabs=torch.stack(tail_batches),
        )
        return evaluation.validate(flat)

    def evaluate(self, batch: ElectronBatch) -> H2NuclearConfinementEvaluation:
        """Evaluate vectorized cusp derivatives and the separate smooth tail."""

        flat, nuclei, charges = _validated_h2_geometry(batch)
        distance = torch.linalg.vector_norm(
            flat.positions.unsqueeze(2) - nuclei.unsqueeze(1), dim=-1
        )
        value, first, second, residual, smooth_tail = self._value_and_derivatives(
            distance, charges, flat
        )
        evaluation = H2NuclearConfinementEvaluation(
            distance=distance,
            value=value,
            radial_first_derivative=first,
            radial_second_derivative=second,
            origin_radial_derivative=-charges,
            cusp_residual=residual,
            smooth_tail_logabs=smooth_tail,
        )
        return evaluation.validate(flat)

    def value_parts(self, batch: ElectronBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded cusp and smooth-tail values without derivatives."""

        flat, nuclei, charges = _validated_h2_geometry(batch)
        distance = torch.linalg.vector_norm(
            flat.positions.unsqueeze(2) - nuclei.unsqueeze(1), dim=-1
        )
        beta_H, a, kappa = self._scalars_for(flat)
        pair_charges = charges.unsqueeze(1)
        cusp_value = (pair_charges / beta_H) * torch.expm1(-beta_H * distance)
        smooth_tail = _smooth_tail_logabs(distance, a=a, kappa=kappa)
        return cusp_value, smooth_tail

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the complete H2 log-amplitude contribution value-only."""

        cusp_value, smooth_tail = self.value_parts(batch)
        output = _sum_identical_h_pair_values(cusp_value) + smooth_tail.sum(dim=1)
        assert output.shape == (batch.flatten_samples().batch_size,)
        return output

    def _scalars_for(self, batch: ElectronBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            value.to(device=batch.device, dtype=batch.dtype)
            for value in (self.beta_H, self.a, self.kappa)
        )

    def _value_and_derivatives(
        self,
        distance: torch.Tensor,
        charges: torch.Tensor,
        batch: ElectronBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_H, a, kappa = self._scalars_for(batch)
        pair_charges = charges.unsqueeze(1)
        exponential = torch.exp(-beta_H * distance)
        value = (pair_charges / beta_H) * torch.expm1(-beta_H * distance)
        first = -pair_charges * exponential
        second = pair_charges * beta_H * exponential
        safe_distance = torch.where(distance == 0, torch.ones_like(distance), distance)
        residual = pair_charges * (-torch.expm1(-beta_H * distance)) / safe_distance
        residual = torch.where(distance == 0, pair_charges * beta_H, residual)
        smooth_tail = _smooth_tail_logabs(distance, a=a, kappa=kappa)
        return value, first, second, residual, smooth_tail


class H2NuclearFactorizedEnvelope(Envelope):
    """Explicit ownership of regular and H2 nuclear log-amplitude factors."""

    def __init__(
        self,
        regular_envelope: nn.Module,
        nuclear_confinement: H2NuclearConfinement,
    ) -> None:
        super().__init__(enabled=True)
        self.regular_envelope = regular_envelope
        self.nuclear_confinement = nuclear_confinement

    def factorized_value(self, batch: ElectronBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return regular-plus-tail and bounded-cusp log-amplitude values."""

        flat = batch.flatten_samples()
        regular = self.regular_envelope(batch)
        _check_envelope_tensor(regular, flat, name="regular_envelope")
        cusp_value, smooth_tail = self.nuclear_confinement.value_parts(flat)
        regular_with_tail = regular + smooth_tail.sum(dim=1)
        cusp = _sum_identical_h_pair_values(cusp_value)
        return regular_with_tail, cusp

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the complete value-only H2 envelope."""

        regular_with_tail, cusp = self.factorized_value(batch)
        return regular_with_tail + cusp


def _validated_h2_geometry(
    batch: ElectronBatch,
) -> tuple[ElectronBatch, torch.Tensor, torch.Tensor]:
    """Return flattened, expanded H2 geometry after strict metadata checks."""

    flat = batch.flatten_samples()
    if flat.nuclear_positions is None:
        raise ValueError("H2NuclearConfinement requires batch.nuclear_positions")
    if flat.nuclear_charges is None:
        raise ValueError("H2NuclearConfinement requires batch.nuclear_charges")
    if flat.nuclear_positions.shape[-2] != 2 or flat.nuclear_charges.shape[-1] != 2:
        raise ValueError("H2NuclearConfinement requires exactly two H nuclei; N=1 is invalid")
    nuclei = flat.nuclear_positions.to(device=flat.device, dtype=flat.dtype)
    charges = flat.nuclear_charges.to(device=flat.device, dtype=flat.dtype)
    if nuclei.ndim == 2:
        nuclei = nuclei.unsqueeze(0).expand(flat.batch_size, -1, -1)
    if charges.ndim == 1:
        charges = charges.unsqueeze(0).expand(flat.batch_size, -1)
    if not torch.isfinite(flat.positions).all():
        raise ValueError("H2NuclearConfinement requires finite electron positions")
    if not torch.isfinite(nuclei).all():
        raise ValueError("H2NuclearConfinement requires finite nuclear positions")
    if not torch.isfinite(charges).all():
        raise ValueError("H2NuclearConfinement requires finite nuclear charges")
    if not torch.equal(charges, torch.ones_like(charges)):
        raise ValueError("H2NuclearConfinement requires exactly two unit H charges")
    separation = torch.linalg.vector_norm(nuclei[:, 0] - nuclei[:, 1], dim=-1)
    if torch.any(separation == 0):
        raise ValueError("H2NuclearConfinement requires distinct, nondegenerate nuclear positions")
    return flat, nuclei, charges


def _smooth_tail_logabs(
    distance: torch.Tensor,
    *,
    a: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    """Return stable ``-kappa * (sqrt(a^2 + 2F) - a)`` values."""

    radius_squared = distance.square()
    fraction = radius_squared[..., 0] * radius_squared[..., 1] / radius_squared.sum(dim=-1)
    root = torch.sqrt(a.square() + 2.0 * fraction)
    return -kappa * (2.0 * fraction / (root + a))


def _sum_identical_h_pair_values(value: torch.Tensor) -> torch.Tensor:
    """Sum each H pair symmetrically before reducing over electrons."""

    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError("H2 pair values must have shape [batch, n_electrons, 2]")
    return (value[..., 0] + value[..., 1]).sum(dim=1)


def _positive_scalar(value: float, name: str) -> torch.Tensor:
    """Return one finite positive float64 scalar tensor."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a shared scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return torch.tensor(scalar, dtype=torch.float64)


__all__ = [
    "H2NuclearConfinement",
    "H2NuclearConfinementEvaluation",
    "H2NuclearFactorizedEnvelope",
]
