"""Primitive calculators for helium singular-limit and tail atlases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.bundle import (
    AtlasDerivativeValues,
    AtlasNumericalProvenance,
    EvaluationBundle,
    HeliumAtlasValues,
)
from tpen.evaluation.calculators.local_energy import (
    evaluate_local_energy_in_chunks,
    slice_flat_batch,
    split_local_energy_result,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.nn.cusp import ElectronElectronCusp
from tpen.physics.hamiltonian import HamiltonianTerm, normalize_hamiltonian_terms
from tpen.physics.kinetic import KineticEnergy, per_electron_kinetic_from_logabs


class HeliumAtlasCalculator:
    """Compute derivatives, Hamiltonian decomposition, and numerical status.

    Parameters
    ----------
    hamiltonian_terms : mapping or sequence
        Configured Hamiltonian registry. Mapping keys remain the authoritative
        term names carried by the typed output.
    factor_indices : mapping of str to int
        Public diagnostic names mapped to indices in the restored model's own
        ``model.factors`` module list. A separately instantiated factor would
        not carry restored checkpoint parameters and is therefore rejected by
        construction.
    chunk_size : int or None, optional
        Maximum configurations evaluated with one live autograd graph.
    """

    name = "helium_atlas"

    def __init__(
        self,
        *,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        factor_indices: Mapping[str, int],
        chunk_size: int | None = None,
    ) -> None:
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        indices: dict[str, int] = {}
        for name, index in factor_indices.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("HeliumAtlasCalculator factor names must be non-empty strings")
            if name == "executed_full_logabs":
                raise ValueError("executed_full_logabs is reserved for the complete restored model")
            resolved = int(index)
            if resolved < 0:
                raise ValueError("HeliumAtlasCalculator factor indices must be nonnegative")
            indices[name] = resolved
        if not indices:
            raise ValueError("HeliumAtlasCalculator requires at least one restored-model factor")
        if len(set(indices.values())) != len(indices):
            raise ValueError("HeliumAtlasCalculator factor indices must be unique")
        self.factor_indices = indices
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("HeliumAtlasCalculator chunk_size must be positive when provided")

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Return the bundle with one validated typed helium atlas value."""

        flat = bundle.generated.batch.flatten_samples()
        if flat.dtype != torch.float64:
            raise ValueError(
                "HeliumAtlasCalculator requires float64 so cancellation diagnostics remain float64 end to end"
            )
        if flat.n_electrons != 2 or flat.spatial_dim != 3:
            raise ValueError("HeliumAtlasCalculator requires two three-dimensional electrons")
        factors = _restored_factors(model, self.factor_indices)
        metadata = bundle.generated.metadata
        requested = _metadata_tensor(metadata, "requested_coordinate", flat, shape=(flat.batch_size,))
        tangents = _metadata_tensor(
            metadata,
            "coordinate_tangent",
            flat,
            shape=(flat.batch_size, flat.n_electrons, flat.spatial_dim),
        )
        coordinate_boundary = _metadata_bool(
            metadata, "is_coordinate_representability_boundary", flat
        )
        sentinel = _metadata_bool(metadata, "is_exact_zero_sentinel", flat)
        coordinate_kinds = _metadata_strings(metadata, "atlas_coordinate_kind", flat.batch_size)
        realized = _realized_coordinate(flat, coordinate_kinds, metadata)
        generated_realized = _metadata_tensor(
            metadata, "generated_realized_coordinate", flat, shape=(flat.batch_size,)
        )
        if not torch.allclose(realized, generated_realized, atol=0.0, rtol=1.0e-15):
            raise ValueError(
                "generated_realized_coordinate must equal the unfloored coordinate measured from the batch"
            )
        _validate_provenance_metadata(metadata, flat=flat, context=context)

        size = flat.batch_size if self.chunk_size is None else self.chunk_size
        derivative_chunks: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            "executed_full_logabs": []
        }
        derivative_chunks.update({name: [] for name in factors})
        per_electron_chunks: list[torch.Tensor] = []
        kinetic_names = tuple(
            name for name, term in self.hamiltonian_terms.items() if isinstance(term, KineticEnergy)
        )
        if len(kinetic_names) > 1:
            raise ValueError("HeliumAtlasCalculator supports at most one KineticEnergy registry term")
        for start in range(0, flat.batch_size, size):
            end = min(start + size, flat.batch_size)
            chunk = slice_flat_batch(flat, start, end)
            tangent = tangents[start:end]
            derivative_chunks["executed_full_logabs"].append(
                directional_log_derivatives(model, chunk, tangent)
            )
            for name, factor in factors.items():
                derivative_chunks[name].append(
                    directional_log_derivatives(factor, chunk, tangent)
                )
            if kinetic_names:
                # The helper detaches only after every second derivative exists.
                per_electron_chunks.append(per_electron_kinetic_from_logabs(model, chunk))

        derivatives = {
            name: _concatenate_derivatives(chunks, batch=flat)
            for name, chunks in derivative_chunks.items()
        }
        local_result = evaluate_local_energy_in_chunks(
            self.hamiltonian_terms,
            model,
            flat,
            return_terms=True,
            chunk_size=size,
        )
        total_raw, terms_raw = split_local_energy_result(local_result)
        if terms_raw is None or tuple(terms_raw) != tuple(self.hamiltonian_terms):
            raise ValueError("HeliumAtlasCalculator requires every registry-named Hamiltonian term")
        total = total_raw.detach().to(dtype=torch.float64)
        terms = {
            name: value.detach().to(dtype=torch.float64) for name, value in terms_raw.items()
        }
        if kinetic_names:
            per_electron = torch.cat(per_electron_chunks, dim=0).to(dtype=torch.float64)
        else:
            per_electron = torch.full(
                (flat.batch_size, flat.n_electrons),
                float("nan"),
                device=flat.device,
                dtype=torch.float64,
            )
        per_electron_domain = torch.isfinite(per_electron)
        undefined_status = (
            "undefined_nonfinite" if kinetic_names else "undefined_no_kinetic_registry_term"
        )
        per_electron_status = tuple(
            tuple("defined" if bool(value) else undefined_status for value in row)
            for row in per_electron_domain.detach().cpu().tolist()
        )

        term_stack = torch.stack(tuple(terms.values()), dim=0)
        signed_term_sum = term_stack.sum(dim=0)
        cancellation_abs_sum = term_stack.abs().sum(dim=0)
        cancellation_residual = total - signed_term_sum
        # No clamp: exact cancellation at total == 0 is itself a recorded limit.
        cancellation_ratio = cancellation_abs_sum / total.abs()

        is_ee = all(kind == "electron_electron_distance" for kind in coordinate_kinds)
        is_coordinate_refinement = all(
            kind in {"electron_nucleus_distance", "electron_electron_distance"}
            for kind in coordinate_kinds
        )
        reciprocal_boundary: torch.Tensor | None
        reciprocal_failure_radius: torch.Tensor | None
        positive_separation_domain: torch.Tensor | None
        reciprocal_evaluation_defined: torch.Tensor | None
        coordinate_boundary_radius = torch.full_like(requested, float("nan"))
        coordinate_boundary_radius[coordinate_boundary] = requested[coordinate_boundary]
        ray = _metadata_long(metadata, "ray_id", flat)
        if is_coordinate_refinement or bool(
            torch.any(coordinate_boundary | sentinel).item()
        ):
            _validate_coordinate_refinement(
                requested_coordinate=requested,
                coordinate_boundary_radius=coordinate_boundary_radius,
                coordinate_boundary=coordinate_boundary,
                sentinel=sentinel,
                ray=ray,
            )
        if is_ee:
            # The ideal law is defined on the requested unfloored path coordinate;
            # its numerical boundary stays on the provenance-pinned CPU float64
            # reference while the executed model consumes the realized geometry.
            ideal_inverse = (
                requested.detach().to(device="cpu", dtype=torch.float64).reciprocal()
            ).to(device=flat.device, dtype=flat.dtype)
            positive_separation_domain = requested > 0
            reciprocal_evaluation_defined = torch.isfinite(ideal_inverse)
            reciprocal_boundary = _first_nonfinite_boundaries(
                reciprocal_evaluation_defined,
                ray=ray,
                sentinel=sentinel,
            )
            reciprocal_failure_radius = torch.full_like(requested, float("nan"))
            reciprocal_failure_radius[reciprocal_boundary] = requested[reciprocal_boundary]
            _validate_boundary_order(
                requested_coordinate=requested,
                reciprocal_evaluation_defined_mask=reciprocal_evaluation_defined,
                reciprocal_failure_radius=reciprocal_failure_radius,
                reciprocal_boundary=reciprocal_boundary,
                coordinate_boundary_radius=coordinate_boundary_radius,
                coordinate_boundary=coordinate_boundary,
                sentinel=sentinel,
                ray=ray,
            )
        else:
            if any(kind == "electron_electron_distance" for kind in coordinate_kinds):
                raise ValueError("one helium atlas task may not mix e-e and non-e-e coordinate kinds")
            ideal_inverse = None
            positive_separation_domain = None
            reciprocal_evaluation_defined = None
            reciprocal_boundary = None
            reciprocal_failure_radius = None

        sample_finite = derivatives["executed_full_logabs"].value_finite_mask.clone()
        for values in derivatives.values():
            sample_finite &= values.value_finite_mask
            sample_finite &= values.first_derivative_finite_mask
            sample_finite &= values.second_derivative_finite_mask
        sample_finite &= torch.isfinite(total)
        for values in terms.values():
            sample_finite &= torch.isfinite(values)
        if kinetic_names:
            sample_finite &= per_electron_domain.all(dim=1)
        domain_status = tuple(
            "exact_zero_sentinel"
            if bool(sentinel[index].item())
            else "ideal_unfloored_ee_reciprocal_and_coordinate_representability_boundary"
            if bool(coordinate_boundary[index].item())
            and reciprocal_boundary is not None
            and bool(reciprocal_boundary[index].item())
            else "coordinate_representability_boundary"
            if bool(coordinate_boundary[index].item())
            else "ideal_unfloored_ee_reciprocal_failure_boundary"
            if reciprocal_boundary is not None
            and bool(reciprocal_boundary[index].item())
            else "finite"
            if bool(sample_finite[index].item())
            else "computed_nonfinite_retained"
            for index in range(flat.batch_size)
        )
        values = HeliumAtlasValues(
            requested_coordinate=requested.detach(),
            realized_coordinate=realized.detach(),
            coordinate_representability_boundary_radius=(
                coordinate_boundary_radius.detach()
            ),
            is_coordinate_representability_boundary=coordinate_boundary,
            is_exact_zero_sentinel=sentinel,
            ideal_unfloored_ee_inverse_distance=(
                None if ideal_inverse is None else ideal_inverse.detach()
            ),
            ideal_unfloored_ee_positive_separation_domain_mask=(
                None
                if positive_separation_domain is None
                else positive_separation_domain.detach()
            ),
            ideal_unfloored_ee_reciprocal_evaluation_defined_mask=(
                None
                if reciprocal_evaluation_defined is None
                else reciprocal_evaluation_defined.detach()
            ),
            ideal_unfloored_ee_reciprocal_failure_radius=(
                None
                if reciprocal_failure_radius is None
                else reciprocal_failure_radius.detach()
            ),
            is_ideal_unfloored_ee_reciprocal_failure_boundary=(
                None if reciprocal_boundary is None else reciprocal_boundary.detach()
            ),
            derivatives=derivatives,
            total_local_energy=total,
            total_local_energy_finite_mask=torch.isfinite(total),
            hamiltonian_terms=terms,
            hamiltonian_term_finite_masks={
                name: torch.isfinite(value) for name, value in terms.items()
            },
            per_electron_kinetic=per_electron,
            per_electron_kinetic_domain_mask=per_electron_domain,
            per_electron_kinetic_status=per_electron_status,
            cancellation_abs_sum=cancellation_abs_sum,
            cancellation_residual=cancellation_residual,
            cancellation_ratio=cancellation_ratio,
            cancellation_abs_sum_finite_mask=torch.isfinite(cancellation_abs_sum),
            cancellation_residual_finite_mask=torch.isfinite(cancellation_residual),
            cancellation_ratio_finite_mask=torch.isfinite(cancellation_ratio),
            domain_status=domain_status,
            provenance=AtlasNumericalProvenance(
                dtype=str(metadata["atlas_boundary_dtype"]),
                device=str(metadata["atlas_boundary_device"]),
                evaluation_dtype=str(flat.dtype).removeprefix("torch."),
                evaluation_device=str(flat.device),
                seed=int(context.seed),
            ),
        ).validate(flat)
        return replace(bundle, helium_atlas=values)


def directional_log_derivatives(
    evaluator,
    batch: ElectronBatch,
    tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return vectorized value and two path derivatives before detaching.

    The first backward uses ``create_graph=True``. Detachment occurs only after
    the Hessian-vector contraction has been formed.
    """

    flat = batch.flatten_samples()
    if tuple(tangent.shape) != tuple(flat.positions.shape):
        raise ValueError("directional tangent must match flattened electron positions")
    positions = flat.positions.detach().clone().requires_grad_(True)
    probe = _probe_batch(flat, positions)
    value = _scalar_output(evaluator(probe), batch_size=flat.batch_size)
    gradient = torch.autograd.grad(value.sum(), positions, create_graph=True)[0]
    first = (gradient * tangent).sum(dim=(1, 2))
    hessian_tangent = torch.autograd.grad(first.sum(), positions, retain_graph=False)[0]
    second = (hessian_tangent * tangent).sum(dim=(1, 2))
    return value.detach(), first.detach(), second.detach()


def directional_log_derivatives_reference(
    evaluator,
    batch: ElectronBatch,
    tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the slow one-sample reference for pathwise derivatives."""

    flat = batch.flatten_samples()
    if tuple(tangent.shape) != tuple(flat.positions.shape):
        raise ValueError("directional tangent must match flattened electron positions")
    values: list[torch.Tensor] = []
    first_values: list[torch.Tensor] = []
    second_values: list[torch.Tensor] = []
    for index in range(flat.batch_size):
        sample = slice_flat_batch(flat, index, index + 1)
        positions = sample.positions.detach().clone().requires_grad_(True)
        probe = _probe_batch(sample, positions)
        value = _scalar_output(evaluator(probe), batch_size=1)
        gradient = torch.autograd.grad(value[0], positions, create_graph=True)[0]
        first = (gradient * tangent[index : index + 1]).sum()
        second_gradient = torch.autograd.grad(first, positions)[0]
        second = (second_gradient * tangent[index : index + 1]).sum()
        values.append(value[0].detach())
        first_values.append(first.detach())
        second_values.append(second.detach())
    empty = torch.empty(0, device=flat.device, dtype=flat.dtype)
    return (
        torch.stack(values) if values else empty,
        torch.stack(first_values) if first_values else empty,
        torch.stack(second_values) if second_values else empty,
    )


def _restored_factors(
    model: torch.nn.Module,
    factor_indices: Mapping[str, int],
) -> dict[str, torch.nn.Module]:
    factors = getattr(model, "factors", None)
    if not isinstance(factors, torch.nn.ModuleList):
        raise TypeError(
            "HeliumAtlasCalculator requires the restored model to expose factors as torch.nn.ModuleList"
        )
    selected: dict[str, torch.nn.Module] = {}
    for name, index in factor_indices.items():
        if index >= len(factors):
            raise ValueError(
                f"restored model has {len(factors)} factors, cannot select index {index} for {name!r}"
            )
        factor = factors[index]
        if isinstance(factor, ElectronElectronCusp) and name != "executed_smoothed_ee_factor":
            raise ValueError(
                "ElectronElectronCusp diagnostics must be labelled executed_smoothed_ee_factor "
                "to distinguish them from the ideal unfloored e-e law"
            )
        selected[name] = factor
    return selected


def _concatenate_derivatives(
    chunks: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    batch: ElectronBatch,
) -> AtlasDerivativeValues:
    empty = torch.empty(0, device=batch.device, dtype=batch.dtype)
    value = torch.cat([chunk[0] for chunk in chunks]) if chunks else empty
    first = torch.cat([chunk[1] for chunk in chunks]) if chunks else empty
    second = torch.cat([chunk[2] for chunk in chunks]) if chunks else empty
    return AtlasDerivativeValues(
        value=value,
        first_derivative=first,
        second_derivative=second,
        value_finite_mask=torch.isfinite(value),
        first_derivative_finite_mask=torch.isfinite(first),
        second_derivative_finite_mask=torch.isfinite(second),
    ).validate(batch)


def _scalar_output(output: object, *, batch_size: int) -> torch.Tensor:
    value = output.logabs if isinstance(output, WavefunctionOutput) else output
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,):
        raise TypeError(
            "helium atlas derivative evaluators must return WavefunctionOutput or "
            f"a tensor with shape ({batch_size},)"
        )
    return value


def _probe_batch(batch: ElectronBatch, positions: torch.Tensor) -> ElectronBatch:
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=batch.spins,
        aux=dict(batch.aux),
    )


def _realized_coordinate(
    batch: ElectronBatch,
    coordinate_kinds: Sequence[str],
    metadata: Mapping[str, object],
) -> torch.Tensor:
    positions = batch.positions
    if batch.nuclear_positions is None:
        raise ValueError("helium atlas coordinates require nuclear positions")
    nuclei = batch.nuclear_positions
    nucleus = nuclei[0].expand(batch.batch_size, -1) if nuclei.ndim == 2 else nuclei[:, 0]
    probe = _metadata_long(metadata, "probe_electron", batch)
    values: list[torch.Tensor] = []
    for index, kind in enumerate(coordinate_kinds):
        if kind == "electron_electron_distance":
            value = torch.linalg.vector_norm(positions[index, 0] - positions[index, 1])
        elif kind == "center_of_mass_escape_radius":
            value = torch.linalg.vector_norm(positions[index].mean(dim=0) - nucleus[index])
        elif kind in {
            "electron_nucleus_distance",
            "one_electron_escape_radius",
            "angular_shell_radius",
        }:
            electron = int(probe[index].item())
            if electron not in (0, 1):
                raise ValueError(f"{kind} requires probe_electron 0 or 1")
            value = torch.linalg.vector_norm(positions[index, electron] - nucleus[index])
        else:
            raise ValueError(f"unsupported atlas_coordinate_kind {kind!r}")
        values.append(value)
    return torch.stack(values)


def _metadata_tensor(
    metadata: Mapping[str, object],
    key: str,
    batch: ElectronBatch,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = metadata.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{key} metadata must be a tensor with shape {shape}")
    if value.device != batch.device or value.dtype != batch.dtype:
        raise ValueError(f"{key} metadata must match batch dtype/device")
    return value


def _metadata_bool(
    metadata: Mapping[str, object], key: str, batch: ElectronBatch
) -> torch.Tensor:
    value = metadata.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (batch.batch_size,)
        or value.dtype != torch.bool
        or value.device != batch.device
    ):
        raise ValueError(f"{key} metadata must be bool with shape ({batch.batch_size},)")
    return value


def _metadata_long(
    metadata: Mapping[str, object], key: str, batch: ElectronBatch
) -> torch.Tensor:
    value = metadata.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (batch.batch_size,)
        or value.dtype != torch.long
        or value.device != batch.device
    ):
        raise ValueError(f"{key} metadata must be long with shape ({batch.batch_size},)")
    return value


def _metadata_strings(
    metadata: Mapping[str, object], key: str, batch_size: int
) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} metadata must be a sequence of strings")
    values = tuple(value)
    if len(values) != batch_size or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{key} metadata must contain {batch_size} non-empty strings")
    return values


def _validate_provenance_metadata(
    metadata: Mapping[str, object],
    *,
    flat: ElectronBatch,
    context: EvaluationContext,
) -> None:
    expected = {
        "atlas_seed": int(context.seed),
        "atlas_boundary_dtype": "float64",
        "atlas_boundary_device": "cpu",
        "atlas_evaluation_dtype": str(flat.dtype).removeprefix("torch."),
        "atlas_evaluation_device": str(flat.device),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{key} metadata must record evaluated provenance value {value!r}")


def _first_nonfinite_boundaries(
    finite_mask: torch.Tensor,
    *,
    ray: torch.Tensor,
    sentinel: torch.Tensor,
) -> torch.Tensor:
    """Mark the first finite-to-nonfinite destination on every refinement ray."""

    boundary = torch.zeros_like(finite_mask)
    for ray_id in sorted(set(ray.detach().cpu().tolist())):
        indices = torch.nonzero((ray == ray_id) & ~sentinel, as_tuple=False).reshape(-1)
        transitions = finite_mask[indices][:-1] & ~finite_mask[indices][1:]
        if int(transitions.sum().item()) != 1:
            raise ValueError(
                "each ideal unfloored e-e refinement ray must contain exactly one "
                "finite-to-nonfinite reciprocal transition"
            )
        transition = int(torch.nonzero(transitions, as_tuple=False).item())
        boundary[indices[transition + 1]] = True
    return boundary


def _validate_boundary_order(
    *,
    requested_coordinate: torch.Tensor,
    reciprocal_evaluation_defined_mask: torch.Tensor,
    reciprocal_failure_radius: torch.Tensor,
    reciprocal_boundary: torch.Tensor,
    coordinate_boundary_radius: torch.Tensor,
    coordinate_boundary: torch.Tensor,
    sentinel: torch.Tensor,
    ray: torch.Tensor,
) -> None:
    """Validate reciprocal failure against an already valid coordinate refinement."""

    for ray_id in sorted(set(ray.detach().cpu().tolist())):
        selection = ray == ray_id
        indices = torch.nonzero(selection, as_tuple=False).reshape(-1)
        ray_sentinel = sentinel[indices]
        nonzero_indices = indices[~ray_sentinel]

        evaluation_defined = reciprocal_evaluation_defined_mask[nonzero_indices]
        transitions = evaluation_defined[:-1] & ~evaluation_defined[1:]
        if int(transitions.sum().item()) != 1:
            raise ValueError(
                "each ideal unfloored e-e ray must contain exactly one finite-to-nonfinite "
                "reciprocal transition"
            )
        transition = int(torch.nonzero(transitions, as_tuple=False).item())
        reciprocal_destination = nonzero_indices[transition + 1]
        if (
            not bool(torch.all(evaluation_defined[: transition + 1]).item())
            or bool(torch.any(evaluation_defined[transition + 1 :]).item())
        ):
            raise ValueError(
                "ideal unfloored e-e reciprocal evaluation must stay undefined after its "
                "first failure"
            )

        reciprocal = reciprocal_failure_radius[selection & reciprocal_boundary]
        coordinate = coordinate_boundary_radius[selection & coordinate_boundary]
        if reciprocal.numel() != 1 or coordinate.numel() != 1:
            raise ValueError(
                "each ideal unfloored e-e ray must emit one reciprocal-failure radius "
                "and one coordinate-representability radius"
            )
        if bool((reciprocal[0] < coordinate[0]).item()):
            raise ValueError(
                "ideal unfloored e-e reciprocal evaluation must fail at a radius "
                "greater than or equal to the coordinate-representability boundary"
            )
        reciprocal_index = torch.nonzero(
            selection & reciprocal_boundary, as_tuple=False
        ).reshape(-1)[0]
        coordinate_index = torch.nonzero(
            selection & coordinate_boundary, as_tuple=False
        ).reshape(-1)[0]
        if bool((reciprocal_index != reciprocal_destination).item()):
            raise ValueError(
                "the reciprocal-failure boundary must mark the finite-to-nonfinite "
                "transition destination"
            )
        if (
            bool((reciprocal[0] <= 0).item())
            or bool((coordinate[0] <= 0).item())
            or bool(
                (reciprocal[0] != requested_coordinate[reciprocal_index]).item()
            )
            or bool(
                (coordinate[0] != requested_coordinate[coordinate_index]).item()
            )
        ):
            raise ValueError(
                "named numerical-boundary radii must be positive requested coordinates "
                "and distinct from the exact-zero sentinel"
            )
        ray_reciprocal_index = int(
            torch.nonzero(indices == reciprocal_index, as_tuple=False).item()
        )
        ray_coordinate_index = int(
            torch.nonzero(indices == coordinate_index, as_tuple=False).item()
        )
        if ray_reciprocal_index > ray_coordinate_index:
            raise ValueError(
                "ideal unfloored e-e reciprocal evaluation must fail before or at the "
                "coordinate-representability boundary"
            )


def _validate_coordinate_refinement(
    *,
    requested_coordinate: torch.Tensor,
    coordinate_boundary_radius: torch.Tensor,
    coordinate_boundary: torch.Tensor,
    sentinel: torch.Tensor,
    ray: torch.Tensor,
) -> None:
    """Require a monotone positive ray, one terminal boundary, and one zero sentinel."""

    for ray_id in sorted(set(ray.detach().cpu().tolist())):
        selection = ray == ray_id
        indices = torch.nonzero(selection, as_tuple=False).reshape(-1)
        ray_sentinel = sentinel[indices]
        if int(ray_sentinel.sum().item()) != 1 or not bool(ray_sentinel[-1].item()):
            raise ValueError(
                "each coordinate-refinement ray must terminate with one exact-zero sentinel"
            )
        nonzero_indices = indices[~ray_sentinel]
        nonzero_requested = requested_coordinate[nonzero_indices]
        if (
            nonzero_requested.numel() < 2
            or bool(torch.any(nonzero_requested <= 0).item())
            or not bool(torch.all(nonzero_requested[1:] < nonzero_requested[:-1]).item())
        ):
            raise ValueError(
                "each coordinate-refinement ray must approach zero through a strictly "
                "decreasing positive requested-coordinate sequence"
            )
        if bool((requested_coordinate[indices[-1]] != 0).item()):
            raise ValueError("the exact-zero sentinel must have requested_coordinate == 0")
        ray_coordinate_boundary = coordinate_boundary[indices]
        if int(ray_coordinate_boundary.sum().item()) != 1:
            raise ValueError(
                "each coordinate-refinement ray must emit one coordinate-representability boundary"
            )
        coordinate_index = indices[
            torch.nonzero(ray_coordinate_boundary, as_tuple=False).reshape(-1)[0]
        ]
        if bool((coordinate_index != nonzero_indices[-1]).item()):
            raise ValueError(
                "the coordinate-representability boundary must terminate the positive ray"
            )
        coordinate_radius = coordinate_boundary_radius[coordinate_index]
        if (
            bool((coordinate_radius <= 0).item())
            or bool(
                (coordinate_radius != requested_coordinate[coordinate_index]).item()
            )
        ):
            raise ValueError(
                "the coordinate-representability radius must be the terminal positive "
                "requested coordinate and distinct from the exact-zero sentinel"
            )


__all__ = [
    "HeliumAtlasCalculator",
    "directional_log_derivatives",
    "directional_log_derivatives_reference",
]
