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
        boundary = _metadata_bool(metadata, "is_refinement_boundary", flat)
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
        if is_ee:
            ideal_inverse = realized.reciprocal()
            ideal_domain = realized > 0
        else:
            if any(kind == "electron_electron_distance" for kind in coordinate_kinds):
                raise ValueError("one helium atlas task may not mix e-e and non-e-e coordinate kinds")
            ideal_inverse = None
            ideal_domain = None

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
            else "recorded_numerical_refinement_boundary"
            if bool(boundary[index].item())
            else "finite"
            if bool(sample_finite[index].item())
            else "computed_nonfinite_retained"
            for index in range(flat.batch_size)
        )
        values = HeliumAtlasValues(
            requested_coordinate=requested.detach(),
            realized_coordinate=realized.detach(),
            is_refinement_boundary=boundary,
            is_exact_zero_sentinel=sentinel,
            ideal_unfloored_ee_inverse_distance=(
                None if ideal_inverse is None else ideal_inverse.detach()
            ),
            ideal_unfloored_ee_domain_mask=(
                None if ideal_domain is None else ideal_domain.detach()
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


__all__ = [
    "HeliumAtlasCalculator",
    "directional_log_derivatives",
    "directional_log_derivatives_reference",
]
