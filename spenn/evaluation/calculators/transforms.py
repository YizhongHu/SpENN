"""Transform-comparison calculators for evaluation tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.evaluation.bundle import EvaluationBundle, TransformComparisonValues
from spenn.evaluation.calculators.local_energy import evaluate_local_energy_in_chunks, split_local_energy_result
from spenn.evaluation.protocols import EvaluationContext
from spenn.physics.hamiltonian import HamiltonianTerm, normalize_hamiltonian_terms


class FullModelAntisymmetryCalculator:
    """Compare full model outputs under fermionic particle permutations."""

    name = "full_model_antisymmetry"

    def __init__(self, *, atol: float = 1.0e-6, rtol: float = 1.0e-6, compare_sign: bool = True) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.compare_sign = bool(compare_sign)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Evaluate original/transformed pairs and store raw errors."""

        del context
        original, transformed = split_paired_batch(bundle.generated.batch)
        original_output, transformed_output = _evaluate_pair(model, original, transformed)
        parity = _metadata_vector(
            bundle.generated.metadata,
            "permutation_parity",
            like=original_output.sign,
        )
        expected_sign = original_output.sign.reshape(-1) * parity.reshape(-1)
        sign_mismatch = torch.zeros_like(expected_sign, dtype=torch.bool)
        if self.compare_sign:
            sign_mismatch = ~torch.isclose(
                transformed_output.sign.reshape(-1),
                expected_sign,
                atol=self.atol,
                rtol=self.rtol,
            )
        return replace(
            bundle,
            transform=TransformComparisonValues(
                original_logabs=original_output.logabs.detach().reshape(-1),
                transformed_logabs=transformed_output.logabs.detach().reshape(-1),
                original_sign=original_output.sign.detach().reshape(-1),
                transformed_sign=transformed_output.sign.detach().reshape(-1),
                logabs_abs_error=(transformed_output.logabs.reshape(-1) - original_output.logabs.reshape(-1)).abs().detach(),
                sign_mismatch=sign_mismatch.detach(),
                metadata={**bundle.generated.metadata, "expected_sign": expected_sign.detach()},
            ),
        )


class SpatialExchangeSymmetryCalculator:
    """Compare Hooke spatial-singlet outputs under coordinate exchange."""

    name = "spatial_exchange_symmetry"

    def __init__(self, *, atol: float = 1.0e-6, rtol: float = 1.0e-6, compare_sign: bool = True) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.compare_sign = bool(compare_sign)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Store same-sign transform errors for spatial exchange."""

        del context
        original, transformed = split_paired_batch(bundle.generated.batch)
        original_output, transformed_output = _evaluate_pair(model, original, transformed)
        expected_sign = original_output.sign.reshape(-1)
        sign_mismatch = torch.zeros_like(expected_sign, dtype=torch.bool)
        if self.compare_sign:
            sign_mismatch = ~torch.isclose(
                transformed_output.sign.reshape(-1),
                expected_sign,
                atol=self.atol,
                rtol=self.rtol,
            )
        return replace(
            bundle,
            transform=TransformComparisonValues(
                original_logabs=original_output.logabs.detach().reshape(-1),
                transformed_logabs=transformed_output.logabs.detach().reshape(-1),
                original_sign=original_output.sign.detach().reshape(-1),
                transformed_sign=transformed_output.sign.detach().reshape(-1),
                logabs_abs_error=(transformed_output.logabs.reshape(-1) - original_output.logabs.reshape(-1)).abs().detach(),
                sign_mismatch=sign_mismatch.detach(),
                metadata={**bundle.generated.metadata, "expected_sign": expected_sign.detach()},
            ),
        )


class RotationConsistencyCalculator:
    """Compare model outputs and optional local energies under rotations."""

    name = "rotation_consistency"

    def __init__(
        self,
        *,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm] | None = None,
        chunk_size: int | None = None,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
        compare_sign: bool = True,
    ) -> None:
        self.hamiltonian_terms = None if hamiltonian_terms is None else normalize_hamiltonian_terms(hamiltonian_terms)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.compare_sign = bool(compare_sign)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Store same-sign wavefunction and optional local-energy errors."""

        del context
        original, transformed = split_paired_batch(bundle.generated.batch)
        original_output, transformed_output = _evaluate_pair(model, original, transformed)
        expected_sign = original_output.sign.reshape(-1)
        sign_mismatch = torch.zeros_like(expected_sign, dtype=torch.bool)
        if self.compare_sign:
            sign_mismatch = ~torch.isclose(
                transformed_output.sign.reshape(-1),
                expected_sign,
                atol=self.atol,
                rtol=self.rtol,
            )
        local_energy_abs_error = None
        if self.hamiltonian_terms is not None:
            original_energy, _ = split_local_energy_result(
                evaluate_local_energy_in_chunks(
                    self.hamiltonian_terms,
                    model,
                    original,
                    chunk_size=self.chunk_size,
                )
            )
            transformed_energy, _ = split_local_energy_result(
                evaluate_local_energy_in_chunks(
                    self.hamiltonian_terms,
                    model,
                    transformed,
                    chunk_size=self.chunk_size,
                )
            )
            local_energy_abs_error = (transformed_energy.reshape(-1) - original_energy.reshape(-1)).abs().detach()
        return replace(
            bundle,
            transform=TransformComparisonValues(
                original_logabs=original_output.logabs.detach().reshape(-1),
                transformed_logabs=transformed_output.logabs.detach().reshape(-1),
                original_sign=original_output.sign.detach().reshape(-1),
                transformed_sign=transformed_output.sign.detach().reshape(-1),
                logabs_abs_error=(transformed_output.logabs.reshape(-1) - original_output.logabs.reshape(-1)).abs().detach(),
                sign_mismatch=sign_mismatch.detach(),
                metadata={**bundle.generated.metadata, "expected_sign": expected_sign.detach()},
                local_energy_abs_error=local_energy_abs_error,
            ),
        )


def split_paired_batch(batch: ElectronBatch) -> tuple[ElectronBatch, ElectronBatch]:
    """Split a paired ``[orbit, 2]`` evaluation batch into two flat batches."""

    if len(batch.sample_shape) < 1 or batch.sample_shape[-1] != 2:
        raise ValueError(
            "transform calculators require generated batch sample_shape ending in 2 "
            f"for original/transformed pairs, got {batch.sample_shape}"
        )
    positions = batch.positions.reshape(-1, 2, batch.n_electrons, batch.spatial_dim)
    spins = None if batch.spins is None else batch.spins.reshape(-1, 2, batch.n_electrons)
    original = ElectronBatch(
        positions=positions[:, 0, :, :],
        system=batch.system,
        nuclear_positions=_split_optional(batch.nuclear_positions, side=0, n_pairs=positions.shape[0], tail_rank=2),
        nuclear_charges=_split_optional(batch.nuclear_charges, side=0, n_pairs=positions.shape[0], tail_rank=1),
        spins=None if spins is None else spins[:, 0, :],
        aux={},
    )
    transformed = ElectronBatch(
        positions=positions[:, 1, :, :],
        system=batch.system,
        nuclear_positions=_split_optional(batch.nuclear_positions, side=1, n_pairs=positions.shape[0], tail_rank=2),
        nuclear_charges=_split_optional(batch.nuclear_charges, side=1, n_pairs=positions.shape[0], tail_rank=1),
        spins=None if spins is None else spins[:, 1, :],
        aux={},
    )
    return original, transformed


def _evaluate_pair(
    model: torch.nn.Module,
    original: ElectronBatch,
    transformed: ElectronBatch,
) -> tuple[WavefunctionOutput, WavefunctionOutput]:
    with torch.no_grad():
        original_output = model(original)
        transformed_output = model(transformed)
    if not isinstance(original_output, WavefunctionOutput) or not isinstance(transformed_output, WavefunctionOutput):
        raise TypeError("transform calculators require the model to return WavefunctionOutput")
    return original_output, transformed_output


def _split_optional(value: torch.Tensor | None, *, side: int, n_pairs: int, tail_rank: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == tail_rank:
        return value
    paired_shape = (n_pairs, 2)
    if value.ndim == tail_rank + 2 and tuple(value.shape[:2]) == paired_shape:
        return value[:, side, ...]
    raise ValueError(f"optional paired tensor has unsupported shape {tuple(value.shape)}")


def _metadata_vector(
    metadata: Mapping[str, object],
    key: str,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"metadata field {key!r} is required")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"metadata field {key!r} must be a tensor")
    return value.to(device=like.device, dtype=like.dtype).reshape(-1)


__all__ = [
    "FullModelAntisymmetryCalculator",
    "RotationConsistencyCalculator",
    "SpatialExchangeSymmetryCalculator",
    "split_paired_batch",
]
