"""Local-energy evaluation calculator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.bundle import EvaluationBundle, LocalEnergyValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.hamiltonian import HamiltonianTerm, LocalEnergyResult, local_energy, normalize_hamiltonian_terms


class LocalEnergyCalculator:
    """Compute raw local-energy values for generated configurations."""

    name = "local_energy"

    def __init__(
        self,
        *,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        return_terms: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        self.return_terms = bool(return_terms)
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Evaluate local energy and return a bundle with raw values."""

        records = bundle.generated.trajectory_records
        if records is not None:
            records.validate(check_files=False)
            if tuple(self.hamiltonian_terms) != records.term_names:
                raise ValueError(
                    "LocalEnergyCalculator terms disagree with streamed trajectory records"
                )
            flat = bundle.generated.batch.flatten_samples()
            n_rows = records.validate_snapshot_batch(flat)
            total = records.final_draw.local_energy[:n_rows].to(
                device=flat.device,
                dtype=flat.dtype,
            )
            terms = None
            if self.return_terms:
                terms = {
                    name: records.final_draw.term_energies[name][:n_rows].to(
                        device=flat.device,
                        dtype=flat.dtype,
                    )
                    for name in records.term_names
                }
            return replace(
                bundle,
                local_energy=LocalEnergyValues(
                    local_energy=total,
                    finite_mask=torch.isfinite(total),
                    term_energies=terms,
                ),
            )

        result = evaluate_local_energy_in_chunks(
            self.hamiltonian_terms,
            model,
            bundle.generated.batch,
            return_terms=self.return_terms,
            chunk_size=self.chunk_size,
        )
        total, term_energies = split_local_energy_result(result)
        total = total.detach()
        terms = None if term_energies is None else {name: value.detach() for name, value in term_energies.items()}
        local = LocalEnergyValues(
            local_energy=total,
            finite_mask=torch.isfinite(total),
            term_energies=terms,
        )
        return replace(bundle, local_energy=local)


def split_local_energy_result(
    result: LocalEnergyResult | torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
    """Return ``(total, terms_or_none)`` from a local-energy result."""

    if isinstance(result, LocalEnergyResult):
        return result.total, result.terms
    return result, None


def evaluate_local_energy_in_chunks(
    terms: Mapping[str, HamiltonianTerm],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
    chunk_size: int | None = None,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate local energy on bounded flattened batches."""

    flat = batch.flatten_samples()
    batch_size = flat.batch_size
    if batch_size == 0:
        total = torch.empty(0, device=flat.device, dtype=flat.dtype)
        return LocalEnergyResult(total=total, terms={}) if return_terms else total

    size = batch_size if chunk_size is None or int(chunk_size) <= 0 else int(chunk_size)
    total_chunks: list[torch.Tensor] = []
    term_chunks: dict[str, list[torch.Tensor]] = {}
    logabs_chunks: list[torch.Tensor] = []
    sign_chunks: list[torch.Tensor] = []
    term_order: tuple[str, ...] | None = None
    captured_wavefunction = False
    for start in range(0, batch_size, size):
        chunk = slice_flat_batch(flat, start, min(start + size, batch_size))
        result = local_energy(terms, wavefunction, chunk, return_terms=return_terms)
        if return_terms:
            if not isinstance(result, LocalEnergyResult):
                raise TypeError("local_energy(return_terms=True) must return LocalEnergyResult")
            chunk_terms = tuple(result.terms)
            if term_order is None:
                term_order = chunk_terms
            elif chunk_terms != term_order:
                raise ValueError("chunked local-energy terms changed between chunks")
            total_chunks.append(result.total.detach())
            for name, value in result.terms.items():
                term_chunks.setdefault(name, []).append(value.detach())
            output = result.wavefunction_output
            if output is not None:
                captured_wavefunction = True
                # Keep only the signed-log primitives. ``aux`` and the original
                # output may retain the derivative graph used by the kinetic
                # term, so neither may escape the current chunk.
                logabs_chunks.append(output.logabs.detach().reshape(-1))
                sign_chunks.append(output.sign.detach().reshape(-1))
        else:
            if not isinstance(result, torch.Tensor):
                raise TypeError("local_energy(return_terms=False) must return a torch.Tensor")
            total_chunks.append(result.detach())
    total = torch.cat(total_chunks, dim=0)
    if not return_terms:
        return total
    wavefunction_output = None
    if captured_wavefunction:
        if len(logabs_chunks) != len(total_chunks):
            raise ValueError("wavefunction output was not produced for every local-energy chunk")
        wavefunction_output = WavefunctionOutput(
            logabs=torch.cat(logabs_chunks, dim=0),
            sign=torch.cat(sign_chunks, dim=0),
        )
    return LocalEnergyResult(
        total=total,
        terms={name: torch.cat(chunks, dim=0) for name, chunks in term_chunks.items()},
        wavefunction_output=wavefunction_output,
    )


def slice_flat_batch(batch: ElectronBatch, start: int, end: int) -> ElectronBatch:
    """Slice a flattened electron batch along its sample axis."""

    positions = batch.positions[start:end]
    spins = None if batch.spins is None else batch.spins[start:end]
    nuclear_positions = batch.nuclear_positions
    if nuclear_positions is not None and nuclear_positions.ndim == 3 and nuclear_positions.shape[0] == batch.batch_size:
        nuclear_positions = nuclear_positions[start:end]
    nuclear_charges = batch.nuclear_charges
    if nuclear_charges is not None and nuclear_charges.ndim == 2 and nuclear_charges.shape[0] == batch.batch_size:
        nuclear_charges = nuclear_charges[start:end]
    aux: dict[str, Any] = {}
    for key, value in batch.aux.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch.batch_size,):
            aux[key] = value[start:end]
        else:
            aux[key] = value
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=spins,
        aux=aux,
    )


__all__ = [
    "LocalEnergyCalculator",
    "evaluate_local_energy_in_chunks",
    "slice_flat_batch",
    "split_local_energy_result",
]
