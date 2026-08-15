"""Atom-owned primitive evaluation calculators."""

from __future__ import annotations

from dataclasses import replace

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.evaluation.bundle import ElectronNucleusRadialValues, EvaluationBundle
from tpen.evaluation.calculators.local_energy import slice_flat_batch
from tpen.evaluation.protocols import EvaluationContext


class ElectronNucleusRadialCalculator:
    """Compute raw ``d log|psi| / d r_iA`` values without a distance floor."""

    name = "electron_nucleus_radial"

    def __init__(self, *, chunk_size: int | None = None) -> None:
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Return an updated bundle with typed radial derivatives."""

        del context
        flat = bundle.generated.batch.flatten_samples()
        if flat.nuclear_positions is None or flat.nuclear_charges is None:
            raise ValueError("ElectronNucleusRadialCalculator requires batch nuclear positions and charges")
        size = flat.batch_size if self.chunk_size is None or self.chunk_size <= 0 else self.chunk_size
        distances: list[torch.Tensor] = []
        derivatives: list[torch.Tensor] = []
        for start in range(0, flat.batch_size, size):
            chunk = slice_flat_batch(flat, start, min(start + size, flat.batch_size))
            distance, derivative = _radial_chunk(model, chunk)
            distances.append(distance)
            derivatives.append(derivative)
        pair_shape = (0, flat.n_electrons, flat.nuclear_positions.shape[-2])
        distance = (
            torch.cat(distances, dim=0)
            if distances
            else torch.empty(pair_shape, device=flat.device, dtype=flat.dtype)
        )
        radial_dlogabs = (
            torch.cat(derivatives, dim=0)
            if derivatives
            else torch.empty(pair_shape, device=flat.device, dtype=flat.dtype)
        )
        values = ElectronNucleusRadialValues(
            distance=distance,
            radial_dlogabs=radial_dlogabs,
            finite_mask=torch.isfinite(radial_dlogabs),
        ).validate(flat)
        return replace(bundle, electron_nucleus_radial=values)


def _radial_chunk(
    model: torch.nn.Module,
    batch: ElectronBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = batch.positions.detach().clone().requires_grad_(True)
    work_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        spins=batch.spins,
        aux=dict(batch.aux),
    )
    output = model(work_batch)
    if not isinstance(output, WavefunctionOutput):
        raise TypeError("ElectronNucleusRadialCalculator requires the model to return WavefunctionOutput")
    output.validate(batch_size=work_batch.batch_size)
    gradient = torch.autograd.grad(
        output.logabs.sum(),
        positions,
        create_graph=False,
        retain_graph=False,
    )[0]
    displacement = electron_nuclear_displacements(work_batch)
    distance = displacement.norm(dim=-1)
    if torch.any(distance <= 0):
        raise ValueError(
            "ElectronNucleusRadialCalculator is undefined at exact electron-nucleus coalescence"
        )
    radial_direction = displacement / distance.unsqueeze(-1)
    radial_dlogabs = (gradient.unsqueeze(-2) * radial_direction).sum(dim=-1)
    return distance.detach(), radial_dlogabs.detach()


__all__ = ["ElectronNucleusRadialCalculator"]
