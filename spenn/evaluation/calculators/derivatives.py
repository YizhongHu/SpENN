"""Derivative calculators for deterministic evaluation tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

import torch

from spenn.data.batch import ElectronBatch
from spenn.evaluation.bundle import DerivativeValues, EvaluationBundle
from spenn.evaluation.calculators.local_energy import slice_flat_batch
from spenn.evaluation.protocols import EvaluationContext


class RadialLogAbsDerivativeCalculator:
    """Compute ``d log|psi| / d r12`` for two-electron relative coordinates."""

    name = "radial_logabs_derivative"

    def __init__(
        self,
        *,
        coordinate: Literal["r12"] = "r12",
        create_graph: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        if coordinate != "r12":
            raise ValueError("RadialLogAbsDerivativeCalculator currently supports coordinate='r12' only")
        self.coordinate = coordinate
        self.create_graph = bool(create_graph)
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Compute radial logabs derivatives and return an updated bundle."""

        del context
        flat = bundle.generated.batch.flatten_samples()
        if flat.n_electrons != 2:
            raise ValueError("RadialLogAbsDerivativeCalculator requires exactly two electrons")
        size = flat.batch_size if self.chunk_size is None or self.chunk_size <= 0 else self.chunk_size
        chunks: list[torch.Tensor] = []
        for start in range(0, flat.batch_size, size):
            chunk = slice_flat_batch(flat, start, min(start + size, flat.batch_size))
            chunks.append(_radial_derivative_chunk(model, chunk, create_graph=self.create_graph))
        radial = torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=flat.device, dtype=flat.dtype)
        metadata = bundle.generated.metadata
        values = DerivativeValues(
            radial_dlogabs=radial,
            r12=_metadata_tensor(metadata, "r12", radial),
            direction_id=_metadata_long(metadata, "direction_id", radial),
            antipodal_pair_id=_metadata_optional_long(metadata, "antipodal_pair_id", radial),
            direction_sign=_metadata_optional_long(metadata, "direction_sign", radial),
        )
        derivatives = dict(bundle.derivatives or {})
        derivatives[self.coordinate] = values
        return replace(bundle, derivatives=derivatives)


def _radial_derivative_chunk(model: torch.nn.Module, batch: ElectronBatch, *, create_graph: bool) -> torch.Tensor:
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
    logabs = output.logabs.reshape(-1)
    gradient = torch.autograd.grad(
        logabs.sum(),
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    relative = positions[:, 0, :] - positions[:, 1, :]
    direction = relative / torch.linalg.norm(relative, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return 0.5 * (direction * gradient[:, 0, :]).sum(dim=-1) - 0.5 * (direction * gradient[:, 1, :]).sum(dim=-1)


def _metadata_tensor(metadata: Mapping[str, object], key: str, like: torch.Tensor) -> torch.Tensor:
    value = metadata.get(key)
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype).reshape(-1)
    raise ValueError(f"RadialLogAbsDerivativeCalculator requires generated metadata {key!r}")


def _metadata_long(metadata: Mapping[str, object], key: str, like: torch.Tensor) -> torch.Tensor:
    value = metadata.get(key)
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=torch.long).reshape(-1)
    raise ValueError(f"RadialLogAbsDerivativeCalculator requires generated metadata {key!r}")


def _metadata_optional_long(metadata: Mapping[str, object], key: str, like: torch.Tensor) -> torch.Tensor | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=torch.long).reshape(-1)
    raise ValueError(f"generated metadata {key!r} must be a tensor when present")


__all__ = ["RadialLogAbsDerivativeCalculator"]
