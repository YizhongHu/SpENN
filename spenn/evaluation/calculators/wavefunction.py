"""Wavefunction evaluation calculator."""

from __future__ import annotations

from dataclasses import replace

import torch

from spenn.data.batch import WavefunctionOutput
from spenn.evaluation.bundle import EvaluationBundle, WavefunctionValues
from spenn.evaluation.calculators.local_energy import slice_flat_batch
from spenn.evaluation.protocols import EvaluationContext


class WavefunctionCalculator:
    """Evaluate a model and store signed-log wavefunction values."""

    name = "wavefunction"

    def __init__(self, *, chunk_size: int | None = None, return_components: bool = False) -> None:
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.return_components = bool(return_components)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Evaluate wavefunction outputs without retaining autograd graphs."""

        flat = bundle.generated.batch.flatten_samples()
        size = flat.batch_size if self.chunk_size is None or self.chunk_size <= 0 else self.chunk_size
        logabs_chunks: list[torch.Tensor] = []
        sign_chunks: list[torch.Tensor] = []
        component_chunks: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for start in range(0, flat.batch_size, size):
                chunk = slice_flat_batch(flat, start, min(start + size, flat.batch_size))
                output = model(chunk)
                if not isinstance(output, WavefunctionOutput):
                    raise TypeError(f"wavefunction model must return WavefunctionOutput, got {type(output)!r}")
                logabs_chunks.append(output.logabs.detach().reshape(-1))
                sign_chunks.append(output.sign.detach().reshape(-1))
                if self.return_components:
                    for name, value in output.aux.items():
                        if isinstance(value, torch.Tensor):
                            component_chunks.setdefault(name, []).append(value.detach())
        components = None
        if self.return_components:
            components = {name: torch.cat(chunks, dim=0) for name, chunks in component_chunks.items()}
        values = WavefunctionValues(
            logabs=torch.cat(logabs_chunks, dim=0),
            sign=torch.cat(sign_chunks, dim=0),
            components=components,
        )
        return replace(bundle, wavefunction=values)


__all__ = ["WavefunctionCalculator"]
