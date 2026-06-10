"""Composed SpENN wavefunction scaffold."""

from __future__ import annotations

import inspect
from collections.abc import Iterable

from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.equivariance import EquivariantMap


class SpENNWaveFunction(EquivariantMap):
    """Compose embedding, SpENN layers, readout, and optional cusp.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping :class:`ElectronBatch` to
        :class:`spenn.data.real.RealFeature`.
    layers : iterable of torch.nn.Module
        Sequence of SpENN layers.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    cusp : torch.nn.Module or None, optional
        Optional additive log-amplitude cusp. A cusp may either accept
        ``(batch, output)`` and return a full output, or accept ``batch`` and
        return an additive tensor matching ``output.logabs``.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] = (),
        readout: nn.Module,
        cusp: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.layers = nn.ModuleList(tuple(layers))
        self.readout = readout
        self.cusp = cusp
        self._cusp_accepts_output = False if cusp is None else _cusp_accepts_output(cusp)

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        features = self.embedding(batch)
        for layer in self.layers:
            features = layer(features)
        output = self.readout(features, batch)
        if self.cusp is None:
            return output
        if self._cusp_accepts_output:
            cusp_output = self.cusp(batch, output)
        else:
            cusp_output = self.cusp(batch)
        if isinstance(cusp_output, WavefunctionOutput):
            return cusp_output
        if cusp_output.shape != output.logabs.shape:
            raise ValueError(
                f"Cusp output must have shape {tuple(output.logabs.shape)}, got {tuple(cusp_output.shape)}"
            )
        return WavefunctionOutput(
            logabs=output.logabs + cusp_output,
            sign=output.sign,
            phase=output.phase,
            aux=dict(output.aux),
        )


def _cusp_accepts_output(cusp: nn.Module) -> bool:
    """Return whether a cusp accepts the readout output as a second argument."""

    try:
        signature = inspect.signature(cusp.forward)
    except (TypeError, ValueError) as exc:
        raise TypeError("cusp must expose an inspectable forward signature") from exc

    parameters = tuple(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional) >= 2:
        return True
    if len(positional) == 1:
        return False
    raise TypeError("cusp forward must accept either (batch) or (batch, output)")


__all__ = ["SpENNWaveFunction"]
