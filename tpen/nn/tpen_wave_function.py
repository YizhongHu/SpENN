"""Composed TPEN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from tpen.data.batch import ElectronBatch, FactorizedLocalEnergyInput, WavefunctionOutput
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.context import TPENForwardContext
from tpen.nn.cusp import ElectronNucleusCusp
from tpen.nn.tpen_stack import TPENStack

torch = require_torch(feature="TPEN wavefunction modules")
nn = require_torch_nn(feature="TPEN wavefunction modules")


class TPENWaveFunction(EquivariantMap):
    """Compose basis, embedding, a TPEN layer stack, readout, and post-readout factors.

    The full pipeline is::

        ElectronBatch
          -> ElectronBasis (optional)
          -> ElectronBasisFeatures
          -> embedding
          -> TPENStack (TPEN layers)
          -> readout
          -> + envelope (legacy, optional)
          -> + sum(factors) (generic, optional)

    ``envelope`` and each entry of ``factors`` are additive post-readout
    log-amplitude contributions: every one of them accepts an
    :class:`ElectronBatch` and returns a ``[batch]``-shaped tensor (the
    contract shared by both `tpen.nn.envelope.Envelope` and
    `tpen.nn.factor.LogAmplitudeFactor`). There is no mutual exclusion
    between them -- they compose in one pipeline, and either (or both) may be
    omitted for a bare readout output.

    The raw :class:`ElectronBatch` is still passed to the readout and every
    factor so they see true coordinates; the basis only re-represents the
    per-particle input to the embedding.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping the basis output (or, when ``basis`` is ``None``, an
        :class:`ElectronBatch`) to :class:`tpen.data.real.Feature`.
    layers : iterable of torch.nn.Module or TPENStack
        TPEN layers, or an already-constructed :class:`TPENStack`. Iterables
        are wrapped into a stack; the layers always live in ``self.stack``.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    envelope : torch.nn.Module or None, optional
        Legacy additive log-amplitude envelope (e.g. `tpen.nn.AdditiveEnvelope`).
    factors : iterable of torch.nn.Module, optional
        Generic additive post-readout log-amplitude factors (e.g.
        `tpen.nn.ElectronNucleusCusp`, `tpen.nn.AdditiveCusp`).
    basis : torch.nn.Module or None, optional
        Optional :class:`tpen.nn.ElectronBasis` applied before the embedding.
        When ``None``, the embedding consumes the raw :class:`ElectronBatch`.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] | TPENStack = (),
        readout: nn.Module,
        envelope: nn.Module | None = None,
        factors: Iterable[nn.Module] = (),
        basis: nn.Module | None = None,
        analytic_cusp_provider: ElectronNucleusCusp | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.basis = basis
        self.embedding = embedding
        self.stack = layers if isinstance(layers, TPENStack) else TPENStack(layers)
        self.readout = readout
        self.envelope = envelope
        factors = tuple(factors)
        self.factors = nn.ModuleList(factors)
        if analytic_cusp_provider is not None:
            if not isinstance(analytic_cusp_provider, ElectronNucleusCusp):
                raise TypeError("analytic_cusp_provider must be an ElectronNucleusCusp")
            occurrences = sum(factor is analytic_cusp_provider for factor in factors)
            if occurrences != 1:
                raise ValueError(
                    "analytic_cusp_provider must be the unique participating ElectronNucleusCusp factor"
                )
        self.analytic_cusp_provider = analytic_cusp_provider

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        output = self._construct_output(batch, include_analytic_cusp=True)
        return output

    def factorized_local_energy_input(self, batch: ElectronBatch) -> FactorizedLocalEnergyInput:
        """Return the regular output and analytic data for local-energy evaluation.

        The explicitly bound cusp is omitted while constructing the regular
        output, then queried once from that same live factor instance.
        """

        provider = self.analytic_cusp_provider
        if provider is None:
            raise ValueError("factorized local-energy input requires an analytic_cusp_provider at construction")
        regular = self._construct_output(batch, include_analytic_cusp=False)
        evaluation = provider.analytic_evaluation(batch)
        return FactorizedLocalEnergyInput(regular, evaluation)

    def _construct_output(self, batch: ElectronBatch, *, include_analytic_cusp: bool) -> WavefunctionOutput:
        output = self._readout_output(batch)
        logabs = output.logabs
        if self.envelope is not None:
            logabs = logabs + _log_factor(self.envelope, batch, logabs.shape, name="Envelope")
        for index, factor in enumerate(self.factors):
            if not include_analytic_cusp and factor is self.analytic_cusp_provider:
                continue
            logabs = logabs + _log_factor(factor, batch, logabs.shape, name=f"factors[{index}]")
        return WavefunctionOutput(logabs=logabs, sign=output.sign, phase=output.phase, aux=dict(output.aux))

    def _readout_output(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Build readout output once for the public evaluation path."""

        basis_features = self.basis(batch) if self.basis is not None else None
        context = TPENForwardContext(batch=batch, basis_features=basis_features)
        embedded_input = basis_features if basis_features is not None else batch
        features = self.embedding(embedded_input, context=context)
        features = self.stack(features, context)
        return self.readout(features, batch)


def _log_factor(module: nn.Module, batch: ElectronBatch, shape: torch.Size, *, name: str) -> torch.Tensor:
    value = module(batch)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    if value.shape != shape:
        raise ValueError(f"{name} output must have shape {tuple(shape)}, got {tuple(value.shape)}")
    return value


__all__ = ["TPENWaveFunction"]
