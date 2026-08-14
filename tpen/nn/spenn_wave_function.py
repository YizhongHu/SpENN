"""Composed TPEN wavefunction scaffold."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.permutation import Permutation
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.context import TPENForwardContext
from tpen.nn.envelope import NuclearConfinementEvaluation, NuclearFactorizedEnvelope
from tpen.nn.tpen_stack import TPENStack

torch = require_torch(feature="TPEN wavefunction modules")
nn = require_torch_nn(feature="TPEN wavefunction modules")


@dataclass(frozen=True)
class NuclearFactorizedWavefunctionParts:
    """Typed factorization of an atom wavefunction output.

    The readout and all non-nuclear envelope contributions form
    ``regular_logabs``.  The analytic nuclear factor remains separate with its
    radial derivatives for the local-energy implementation.
    """

    regular_logabs: torch.Tensor
    nuclear: NuclearConfinementEvaluation
    sign: torch.Tensor
    phase: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def validate(self, batch: ElectronBatch) -> "NuclearFactorizedWavefunctionParts":
        """Validate output and factorization shape semantics."""

        flat = batch.flatten_samples()
        expected = (flat.batch_size,)
        if self.regular_logabs.shape != expected or self.sign.shape != expected:
            raise ValueError(f"NuclearFactorizedWavefunctionParts scalar fields must have shape {expected}")
        if self.phase is not None and self.phase.shape != expected:
            raise ValueError(f"NuclearFactorizedWavefunctionParts.phase must have shape {expected}")
        for name, value in (("regular_logabs", self.regular_logabs), ("sign", self.sign)):
            if value.device != flat.device or value.dtype != flat.dtype:
                raise ValueError(f"NuclearFactorizedWavefunctionParts.{name} must match batch dtype/device")
        if self.phase is not None and (self.phase.device != flat.device or self.phase.dtype != flat.dtype):
            raise ValueError("NuclearFactorizedWavefunctionParts.phase must match batch dtype/device")
        self.nuclear.validate(flat)
        self.as_output().validate(batch_size=flat.batch_size)
        return self

    def permute(self, permutation: Permutation) -> "NuclearFactorizedWavefunctionParts":
        """Apply fermionic parity while permuting nuclear pair fields."""

        return type(self)(
            regular_logabs=self.regular_logabs.clone(),
            nuclear=self.nuclear.permute(permutation),
            sign=self.sign * permutation.sign,
            phase=None if self.phase is None else self.phase.clone(),
            aux=dict(self.aux),
        )

    def compare(
        self, other: "NuclearFactorizedWavefunctionParts", *, atol: float = 1.0e-6, rtol: float = 1.0e-6
    ) -> tuple[bool, dict[str, float]]:
        """Compare physical scalar fields and explicit nuclear derivatives."""

        if type(self) is not type(other) or (self.phase is None) != (other.phase is None):
            return False, {"max_abs_error": float("inf")}
        scalar_ok, scalar_metrics = compare_tensor_blocks(
            [self.regular_logabs, self.sign] + ([] if self.phase is None else [self.phase]),
            [other.regular_logabs, other.sign] + ([] if other.phase is None else [other.phase]),
            atol=atol,
            rtol=rtol,
        )
        nuclear_ok, nuclear_metrics = self.nuclear.compare(other.nuclear, atol=atol, rtol=rtol)
        return scalar_ok and nuclear_ok, {"max_abs_error": max(scalar_metrics["max_abs_error"], nuclear_metrics["max_abs_error"])}

    def as_output(self) -> WavefunctionOutput:
        """Materialize a standard output without aliasing mutable aux data."""

        return WavefunctionOutput(
            logabs=self.regular_logabs + self.nuclear.value.sum(dim=(1, 2)),
            sign=self.sign,
            phase=self.phase,
            aux=dict(self.aux),
        )


class TPENWaveFunction(EquivariantMap):
    """Compose basis, embedding, a TPEN layer stack, readout, and an envelope.

    The full pipeline is::

        ElectronBatch
          -> ElectronBasis (optional)
          -> ElectronBasisFeatures
          -> embedding
          -> TPENStack (TPEN layers)
          -> readout
          -> + additive log-amplitude envelope

    The raw :class:`ElectronBatch` is still passed to the readout and envelope so
    they see true coordinates; the basis only re-represents the per-particle
    input to the embedding.

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
        Legacy non-nuclear additive log-amplitude envelope.
    nuclear_envelope : NuclearFactorizedEnvelope or None, optional
        Explicit atomic envelope construction. Exactly one of ``envelope`` and
        ``nuclear_envelope`` must be supplied.
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
        nuclear_envelope: NuclearFactorizedEnvelope | None = None,
        basis: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if (envelope is None) == (nuclear_envelope is None):
            raise ValueError("TPENWaveFunction requires exactly one of envelope or nuclear_envelope")
        self.basis = basis
        self.embedding = embedding
        self.stack = layers if isinstance(layers, TPENStack) else TPENStack(layers)
        self.readout = readout
        self.envelope = envelope
        self.nuclear_envelope = nuclear_envelope

    def forward_impl(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the signed-log wavefunction for an electron batch."""

        if self.nuclear_envelope is not None:
            return self.nuclear_factorization(batch).as_output()
        output = self._readout_output(batch)
        assert self.envelope is not None
        logabs = output.logabs + _log_factor(self.envelope, batch, output.logabs.shape, name="Envelope")
        return WavefunctionOutput(logabs=logabs, sign=output.sign, phase=output.phase, aux=dict(output.aux))

    def nuclear_factorization(self, batch: ElectronBatch) -> NuclearFactorizedWavefunctionParts:
        """Return the explicit atom factorization for a nuclear model."""

        if self.nuclear_envelope is None:
            raise ValueError("TPENWaveFunction has no NuclearFactorizedEnvelope")
        output = self._readout_output(batch)
        regular = output.logabs + _log_factor(
            self.nuclear_envelope.regular_envelope,
            batch,
            output.logabs.shape,
            name="regular_envelope",
        )
        return NuclearFactorizedWavefunctionParts(
            regular_logabs=regular,
            nuclear=self.nuclear_envelope.nuclear_confinement.evaluate(batch),
            sign=output.sign,
            phase=output.phase,
            aux=dict(output.aux),
        ).validate(batch)

    def _readout_output(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Build readout output once for either public evaluation path."""

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


__all__ = ["NuclearFactorizedWavefunctionParts", "TPENWaveFunction"]
