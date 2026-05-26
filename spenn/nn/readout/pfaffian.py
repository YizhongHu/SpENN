"""Differentiable Pfaffian readout."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch, WavefunctionOutput
from spenn.data_structures.feature_dict import FeatureDict
from spenn.utils.tensor_utils import antisymmetrize_pair_tensor, symmetrize_pair_tensor


def _pfaffian_single(matrix: torch.Tensor) -> torch.Tensor:
    """Compute a Pfaffian via recursive expansion.

    This is intentionally simple and exact enough for small phase-1 tests.
    """

    n = matrix.shape[-1]
    if n == 0:
        return matrix.new_tensor(1.0)
    if n == 2:
        return matrix[0, 1]
    if n % 2 == 1:
        return matrix.new_tensor(0.0)

    total = matrix.new_tensor(0.0)
    remaining = torch.arange(n, device=matrix.device)
    for j in range(1, n):
        sign = 1.0 if j % 2 == 1 else -1.0
        idx = remaining[(remaining != 0) & (remaining != j)]
        submatrix = matrix.index_select(0, idx).index_select(1, idx)
        total = total + sign * matrix[0, j] * _pfaffian_single(submatrix)
    return total


def pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    """Compute Pfaffians for ``[batch, n, n]`` or ``[n, n]`` tensors."""

    if matrix.ndim == 2:
        return _pfaffian_single(matrix)
    if matrix.ndim != 3:
        raise ValueError(f"Expected matrix rank 2 or 3, got shape {tuple(matrix.shape)}")
    return torch.stack([_pfaffian_single(m) for m in matrix], dim=0)


def pfaffian_logabs_sign(matrix: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed-log representation of a Pfaffian."""

    value = pfaffian(matrix)
    sign = torch.sign(value)
    logabs = 0.5 * torch.log(value.square().clamp_min(eps))
    return logabs, sign


def signed_logsumexp_outputs(
    outputs: Sequence[WavefunctionOutput],
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> WavefunctionOutput:
    """Combine signed-log outputs by evaluating the signed sum in real space."""

    if not outputs:
        raise ValueError("Need at least one output to combine")
    logabs = torch.stack([out.logabs for out in outputs], dim=0)
    sign = torch.stack([out.sign for out in outputs], dim=0)
    if weights is None:
        weights = torch.ones(len(outputs), device=logabs.device, dtype=logabs.dtype) / len(outputs)
    weights = weights.to(device=logabs.device, dtype=logabs.dtype)
    total = (weights[:, None] * sign * torch.exp(logabs)).sum(dim=0)
    final_sign = torch.sign(total)
    final_logabs = 0.5 * torch.log(total.square().clamp_min(eps))
    return WavefunctionOutput(logabs=final_logabs, sign=final_sign, aux={"components": outputs, "weights": weights})


class PfaffianReadout(nn.Module):
    """Build one or more skew kernels and convert them into a signed-log output."""

    def __init__(
        self,
        num_pfaffians: int = 1,
        use_symmetric_gates: bool = True,
        allow_odd_electron_bordered: bool = True,
        learn_weights: bool = True,
        eps: float = 1e-12,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_pfaffians = num_pfaffians
        self.use_symmetric_gates = use_symmetric_gates
        self.allow_odd_electron_bordered = allow_odd_electron_bordered
        self.eps = eps
        self.learn_weights = learn_weights
        self.carrier_projections = nn.ModuleList()
        self.gate_projections = nn.ModuleList()
        self.border_projections = nn.ModuleList()
        self._built = False
        if learn_weights and num_pfaffians > 1:
            self.weight_logits = nn.Parameter(torch.zeros(num_pfaffians))
        else:
            self.register_buffer("weight_logits", torch.zeros(num_pfaffians), persistent=False)

    def _ensure_heads(self, features: FeatureDict) -> None:
        if self._built:
            return
        carrier = features.get(2, (1, 1))
        if carrier is None:
            raise KeyError("PfaffianReadout requires pair antisymmetric features at features[2][(1,1)]")
        for _ in range(self.num_pfaffians):
            self.carrier_projections.append(
                nn.LazyLinear(1, bias=False, dtype=torch.float64).to(device=carrier.device, dtype=carrier.dtype)
            )
            self.gate_projections.append(
                nn.LazyLinear(1, bias=True, dtype=torch.float64).to(device=carrier.device, dtype=carrier.dtype)
            )
            self.border_projections.append(
                nn.LazyLinear(1, bias=False, dtype=torch.float64).to(device=carrier.device, dtype=carrier.dtype)
            )
        self._built = True

    def build_skew_kernel(self, features: FeatureDict, batch: ElectronBatch) -> torch.Tensor:
        """Construct one or more skew kernels."""

        self._ensure_heads(features)
        carrier = features.get(2, (1, 1))
        gate = features.get(2, (2))
        one_body = features.get(1, (1))
        if carrier is None:
            raise KeyError("Missing pair antisymmetric carrier features")
        kernels = []
        for idx in range(self.num_pfaffians):
            carrier_scalar = self.carrier_projections[idx](carrier).squeeze(-1)
            carrier_scalar = antisymmetrize_pair_tensor(carrier_scalar)
            if self.use_symmetric_gates:
                if gate is not None:
                    gate_scalar = self.gate_projections[idx](gate).squeeze(-1)
                    gate_scalar = symmetrize_pair_tensor(gate_scalar)
                else:
                    gate_scalar = torch.ones_like(carrier_scalar)
            else:
                gate_scalar = torch.ones_like(carrier_scalar)
            kernel = carrier_scalar * gate_scalar
            if batch.positions.shape[1] % 2 == 1 and self.allow_odd_electron_bordered:
                if one_body is None:
                    raise KeyError("Odd-electron Pfaffian readout requires one-body features at features[1][(1)]")
                border_vec = self.border_projections[idx](one_body).squeeze(-1)
                n = kernel.shape[-1]
                bordered = kernel.new_zeros(kernel.shape[0], n + 1, n + 1)
                bordered[:, :n, :n] = kernel
                bordered[:, :n, n] = border_vec
                bordered[:, n, :n] = -border_vec
                kernel = bordered
            kernels.append(kernel)
        if self.num_pfaffians == 1:
            return kernels[0]
        return torch.stack(kernels, dim=1)

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        kernels = self.build_skew_kernel(features, batch)
        if kernels.ndim == 3:
            logabs, sign = pfaffian_logabs_sign(kernels, eps=self.eps)
            return WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels, "pfaffians": pfaffian(kernels)})

        outputs = []
        for idx in range(kernels.shape[1]):
            logabs, sign = pfaffian_logabs_sign(kernels[:, idx], eps=self.eps)
            outputs.append(WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels[:, idx]}))
        weights = torch.softmax(self.weight_logits, dim=0) if self.learn_weights else None
        out = signed_logsumexp_outputs(outputs, weights=weights, eps=self.eps)
        out.aux["K"] = kernels
        return out
