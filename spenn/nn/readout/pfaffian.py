"""Differentiable Pfaffian readout."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.feature_dict import FeatureDict
from spenn.data.irrep_tensor import scalar_channels_last
from spenn.data.partitions import Par
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
        assert matrix.shape[-1] == matrix.shape[-2]
        return _pfaffian_single(matrix)
    if matrix.ndim != 3:
        raise ValueError(f"Expected matrix rank 2 or 3, got shape {tuple(matrix.shape)}")
    assert matrix.shape[-1] == matrix.shape[-2]
    output = torch.stack([_pfaffian_single(m) for m in matrix], dim=0)
    assert output.shape == (matrix.shape[0],)
    return output


def pfaffian_logabs_sign(matrix: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed-log representation of a Pfaffian."""

    value = pfaffian(matrix)
    sign = torch.sign(value)
    logabs = torch.where(sign == 0, torch.full_like(value, -torch.inf), 0.5 * torch.log(value.square().clamp_min(eps)))
    assert logabs.shape == sign.shape == value.shape
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
    assert logabs.shape == sign.shape
    if weights is None:
        weights = torch.ones(len(outputs), device=logabs.device, dtype=logabs.dtype) / len(outputs)
    weights = weights.to(device=logabs.device, dtype=logabs.dtype)
    total = (weights[:, None] * sign * torch.exp(logabs)).sum(dim=0)
    final_sign = torch.sign(total)
    final_logabs = torch.where(
        final_sign == 0,
        torch.full_like(total, -torch.inf),
        0.5 * torch.log(total.square().clamp_min(eps)),
    )
    assert final_logabs.shape == final_sign.shape == outputs[0].logabs.shape
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
        carrier = features.get(Par("A"))
        if carrier is None:
            raise KeyError("PfaffianReadout requires pair antisymmetric features at features[2][(1,1)]")
        one_body = features.get(Par("H"))
        one_body_features = None if one_body is None else scalar_channels_last(one_body)
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
            if one_body_features is not None:
                self.border_projections[-1](one_body_features)
        self._built = True

    def build_skew_kernel(self, features: FeatureDict, batch: ElectronBatch) -> torch.Tensor:
        """Construct one or more skew kernels."""

        self._ensure_heads(features)
        carrier = features.get(Par("A"))
        gate = features.get(Par("S"))
        one_body = features.get(Par("H"))
        if carrier is None:
            raise KeyError("Missing pair antisymmetric carrier features")
        kernels = []
        for idx in range(self.num_pfaffians):
            carrier_features = scalar_channels_last(carrier)
            assert carrier_features.shape[:3] == (batch.batch_size, batch.n_electrons, batch.n_electrons)
            carrier_scalar = self.carrier_projections[idx](carrier_features).squeeze(-1)
            carrier_scalar = antisymmetrize_pair_tensor(carrier_scalar)
            assert carrier_scalar.shape == (batch.batch_size, batch.n_electrons, batch.n_electrons)
            assert torch.allclose(carrier_scalar, -carrier_scalar.transpose(-1, -2))
            if self.use_symmetric_gates:
                if gate is not None:
                    gate_features = scalar_channels_last(gate)
                    assert gate_features.shape[:3] == (batch.batch_size, batch.n_electrons, batch.n_electrons)
                    gate_scalar = self.gate_projections[idx](gate_features).squeeze(-1)
                    gate_scalar = symmetrize_pair_tensor(gate_scalar)
                    assert torch.allclose(gate_scalar, gate_scalar.transpose(-1, -2))
                else:
                    gate_scalar = torch.ones_like(carrier_scalar)
            else:
                gate_scalar = torch.ones_like(carrier_scalar)
            kernel = carrier_scalar * gate_scalar
            if batch.positions.shape[1] % 2 == 1 and self.allow_odd_electron_bordered:
                if one_body is None:
                    raise KeyError("Odd-electron Pfaffian readout requires one-body features at features[1][(1)]")
                one_body_features = scalar_channels_last(one_body)
                assert one_body_features.shape[:2] == (batch.batch_size, batch.n_electrons)
                border_vec = self.border_projections[idx](one_body_features).squeeze(-1)
                assert border_vec.shape == (batch.batch_size, batch.n_electrons)
                n = kernel.shape[-1]
                bordered = kernel.new_zeros(kernel.shape[0], n + 1, n + 1)
                bordered[:, :n, :n] = kernel
                bordered[:, :n, n] = border_vec
                bordered[:, n, :n] = -border_vec
                kernel = bordered
            assert kernel.shape[0] == batch.batch_size
            assert kernel.shape[-1] == kernel.shape[-2]
            assert torch.allclose(kernel, -kernel.transpose(-1, -2))
            kernels.append(kernel)
        if self.num_pfaffians == 1:
            return kernels[0]
        output = torch.stack(kernels, dim=1)
        assert output.shape[:2] == (batch.batch_size, self.num_pfaffians)
        return output

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        kernels = self.build_skew_kernel(features, batch)
        if kernels.ndim == 3:
            logabs, sign = pfaffian_logabs_sign(kernels, eps=self.eps)
            assert logabs.shape == sign.shape == (batch.batch_size,)
            return WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels, "pfaffians": pfaffian(kernels)})

        outputs = []
        for idx in range(kernels.shape[1]):
            logabs, sign = pfaffian_logabs_sign(kernels[:, idx], eps=self.eps)
            outputs.append(WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels[:, idx]}))
        weights = torch.softmax(self.weight_logits, dim=0) if self.learn_weights else None
        out = signed_logsumexp_outputs(outputs, weights=weights, eps=self.eps)
        assert out.logabs.shape == out.sign.shape == (batch.batch_size,)
        out.aux["K"] = kernels
        return out
