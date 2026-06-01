"""Differentiable Pfaffian readout."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

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
    """Compute Pfaffians for skew-symmetric matrices.

    Parameters
    ----------
    matrix : torch.Tensor
        Skew-symmetric matrix with shape ``[n, n]`` or batched matrices with
        shape ``[batch, n, n]``.

    Returns
    -------
    torch.Tensor
        Scalar Pfaffian for an unbatched input, or one Pfaffian per batch item
        with shape ``[batch]``.
    """

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
    """Return the signed-log representation of a Pfaffian.

    Parameters
    ----------
    matrix : torch.Tensor
        Skew-symmetric matrix with shape ``[n, n]`` or batched matrices with
        shape ``[batch, n, n]``.
    eps : float, optional
        Positive floor for the squared Pfaffian magnitude.

    Returns
    -------
    tuple of torch.Tensor
        Log absolute value and sign tensors. For batched inputs both have shape
        ``[batch]``.
    """

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
    """Combine signed-log wavefunction outputs.

    Parameters
    ----------
    outputs : sequence of WavefunctionOutput
        Component signed-log outputs with matching ``logabs`` and ``sign``
        shapes.
    weights : torch.Tensor or None, optional
        Mixture weights with shape ``[n_outputs]``. If ``None``, uniform
        weights are used.
    eps : float, optional
        Positive floor for the squared signed sum magnitude.

    Returns
    -------
    WavefunctionOutput
        Signed-log output representing the weighted real-space sum.
    """

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
    """Build skew kernels and convert them into a signed-log output.

    Parameters
    ----------
    num_pfaffians : int, optional
        Number of Pfaffian components to combine.
    use_symmetric_gates : bool, optional
        Whether to multiply antisymmetric carrier entries by symmetric gates.
    allow_odd_electron_bordered : bool, optional
        Whether odd-electron systems use a bordered skew matrix.
    learn_weights : bool, optional
        Whether multiple Pfaffian components use trainable mixture weights.
    eps : float, optional
        Positive floor used in signed-log conversion.
    envelope_coefficient : float, optional
        Nonnegative coefficient for the harmonic one-body envelope
        ``-envelope_coefficient * sum_i |r_i|^2`` added to the log-amplitude.
    trainable_envelope : bool, optional
        Whether to optimize `envelope_coefficient` through a positive softplus
        parameterization.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        num_pfaffians: int = 1,
        use_symmetric_gates: bool = True,
        allow_odd_electron_bordered: bool = True,
        learn_weights: bool = True,
        eps: float = 1e-12,
        envelope_coefficient: float = 0.0,
        trainable_envelope: bool = False,
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
        _configure_envelope(self, envelope_coefficient, trainable=trainable_envelope)

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
        """Construct one or more Pfaffian skew kernels.

        Parameters
        ----------
        features : FeatureDict
            Feature dictionary containing pair-antisymmetric carrier features,
            optional pair-symmetric gates, and optional one-body features.
        batch : ElectronBatch
            Flattened electron batch used for shape checks and odd-electron
            bordered kernels.

        Returns
        -------
        torch.Tensor
            Skew kernel with shape ``[batch, n, n]`` for one component, or
            ``[batch, num_pfaffians, n, n]`` for multiple components.
        """

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
        """Return a signed-log Pfaffian readout.

        Parameters
        ----------
        features : FeatureDict
            SpENN features used to build the skew kernels.
        batch : ElectronBatch
            Electron batch with positions shaped ``[batch, n_electrons,
            spatial_dim]`` after sample flattening.

        Returns
        -------
        WavefunctionOutput
            Signed-log wavefunction output with shape ``[batch]``.
        """

        kernels = self.build_skew_kernel(features, batch)
        if kernels.ndim == 3:
            logabs, sign = pfaffian_logabs_sign(kernels, eps=self.eps)
            envelope = _harmonic_envelope(self, batch)
            logabs = logabs + envelope
            assert logabs.shape == sign.shape == (batch.batch_size,)
            return WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels, "pfaffians": pfaffian(kernels), "envelope": envelope})

        outputs = []
        for idx in range(kernels.shape[1]):
            logabs, sign = pfaffian_logabs_sign(kernels[:, idx], eps=self.eps)
            outputs.append(WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": kernels[:, idx]}))
        weights = torch.softmax(self.weight_logits, dim=0) if self.learn_weights else None
        out = signed_logsumexp_outputs(outputs, weights=weights, eps=self.eps)
        envelope = _harmonic_envelope(self, batch)
        out = WavefunctionOutput(logabs=out.logabs + envelope, sign=out.sign, aux=dict(out.aux))
        assert out.logabs.shape == out.sign.shape == (batch.batch_size,)
        out.aux["K"] = kernels
        out.aux["envelope"] = envelope
        return out


def _configure_envelope(module: nn.Module, coefficient: float, *, trainable: bool) -> None:
    if coefficient < 0.0:
        raise ValueError(f"envelope_coefficient must be nonnegative, got {coefficient}")
    if trainable:
        raw = _inverse_softplus(float(coefficient))
        module.register_parameter("envelope_raw", nn.Parameter(torch.tensor(raw, dtype=torch.float64)))
        module.register_buffer("envelope_coefficient", torch.empty(0, dtype=torch.float64), persistent=False)
    else:
        module.register_parameter("envelope_raw", None)
        module.register_buffer("envelope_coefficient", torch.tensor(float(coefficient), dtype=torch.float64))


def _inverse_softplus(value: float) -> float:
    if value == 0.0:
        return -50.0
    return math.log(math.expm1(value))


def _current_envelope_coefficient(module: nn.Module, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    raw = getattr(module, "envelope_raw", None)
    if raw is not None:
        return F.softplus(raw).to(device=device, dtype=dtype)
    return module.envelope_coefficient.to(device=device, dtype=dtype)


def _harmonic_envelope(module: nn.Module, batch: ElectronBatch) -> torch.Tensor:
    coefficient = _current_envelope_coefficient(module, dtype=batch.dtype, device=batch.device)
    radius_squared = batch.positions.square().sum(dim=(1, 2))
    assert radius_squared.shape == (batch.batch_size,)
    output = -coefficient * radius_squared
    assert output.shape == (batch.batch_size,)
    return output
