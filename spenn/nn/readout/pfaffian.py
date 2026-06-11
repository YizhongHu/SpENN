"""Differentiable Pfaffian readout for real tuple features.

All readouts in the new SpENN core consume :class:`spenn.data.real.RealFeature`.
Readout-specific Fourier transforms should happen inside a component readout
before it contributes to the final wavefunction.
"""

from __future__ import annotations

import math

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import RealFeature
from spenn.dependencies import require_torch, require_torch_functional, require_torch_nn

torch = require_torch(feature="Pfaffian readout")
nn = require_torch_nn(feature="Pfaffian readout")
F = require_torch_functional(feature="Pfaffian readout")


def _pfaffian_single(matrix: torch.Tensor) -> torch.Tensor:
    """Compute one Pfaffian by recursive expansion.

    This path is intentionally simple and suited to small scaffold tests. A
    production implementation should replace it with a stable batched routine.
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
    for col in range(1, n):
        sign = 1.0 if col % 2 == 1 else -1.0
        idx = remaining[(remaining != 0) & (remaining != col)]
        submatrix = matrix.index_select(0, idx).index_select(1, idx)
        total = total + sign * matrix[0, col] * _pfaffian_single(submatrix)
    return total


def pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    """Compute Pfaffians for skew-symmetric matrices.

    Parameters
    ----------
    matrix : torch.Tensor
        Matrix with shape ``[n, n]`` or batched matrices with shape
        ``[batch, n, n]``.

    Returns
    -------
    torch.Tensor
        Scalar Pfaffian for an unbatched input or shape ``[batch]`` for a
        batched input.
    """

    if matrix.ndim == 2:
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("Pfaffian matrix must be square")
        return _pfaffian_single(matrix)
    if matrix.ndim != 3:
        raise ValueError(f"Expected matrix rank 2 or 3, got shape {tuple(matrix.shape)}")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("Pfaffian matrices must be square")
    return torch.stack([_pfaffian_single(item) for item in matrix], dim=0)


def pfaffian_logabs_sign(matrix: torch.Tensor, eps: float = 1.0e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed-log Pfaffian values."""

    value = pfaffian(matrix)
    sign = torch.sign(value)
    logabs = torch.where(sign == 0, torch.full_like(value, -torch.inf), 0.5 * torch.log(value.square().clamp_min(eps)))
    return logabs, sign


class PfaffianReadout(nn.Module):
    """Build a skew matrix from order-2 real features and return a Pfaffian.

    Parameters
    ----------
    allow_odd_electron_bordered : bool, optional
        Whether odd-electron systems use an order-1 bordered Pfaffian.
    eps : float, optional
        Positive floor for signed-log conversion.
    envelope_coefficient : float, optional
        Harmonic envelope coefficient added to ``logabs``.
    channels, pair_channels : int
        Number of order-2 real feature channels used by the Pfaffian. `channels`
        is a shorthand for `pair_channels`.
    border_channels : int or None, optional
        Number of order-1 border channels for odd-electron systems. Defaults
        to `pair_channels` when bordered odd-electron Pfaffians are enabled.
    trainable : bool, optional
        Whether pair and odd-electron border readout weights are trainable.
        The default keeps them as fixed buffers for scaffold determinism.
    trainable_envelope : bool, optional
        Whether to optimize the envelope coefficient through a softplus
        parameterization.
    """

    def __init__(
        self,
        *,
        allow_odd_electron_bordered: bool = True,
        eps: float = 1.0e-12,
        envelope_coefficient: float = 0.0,
        channels: int | None = None,
        pair_channels: int | None = None,
        border_channels: int | None = None,
        trainable: bool = False,
        trainable_envelope: bool = False,
    ) -> None:
        super().__init__()
        self.allow_odd_electron_bordered = bool(allow_odd_electron_bordered)
        self.eps = float(eps)
        self.trainable = bool(trainable)
        pair_channels = channels if pair_channels is None else pair_channels
        if pair_channels is None:
            raise ValueError("PfaffianReadout requires pair_channels or channels for eager initialization")
        self.pair_channels = _positive_int(pair_channels, "pair_channels")
        if border_channels is None and self.allow_odd_electron_bordered:
            border_channels = self.pair_channels
        self.border_channels = None if border_channels is None else _positive_int(border_channels, "border_channels")
        self._register_readout_weight("channel_weights", "channel_weight_buffer", self.pair_channels)
        if self.border_channels is None:
            self.register_parameter("border_weights", None)
            self.register_buffer("border_weight_buffer", None, persistent=False)
        else:
            self._register_readout_weight("border_weights", "border_weight_buffer", self.border_channels)
        _configure_envelope(self, envelope_coefficient, trainable=trainable_envelope)

    def _register_readout_weight(self, parameter_name: str, buffer_name: str, channels: int) -> None:
        initial = torch.full((channels,), 1.0 / channels)
        if self.trainable:
            self.register_parameter(parameter_name, nn.Parameter(initial))
            self.register_buffer(buffer_name, None, persistent=False)
        else:
            self.register_parameter(parameter_name, None)
            self.register_buffer(buffer_name, initial, persistent=False)

    def build_skew_kernel(self, features: RealFeature, batch: ElectronBatch | None = None) -> torch.Tensor:
        """Construct the skew matrix consumed by the Pfaffian.

        Parameters
        ----------
        features : RealFeature
            Real feature state containing an order-2 block with shape
            ``[batch, channels, n, n]``.
        batch : ElectronBatch or None, optional
            Optional batch used only for shape checks and harmonic envelope
            metadata.

        Returns
        -------
        torch.Tensor
            Skew matrix with shape ``[batch, n, n]`` or bordered shape
            ``[batch, n + 1, n + 1]`` for odd electron counts.
        """

        if 2 not in features:
            raise KeyError("PfaffianReadout requires an order-2 RealFeature block")
        pair = features.blocks[2]
        if pair.ndim != 4:
            raise ValueError(f"Order-2 block must have shape [batch, channels, n, n], got {tuple(pair.shape)}")
        if batch is not None and pair.shape[0] != batch.batch_size:
            raise ValueError("Feature batch size disagrees with ElectronBatch")
        weights = self._ensure_pair_weights(pair.shape[1])
        antisymmetric = 0.5 * (pair - pair.transpose(-1, -2))
        kernel = (antisymmetric * weights.view(1, -1, 1, 1)).sum(dim=1)
        if kernel.shape[-1] % 2 == 1:
            if not self.allow_odd_electron_bordered:
                raise ValueError("Odd-electron Pfaffian requires allow_odd_electron_bordered=True")
            if 1 not in features:
                raise KeyError("Odd-electron Pfaffian requires an order-1 RealFeature border block")
            one_body = features.blocks[1]
            if one_body.shape[0] != kernel.shape[0] or one_body.shape[-1] != kernel.shape[-1]:
                raise ValueError("Order-1 border block must match order-2 batch and particle axes")
            if one_body.shape[1] == 0:
                raise KeyError("Odd-electron Pfaffian requires a nonempty order-1 RealFeature border block")
            border_weights = self._ensure_border_weights(one_body.shape[1])
            border = (one_body * border_weights.view(1, -1, 1)).sum(dim=1)
            bordered = kernel.new_zeros(kernel.shape[0], kernel.shape[1] + 1, kernel.shape[2] + 1)
            bordered[:, :-1, :-1] = kernel
            bordered[:, :-1, -1] = border
            bordered[:, -1, :-1] = -border
            kernel = bordered
        return kernel

    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        """Return a signed-log Pfaffian readout."""

        kernel = self.build_skew_kernel(features, batch)
        logabs, sign = pfaffian_logabs_sign(kernel, eps=self.eps)
        envelope = _harmonic_envelope(self, batch)
        return WavefunctionOutput(
            logabs=logabs + envelope,
            sign=sign,
            aux={"K": kernel, "pfaffian": pfaffian(kernel), "envelope": envelope},
        )

    def _ensure_pair_weights(self, channels: int) -> torch.Tensor:
        return _ensure_readout_weights(
            self,
            channels,
            parameter_name="channel_weights",
            buffer_name="channel_weight_buffer",
        )

    def _ensure_border_weights(self, channels: int) -> torch.Tensor:
        return _ensure_readout_weights(
            self,
            channels,
            parameter_name="border_weights",
            buffer_name="border_weight_buffer",
        )


def _ensure_readout_weights(
    module: "PfaffianReadout",
    channels: int,
    *,
    parameter_name: str,
    buffer_name: str,
) -> torch.Tensor:
    if module.trainable:
        weight = getattr(module, parameter_name)
        if weight is None:
            raise RuntimeError(f"{parameter_name} was not eagerly initialized")
        if tuple(weight.shape) != (channels,):
            raise ValueError(f"{parameter_name} has shape {tuple(weight.shape)}, expected {(channels,)}")
        return weight

    weight = getattr(module, buffer_name)
    if weight is None:
        raise RuntimeError(f"{buffer_name} was not eagerly initialized")
    if tuple(weight.shape) != (channels,):
        raise ValueError(f"{buffer_name} has shape {tuple(weight.shape)}, expected {(channels,)}")
    return weight


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


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


def _current_envelope_coefficient(module: nn.Module) -> torch.Tensor:
    raw = getattr(module, "envelope_raw", None)
    if raw is not None:
        return F.softplus(raw)
    return module.envelope_coefficient


def _harmonic_envelope(module: nn.Module, batch: ElectronBatch) -> torch.Tensor:
    coefficient = _current_envelope_coefficient(module)
    radius_squared = batch.positions.square().sum(dim=(1, 2))
    return -coefficient * radius_squared


__all__ = ["PfaffianReadout", "pfaffian", "pfaffian_logabs_sign"]
