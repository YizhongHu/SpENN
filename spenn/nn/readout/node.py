"""Readouts with explicit low-electron nodal factors."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.feature_dict import FeatureDict
from spenn.data.irrep_tensor import scalar_channels_last
from spenn.data.partitions import Par


class TwoElectronTripletNodeReadout(nn.Module):
    """Read out a two-electron triplet with an explicit Cartesian node.

    Parameters
    ----------
    node_axis : int, optional
        Cartesian axis used for the antisymmetric factor
        ``r_1[node_axis] - r_2[node_axis]``.
    eps : float, optional
        Positive floor used in the node log-amplitude.
    envelope_coefficient : float, optional
        Nonnegative coefficient for the harmonic one-body envelope
        ``-envelope_coefficient * sum_i |r_i|^2`` added to the log-amplitude.
    trainable_envelope : bool, optional
        Whether to optimize `envelope_coefficient` through a positive softplus
        parameterization.
    zero_init_residual : bool, optional
        Whether to initialize the learned smooth residual projection to zero on
        its first materialized forward pass.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    This readout is intentionally specialized to the two-electron Hooke
    triplet benchmark. It keeps the exchange sign fixed by construction while
    learning the smooth symmetric residual from pair-symmetric Specht features.
    """

    def __init__(
        self,
        node_axis: int = 2,
        eps: float = 1.0e-12,
        envelope_coefficient: float = 0.0,
        trainable_envelope: bool = False,
        zero_init_residual: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.node_axis = int(node_axis)
        self.eps = float(eps)
        self.residual_projection = nn.LazyLinear(1, dtype=torch.float64)
        self.zero_init_residual = bool(zero_init_residual)
        self._residual_initialized = False
        _configure_envelope(self, envelope_coefficient, trainable=trainable_envelope)

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        """Return signed-log triplet wavefunction values.

        Parameters
        ----------
        features : FeatureDict
            SpENN features containing pair-symmetric scalar features at
            partition ``(2)``.
        batch : ElectronBatch
            Electron batch with positions shaped ``[batch, 2, spatial_dim]``
            after sample flattening.

        Returns
        -------
        WavefunctionOutput
            Signed-log output with shape ``[batch]``.
        """

        batch = batch.flatten_samples()
        if batch.n_electrons != 2:
            raise ValueError(f"TwoElectronTripletNodeReadout requires exactly two electrons, got {batch.n_electrons}")
        if not 0 <= self.node_axis < batch.spatial_dim:
            raise ValueError(f"node_axis {self.node_axis} is outside spatial dimension {batch.spatial_dim}")
        symmetric = features.get(Par("S"))
        if symmetric is None:
            raise KeyError("TwoElectronTripletNodeReadout requires pair-symmetric features at partition (2)")
        pair_features = scalar_channels_last(symmetric)
        if pair_features.shape[:3] != (batch.batch_size, 2, 2):
            raise ValueError(
                "Pair-symmetric features must have shape [batch, 2, 2, channels], "
                f"got {tuple(pair_features.shape)}"
            )
        off_diagonal = 0.5 * (pair_features[:, 0, 1, :] + pair_features[:, 1, 0, :])
        residual = _project_residual(self, off_diagonal.to(dtype=batch.dtype, device=batch.device))
        node = batch.positions[:, 0, self.node_axis] - batch.positions[:, 1, self.node_axis]
        sign = torch.sign(node)
        log_node = torch.where(
            sign == 0,
            torch.full_like(node, -torch.inf),
            torch.log(node.abs().clamp_min(self.eps)),
        )
        envelope = _harmonic_envelope(self, batch)
        logabs = residual + log_node + envelope
        assert residual.shape == logabs.shape == sign.shape == envelope.shape == (batch.batch_size,)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"node": node, "smooth_residual": residual})


class TwoElectronSingletSymmetricReadout(nn.Module):
    """Read out a positive two-electron singlet with exchange symmetry.

    Parameters
    ----------
    envelope_coefficient : float, optional
        Nonnegative coefficient for the harmonic one-body envelope
        ``-envelope_coefficient * sum_i |r_i|^2`` added to the log-amplitude.
    trainable_envelope : bool, optional
        Whether to optimize `envelope_coefficient` through a positive softplus
        parameterization.
    zero_init_residual : bool, optional
        Whether to initialize the learned smooth residual projection to zero on
        its first materialized forward pass.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    This readout is specialized to two-electron singlet benchmarks. It pools
    one-body features over electrons and pair-symmetric off-diagonal features
    over ordered pair directions, then predicts a smooth positive
    log-amplitude residual with sign fixed to ``+1``.
    """

    def __init__(
        self,
        envelope_coefficient: float = 0.0,
        trainable_envelope: bool = False,
        zero_init_residual: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.residual_projection = nn.LazyLinear(1, dtype=torch.float64)
        self.zero_init_residual = bool(zero_init_residual)
        self._residual_initialized = False
        _configure_envelope(self, envelope_coefficient, trainable=trainable_envelope)

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        """Return signed-log singlet wavefunction values.

        Parameters
        ----------
        features : FeatureDict
            SpENN features containing pair-symmetric features and optionally
            one-body features.
        batch : ElectronBatch
            Electron batch with positions shaped ``[batch, 2, spatial_dim]``
            after sample flattening.

        Returns
        -------
        WavefunctionOutput
            Positive signed-log output with shape ``[batch]``.
        """

        batch = batch.flatten_samples()
        if batch.n_electrons != 2:
            raise ValueError(f"TwoElectronSingletSymmetricReadout requires exactly two electrons, got {batch.n_electrons}")
        symmetric = features.get(Par("S"))
        if symmetric is None:
            raise KeyError("TwoElectronSingletSymmetricReadout requires pair-symmetric features at partition (2)")
        pair_features = scalar_channels_last(symmetric)
        if pair_features.shape[:3] != (batch.batch_size, 2, 2):
            raise ValueError(
                "Pair-symmetric features must have shape [batch, 2, 2, channels], "
                f"got {tuple(pair_features.shape)}"
            )
        pooled_features = [0.5 * (pair_features[:, 0, 1, :] + pair_features[:, 1, 0, :])]
        one_body = features.get(Par("H"))
        if one_body is not None:
            one_body_features = scalar_channels_last(one_body)
            if one_body_features.shape[:2] != (batch.batch_size, 2):
                raise ValueError(
                    "One-body features must have shape [batch, 2, channels], "
                    f"got {tuple(one_body_features.shape)}"
                )
            pooled_features.append(one_body_features.mean(dim=1))
        residual_input = torch.cat(pooled_features, dim=-1)
        residual = _project_residual(self, residual_input.to(dtype=batch.dtype, device=batch.device))
        envelope = _harmonic_envelope(self, batch)
        logabs = residual + envelope
        sign = torch.ones_like(logabs)
        assert residual.shape == logabs.shape == sign.shape == envelope.shape == (batch.batch_size,)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"smooth_residual": residual})


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


def _project_residual(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    residual = module.residual_projection(inputs).squeeze(-1)
    if module.zero_init_residual and not module._residual_initialized:
        nn.init.zeros_(module.residual_projection.weight)
        nn.init.zeros_(module.residual_projection.bias)
        module._residual_initialized = True
        residual = module.residual_projection(inputs).squeeze(-1)
    assert residual.shape == inputs.shape[:-1]
    return residual
