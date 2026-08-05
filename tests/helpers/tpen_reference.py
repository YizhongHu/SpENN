"""Slow, readable TPEN reference implementations for oracle tests.

These are pytest-only correctness references for the TPEN migration
(MIG-TPEN-000 §5, gates T1/T2/T3/T5/T6/T12). They implement the TPEN layer
contract in literal loops:

``x -> mixing(W, Gamma) -> h -> aggregation(U, Gamma_c) -> u -> update``

Mixing reuse: :class:`tpen.nn.EquivariantMixing` already owns a slow literal
loop implementation; the reference layer composes it with the pinned owned
activations. Aggregation and the per-channel Pfaffian readout are implemented
here from scratch because the future fast modules do not exist yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from tpen.data.real import Feature, Interaction, zero_block
from tpen.nn.readout.pfaffian import pfaffian

Activation = Callable[[torch.Tensor], torch.Tensor]


def apply_pointwise_activation(interaction: Interaction, activation: Activation) -> Interaction:
    """Apply the mixing-owned pointwise activation to every interaction block.

    Pinned contract (MIG-TPEN-000 §2.2): Gamma is applied to the full block,
    including non-distinct tuple entries that mixing never writes. The path
    axis stays independent — Gamma never mixes paths, channels, or particles.

    Parameters
    ----------
    interaction : Interaction
        Path-resolved mixing output ``[batch, channels, paths, indices...]``.
    activation : callable
        Elementwise activation Gamma.

    Returns
    -------
    Interaction
        Activated interaction with unchanged shapes.
    """

    blocks = [interaction.blocks[0]]
    for order in range(1, len(interaction.blocks)):
        blocks.append(activation(interaction.blocks[order]))
    return Interaction(blocks)


def slow_tpen_aggregation(
    interaction: Interaction,
    path_weights: Sequence[torch.Tensor | None],
    activation: Activation,
) -> Feature:
    """Aggregate path-resolved interactions into features, literal-loop form.

    Implements the TPEN aggregation contract (MIG-TPEN-000 §2.2, decision D3):

    ``u[c, I] = Gamma_c( sum_p U[m][c, p] * h[c, p, I] )``

    with elementwise ``Gamma_c`` and ``C_out = C_in``. The path axis is
    contracted per input channel; channel mixing lives in mixing's ``W``
    until the MLP upgrade.

    Parameters
    ----------
    interaction : Interaction
        Blocks ``[batch, channels, paths, indices...]`` indexed by order.
    path_weights : sequence of torch.Tensor or None
        Entry ``m`` holds the order-``m`` weight ``U[c, p]`` with shape
        ``[channels, paths_m]``. Index 0 is ignored (reserved block).
    activation : callable
        Elementwise Gamma_c applied after the path contraction.

    Returns
    -------
    Feature
        Aggregated feature blocks ``[batch, channels, indices...]``.

    Raises
    ------
    ValueError
        If a weight's path count disagrees with the interaction block
        (T5 negative case: silent broadcast/truncation is forbidden).
    """

    batch_size = interaction.batch_size
    blocks: list[torch.Tensor] = [
        zero_block(batch_size=batch_size, device=interaction.blocks[0].device, dtype=interaction.blocks[0].dtype)
    ]
    for order in range(1, len(interaction.blocks)):
        block = interaction.blocks[order]
        weight = path_weights[order] if order < len(path_weights) else None
        if weight is None:
            raise ValueError(f"slow_tpen_aggregation is missing the order-{order} path weight")
        channels, paths = block.shape[1], block.shape[2]
        if tuple(weight.shape) != (channels, paths):
            raise ValueError(
                f"Order-{order} path weight shape {tuple(weight.shape)} does not match "
                f"interaction [channels, paths] = {(channels, paths)}"
            )
        # Literal per-channel, per-path contraction: no einsum, no broadcast.
        contracted = torch.zeros(
            (batch_size, channels, *block.shape[3:]), device=block.device, dtype=block.dtype
        )
        for channel in range(channels):
            for path in range(paths):
                contracted[:, channel] = contracted[:, channel] + weight[channel, path] * block[:, channel, path]
        blocks.append(activation(contracted))
    return Feature(blocks)


def slow_tpen_layer(
    x: Feature,
    *,
    mixing: Callable[[Feature], Interaction],
    mixing_activation: Activation,
    path_weights: Sequence[torch.Tensor | None],
    aggregation_activation: Activation,
) -> Feature:
    """One TPEN layer in reference form with a residual update.

    ``x -> mixing -> Gamma -> aggregation(U, Gamma_c) -> x + u``

    Parameters
    ----------
    x : Feature
        Input feature state.
    mixing : callable
        Module or function producing a path-resolved :class:`Interaction`
        (e.g. ``EquivariantMixing`` with the slow implementation).
    mixing_activation, aggregation_activation : callable
        The op-owned activations Gamma and Gamma_c.
    path_weights : sequence of torch.Tensor or None
        Aggregation weights, see :func:`slow_tpen_aggregation`.

    Returns
    -------
    Feature
        Residually updated feature state.
    """

    interaction = apply_pointwise_activation(mixing(x), mixing_activation)
    update = slow_tpen_aggregation(interaction, path_weights, aggregation_activation)
    # Residual update x + u; channel counts must match for the reference.
    return x.add(update)


def per_channel_pfaffian_readout(features: Feature, weights: torch.Tensor) -> torch.Tensor:
    """B1 reference readout: weighted sum of per-channel Pfaffians.

    ``Psi = sum_c w_c * Pf[ 0.5 * (x[c] - x[c]^T) ]`` (MIG-TPEN-000 §2.2,
    decision B1). Even electron count only; odd-``n`` padding is a later
    slice's concern.

    Parameters
    ----------
    features : Feature
        Feature state with an order-2 block ``[batch, channels, n, n]``.
    weights : torch.Tensor
        Per-channel readout weights ``[channels]``.

    Returns
    -------
    torch.Tensor
        ``Psi`` with shape ``[batch]``.
    """

    pair = features.blocks[2]
    batch_size, channels, n, _ = pair.shape
    if n % 2 == 1:
        raise ValueError("per_channel_pfaffian_readout reference supports even n only")
    if tuple(weights.shape) != (channels,):
        raise ValueError(f"weights must have shape [{channels}], got {tuple(weights.shape)}")
    psi = torch.zeros(batch_size, device=pair.device, dtype=pair.dtype)
    for channel in range(channels):
        kernel = 0.5 * (pair[:, channel] - pair[:, channel].transpose(-1, -2))
        psi = psi + weights[channel] * pfaffian(kernel)
    return psi


def mixed_kernel_pfaffian_readout(features: Feature, weights: torch.Tensor) -> torch.Tensor:
    """Rejected pre-B1 readout: one Pfaffian of the channel-mixed kernel.

    ``Psi = Pf[ sum_c w_c * 0.5 * (x[c] - x[c]^T) ]`` — kept only so T6 can
    assert the two function classes actually differ on the pinned case.
    """

    pair = features.blocks[2]
    skew = 0.5 * (pair - pair.transpose(-1, -2))
    kernel = torch.einsum("c,bcij->bij", weights, skew)
    return pfaffian(kernel)


__all__ = [
    "apply_pointwise_activation",
    "mixed_kernel_pfaffian_readout",
    "per_channel_pfaffian_readout",
    "slow_tpen_aggregation",
    "slow_tpen_layer",
]
