"""Focused materialized parameter-score tests for :class:`TPENWaveFunction`."""

from __future__ import annotations

import io

import pytest
import torch

from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket
from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.nn import (
    CompositeMixing,
    Embedding,
    EquivariantMixing,
    InteractionMode,
    LinearEquivariantMixing,
    MaterializedParameterScoreRequest,
    PathAggregation,
    ResidualUpdater,
    TPENLayer,
    TPENWaveFunction,
)
from tpen.nn.readout import PfaffianReadout


def _build_model(
    mode: InteractionMode,
    *,
    readout: torch.nn.Module | None = None,
) -> TPENWaveFunction:
    """Build one of the landed TP-only, linear-only, and hybrid presets."""

    input_orders = NormalizedOrders((1, 2))
    channels = NormalizedChannels(((1, 1), (2, 1)))
    linear_metadata = LinearPathMetadata.generate(max_order=2)
    tensor_metadata = PathMetadata.generate(max_order=2, max_virtual_order=2, output_embedding="canonical")
    layout = compose_path_layout(
        linear=linear_metadata if mode is not InteractionMode.TENSOR_PRODUCT else None,
        tensor_product=tensor_metadata if mode is not InteractionMode.LINEAR else None,
        input_orders=input_orders,
        output_orders=input_orders,
        input_channels=channels,
        output_channels=channels,
    )
    producers = []
    if mode is not InteractionMode.TENSOR_PRODUCT:
        producers.append(LinearEquivariantMixing(max_order=2, channels=1, metadata=linear_metadata))
    if mode is not InteractionMode.LINEAR:
        producers.append(EquivariantMixing(max_order=2, channels=1, paths=tensor_metadata, activation=None))
    mixing = CompositeMixing(layout=layout, producers=tuple(producers), activation=torch.nn.SiLU())
    aggregation = PathAggregation(max_order=2, channels=1, layout=layout, activation=torch.nn.SiLU())
    layer = TPENLayer(mixing=mixing, path_aggregation=aggregation, update=ResidualUpdater(), layout=layout)
    return TPENWaveFunction(
        embedding=Embedding(max_order=2, spatial_dim=3, out_channels=1, hidden_channels=4, num_hidden_layers=1),
        layers=(layer,),
        readout=PfaffianReadout(channels=1) if readout is None else readout,
        layout=layout,
    ).to(dtype=torch.float64)


def _batch() -> ElectronBatch:
    """Return multidimensional sample axes that the readout flattens."""

    generator = torch.Generator().manual_seed(73)
    return ElectronBatch(
        positions=torch.randn(2, 2, 3, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[[1.0, -1.0, 1.0]] * 2] * 2, dtype=torch.float64),
    )


@pytest.mark.parametrize("mode", tuple(InteractionMode))
def test_slow_and_chunked_scores_agree_for_all_landed_modes(mode: InteractionMode) -> None:
    model = _build_model(mode)
    batch = _batch()

    slow = model(batch, request=MaterializedParameterScoreRequest())
    chunked = model(batch, request=MaterializedParameterScoreRequest(chunk_size=2))

    assert isinstance(slow, ParameterScoreForwardPacket)
    assert isinstance(chunked, ParameterScoreForwardPacket)
    assert tuple(slow.output.logabs.shape) == (4,)
    assert slow.parameter_scores.sample_shape == (4,)
    assert slow.parameter_scores.layout.compare(chunked.parameter_scores.layout)[0]
    for parameter, slow_block, chunked_block in zip(
        model.parameter_binding.parameters,
        slow.parameter_scores.blocks,
        chunked.parameter_scores.blocks,
    ):
        assert slow_block.shape == (4, *tuple(parameter.shape))
        assert not slow_block.requires_grad
        assert not chunked_block.requires_grad
        torch.testing.assert_close(slow_block, chunked_block, rtol=1.0e-10, atol=1.0e-10)


def test_flattened_j_and_jt_products_match_ordinary_autograd() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    batch = _batch()
    parameters = model.parameter_binding.parameters
    sample_count = batch.batch_size
    weights = torch.linspace(0.25, 1.25, sample_count, dtype=torch.float64)

    weighted_output = model(batch)
    ordinary_jt = torch.autograd.grad(
        (weights * weighted_output.logabs.reshape(-1)).sum(),
        parameters,
    )

    ordinary_output = model(batch)
    ordinary_rows = []
    for sample_index, value in enumerate(ordinary_output.logabs.reshape(-1)):
        gradients = torch.autograd.grad(
            value,
            parameters,
            retain_graph=sample_index + 1 < sample_count,
        )
        ordinary_rows.append(torch.cat(tuple(gradient.reshape(-1) for gradient in gradients)))
    ordinary_j = torch.stack(ordinary_rows)

    packet = model(batch, request=MaterializedParameterScoreRequest(chunk_size=2))
    assert isinstance(packet, ParameterScoreForwardPacket)
    materialized_j = torch.cat(
        tuple(block.reshape(sample_count, -1) for block in packet.parameter_scores.blocks),
        dim=1,
    )
    ordinary_jt_flat = torch.cat(tuple(gradient.reshape(-1) for gradient in ordinary_jt))
    direction = torch.linspace(
        -0.4,
        0.6,
        materialized_j.shape[1],
        dtype=torch.float64,
    )

    torch.testing.assert_close(materialized_j, ordinary_j, rtol=1.0e-10, atol=1.0e-10)
    torch.testing.assert_close(materialized_j.transpose(0, 1) @ weights, ordinary_jt_flat)
    torch.testing.assert_close(materialized_j @ direction, ordinary_j @ direction)


class _UnusedPfaffianReadout(PfaffianReadout):
    """Add a registered parameter that the inherited readout never consumes."""

    def __init__(self) -> None:
        super().__init__(channels=1)
        self.unused = torch.nn.Parameter(torch.tensor(1.0))


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_unused_parameter_fails_explicitly(chunk_size: int | None) -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_UnusedPfaffianReadout())

    with pytest.raises(RuntimeError, match="unused or disconnected"):
        model(_batch(), request=MaterializedParameterScoreRequest(chunk_size=chunk_size))


def test_parameter_reordering_is_rejected_before_building_or_updating() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    batch = _batch()
    before = tuple((parameter, parameter.detach().clone()) for parameter in model.parameter_binding.parameters)
    module_items = tuple(model._modules.items())
    model._modules.clear()
    for module_name, module in reversed(module_items):
        model._modules[module_name] = module

    with pytest.raises(ValueError, match="binding/layout mismatch or reordering"):
        model(batch, request=MaterializedParameterScoreRequest())

    assert all(parameter.grad is None for parameter in model.parameters())
    for parameter, prior in before:
        torch.testing.assert_close(parameter, prior, rtol=0.0, atol=0.0)


def test_live_packets_reject_serialization_and_detached_packets_can_be_saved() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    packet = model(_batch(), request=MaterializedParameterScoreRequest(chunk_size=2))
    assert isinstance(packet, ParameterScoreForwardPacket)
    assert packet.output.logabs.requires_grad

    with pytest.raises(RuntimeError, match="graph-bearing"):
        torch.save(packet, io.BytesIO())

    detached = packet.detach()
    assert not detached.output.logabs.requires_grad
    assert all(not block.requires_grad for block in detached.parameter_scores.blocks)
    torch.save(detached, io.BytesIO())


def test_parameter_binding_is_direct_and_refreshes_after_model_owned_cast() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    binding = model.parameter_binding

    assert all(left is right for left, right in zip(model.parameters(), binding.parameters))
    assert all(slot.dtype == torch.float64 for slot in binding.layout.slots)
    assert binding.layout.total_numel == sum(parameter.numel() for parameter in binding.parameters)

    model.to(dtype=torch.float32)
    rebound = model.parameter_binding
    assert all(left is right for left, right in zip(model.parameters(), rebound.parameters))
    assert all(slot.dtype == torch.float32 for slot in rebound.layout.slots)


def test_parameter_score_request_rejects_inference_mode() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)

    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference mode"):
        model(_batch(), request=MaterializedParameterScoreRequest())
