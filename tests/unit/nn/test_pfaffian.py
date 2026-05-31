"""Pfaffian correctness and signed-log readout tests."""

from __future__ import annotations

import torch

from spenn.data import FeatureDict, Par
from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.nn.readout.pfaffian import PfaffianReadout, pfaffian, pfaffian_logabs_sign, signed_logsumexp_outputs


def _skew4(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    a, b, c, d, e, f = [torch.tensor(value, dtype=dtype) for value in (2.0, -3.0, 5.0, 7.0, 11.0, -13.0)]
    return torch.stack(
        [
            torch.stack([a.new_zeros(()), a, b, c]),
            torch.stack([-a, a.new_zeros(()), d, e]),
            torch.stack([-b, -d, a.new_zeros(()), f]),
            torch.stack([-c, -e, -f, a.new_zeros(())]),
        ]
    )


def _features_from_positions(positions: torch.Tensor) -> FeatureDict:
    x = positions[..., 0]
    carrier = (x.unsqueeze(2) - x.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    gate = torch.ones_like(carrier)
    one_body = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return FeatureDict({Par("H"): one_body, Par("S"): gate, Par("A"): carrier})


def _set_unit_readout_weights(readout: PfaffianReadout) -> None:
    with torch.no_grad():
        for carrier, gate, border in zip(
            readout.carrier_projections,
            readout.gate_projections,
            readout.border_projections,
            strict=True,
        ):
            carrier.weight.fill_(1.0)
            gate.weight.zero_()
            gate.bias.fill_(1.0)
            border.weight.fill_(1.0)


def test_pfaffian_matches_hand_computed_small_cases() -> None:
    empty = torch.empty(0, 0, dtype=torch.float64)
    two = torch.tensor([[0.0, -4.0], [4.0, 0.0]], dtype=torch.float64)
    four = _skew4()
    expected_four = four[0, 1] * four[2, 3] - four[0, 2] * four[1, 3] + four[0, 3] * four[1, 2]

    assert torch.equal(pfaffian(empty), torch.tensor(1.0, dtype=torch.float64))
    assert torch.equal(pfaffian(two), torch.tensor(-4.0, dtype=torch.float64))
    assert torch.equal(pfaffian(four), expected_four)

    batched = torch.stack([two, -two])
    assert torch.equal(pfaffian(batched), torch.tensor([-4.0, 4.0], dtype=torch.float64))


def test_pfaffian_signed_log_handles_positive_negative_and_exact_zero() -> None:
    matrices = torch.stack(
        [
            torch.tensor([[0.0, 3.0], [-3.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, -2.0], [2.0, 0.0]], dtype=torch.float64),
            torch.zeros(2, 2, dtype=torch.float64),
        ]
    )

    logabs, sign = pfaffian_logabs_sign(matrices)

    assert torch.allclose(logabs[:2], torch.log(torch.tensor([3.0, 2.0], dtype=torch.float64)))
    assert torch.equal(sign, torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    assert torch.isneginf(logabs[2])


def test_signed_logsumexp_outputs_uses_exact_zero_convention_for_cancellation() -> None:
    positive = WavefunctionOutput(logabs=torch.log(torch.tensor([2.0])), sign=torch.tensor([1.0]))
    negative = WavefunctionOutput(logabs=torch.log(torch.tensor([2.0])), sign=torch.tensor([-1.0]))

    cancelled = signed_logsumexp_outputs([positive, negative])

    assert torch.equal(cancelled.sign, torch.tensor([0.0]))
    assert torch.isneginf(cancelled.logabs[0])

    shifted = signed_logsumexp_outputs(
        [positive, negative],
        weights=torch.tensor([1.0, 0.25]),
    )
    assert torch.equal(shifted.sign, torch.tensor([1.0]))
    assert torch.allclose(shifted.logabs, torch.log(torch.tensor([1.5])))


def test_pfaffian_readout_builds_skew_kernels_for_even_and_odd_electron_counts() -> None:
    for n_electrons in (2, 3):
        positions = torch.arange(n_electrons, dtype=torch.float64).reshape(1, n_electrons, 1)
        batch = ElectronBatch(positions=positions)
        features = _features_from_positions(positions)
        readout = PfaffianReadout()
        readout.build_skew_kernel(features, batch)
        _set_unit_readout_weights(readout)

        kernel = readout.build_skew_kernel(features, batch)

        assert torch.allclose(kernel + kernel.transpose(-1, -2), torch.zeros_like(kernel))
        assert kernel.dtype == torch.float64
        if n_electrons == 3:
            assert kernel.shape[-2:] == (4, 4)


def test_pfaffian_readout_is_antisymmetric_under_electron_swap() -> None:
    positions = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
    batch = ElectronBatch(positions=positions)
    features = _features_from_positions(positions)
    readout = PfaffianReadout()
    readout.build_skew_kernel(features, batch)
    _set_unit_readout_weights(readout)

    original = readout(features, batch)
    swapped_positions = positions[:, [1, 0]]
    swapped = readout(_features_from_positions(swapped_positions), ElectronBatch(positions=swapped_positions))

    assert torch.all(torch.isfinite(original.logabs))
    assert torch.allclose(original.logabs, swapped.logabs)
    assert torch.equal(original.sign, -swapped.sign)
