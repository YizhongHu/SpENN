"""Tests for the order-1/order-2 electron pair encoder."""

from __future__ import annotations

import torch

from spenn.data import Par
from spenn.data.batch import ElectronBatch
from spenn.nn.encoding import ElectronPairEncoder


ORDER1 = Par("H")
ORDER2_SYM = Par("S")
ORDER2_SIGN = Par("A")


def _batch(dtype: torch.dtype = torch.float64) -> ElectronBatch:
    positions = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[1.0, -1.0], [0.5, 2.0], [3.0, -0.5]],
        ],
        dtype=dtype,
    )
    return ElectronBatch(positions=positions)


def test_raw_features_return_q_list_with_tuple_scalar_shapes() -> None:
    q = ElectronPairEncoder(channels=[0, 5, 7]).raw_features(_batch())

    assert len(q) == 3
    assert q[0].shape == (2, 0)
    assert q[1].shape == (2, 5, 3)
    assert q[2].shape == (2, 7, 3, 3)
    assert torch.allclose(q[2].diagonal(dim1=2, dim2=3), torch.zeros_like(q[2].diagonal(dim1=2, dim2=3)))


def test_projected_features_match_configured_channels_and_dtype() -> None:
    encoder = ElectronPairEncoder(channels=[0, 5, 7]).to(dtype=torch.float64)

    features = encoder(_batch(dtype=torch.float64))

    assert features.get(ORDER1).shape == (2, 5, 3, 1, 1)
    assert features.get(ORDER2_SYM).shape == (2, 7, 3, 3, 1, 1)
    assert features.get(ORDER2_SIGN).shape == (2, 7, 3, 3, 1, 1)
    assert features.get(ORDER2_SIGN).dtype == torch.float64
    features.validate(batch_size=2, n_electrons=3, supported=encoder.output_keys())


def test_pair_features_have_exact_symmetry_type() -> None:
    features = ElectronPairEncoder(channels=[0, 2, 4])(_batch())

    symmetric = features.get(ORDER2_SYM)
    antisymmetric = features.get(ORDER2_SIGN)

    assert torch.allclose(symmetric, symmetric.transpose(2, 3))
    assert torch.allclose(antisymmetric, -antisymmetric.transpose(2, 3))
    assert torch.allclose(antisymmetric.diagonal(dim1=2, dim2=3), torch.zeros_like(antisymmetric.diagonal(dim1=2, dim2=3)))


def test_encoder_is_equivariant_under_electron_permutation() -> None:
    encoder = ElectronPairEncoder(channels=[0, 2, 4])
    batch = _batch()
    permutation = torch.tensor([2, 0, 1])
    permuted = ElectronBatch(positions=batch.positions[:, permutation])

    original = encoder(batch)
    transformed = encoder(permuted)
    inverse = torch.argsort(permutation)

    assert torch.allclose(original.get(ORDER1), transformed.get(ORDER1)[:, :, inverse])
    assert torch.allclose(original.get(ORDER2_SYM), transformed.get(ORDER2_SYM)[:, :, inverse][:, :, :, inverse])
    assert torch.allclose(original.get(ORDER2_SIGN), transformed.get(ORDER2_SIGN)[:, :, inverse][:, :, :, inverse])


def test_particle_descriptors_append_spins_only_when_available_and_enabled() -> None:
    batch = _batch()
    spins = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=batch.dtype)
    spin_batch = ElectronBatch(positions=batch.positions, spins=spins)

    with_spins = ElectronPairEncoder(channels=[0, 2, 0]).particle_descriptors(spin_batch)
    without_spins = ElectronPairEncoder(channels=[0, 2, 0]).particle_descriptors(batch)
    disabled = ElectronPairEncoder(channels=[0, 2, 0], include_spins=False).particle_descriptors(spin_batch)

    assert with_spins.shape == (2, 3, 3)
    assert torch.equal(with_spins[..., -1], spins)
    assert without_spins.shape == (2, 3, 2)
    assert disabled.shape == (2, 3, 2)


def test_encoder_permutation_equivariance_moves_spins_with_positions() -> None:
    encoder = ElectronPairEncoder(channels=[0, 2, 4])
    batch = _batch()
    spins = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=batch.dtype)
    spin_batch = ElectronBatch(positions=batch.positions, spins=spins)
    permutation = torch.tensor([2, 0, 1])
    permuted = ElectronBatch(positions=batch.positions[:, permutation], spins=spins[:, permutation])

    original = encoder(spin_batch)
    transformed = encoder(permuted)
    inverse = torch.argsort(permutation)

    assert torch.allclose(original.get(ORDER1), transformed.get(ORDER1)[:, :, inverse])
    assert torch.allclose(original.get(ORDER2_SYM), transformed.get(ORDER2_SYM)[:, :, inverse][:, :, :, inverse])
    assert torch.allclose(original.get(ORDER2_SIGN), transformed.get(ORDER2_SIGN)[:, :, inverse][:, :, :, inverse])


def test_encoder_flattens_higher_rank_sample_shape() -> None:
    positions = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)
    batch = ElectronBatch(positions=positions)
    encoder = ElectronPairEncoder(channels=[0, 5, 7])

    features = encoder(batch)

    assert features.get(ORDER1).shape == (6, 5, 4, 1, 1)
    assert features.get(ORDER2_SYM).shape == (6, 7, 4, 4, 1, 1)
    assert features.get(ORDER2_SIGN).shape == (6, 7, 4, 4, 1, 1)


def test_max_order_one_and_zero_channels_omit_inactive_keys() -> None:
    max_order_one = ElectronPairEncoder(max_order=1, channels=[0, 3, 5])
    zero_channels = ElectronPairEncoder(channels=[0, 0, 0])

    assert max_order_one.output_keys() == (ORDER1,)
    assert list(max_order_one(_batch()).flat_items())[0][0] == ORDER1
    assert zero_channels.output_keys() == ()
    features = zero_channels(_batch())
    assert not features.has(ORDER1)
    assert not features.has(ORDER2_SYM)
    assert not features.has(ORDER2_SIGN)


def test_order2_channel_count_applies_to_both_pair_irreps() -> None:
    encoder = ElectronPairEncoder(channels=[0, 2, 3])

    assert encoder.output_keys() == (ORDER1, ORDER2_SYM, ORDER2_SIGN)
    features = encoder(_batch())
    assert features.get(ORDER2_SYM).shape == (2, 3, 3, 3, 1, 1)
    assert features.get(ORDER2_SIGN).shape == (2, 3, 3, 3, 1, 1)
def test_mapping_channels_are_rejected() -> None:
    try:
        ElectronPairEncoder(channels={"order1": {"(1)": 2}, "order2": {"(2)": 3, "(1,1)": 3}})
    except TypeError as exc:
        assert "sequence" in str(exc)
    else:
        raise AssertionError("Expected mapping-style channels to raise")
