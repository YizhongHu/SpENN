"""Runtime equivariance tests for tuple particle-vector embedding."""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch
from spenn.nn import Embedding


def test_embedding_passes_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(13579)
    batch = ElectronBatch(
        positions=torch.randn(2, 4, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]], dtype=torch.float64),
        aux={"types": torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)},
    )
    embedding = Embedding(
        max_order=3,
        out_channels={1: 3, 2: 4, 3: 5},
        hidden_channels=8,
        num_hidden_layers=1,
        aux_feature_keys=("types",),
        equivariance_check=True,
        check_probability=1.0,
        tensor_validation_check=True,
    )

    feature = embedding(batch)

    assert feature.validate() is feature


def test_embedding_passes_runtime_equivariance_with_sample_axes() -> None:
    generator = torch.Generator().manual_seed(24680)
    batch = ElectronBatch(
        positions=torch.randn(2, 3, 4, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[[1.0, -1.0, 1.0, -1.0]] * 3] * 2, dtype=torch.float64),
        aux={"types": torch.randn(2, 3, 4, 2, generator=generator, dtype=torch.float64)},
    )
    embedding = Embedding(
        max_order=2,
        out_channels=4,
        hidden_channels=7,
        num_hidden_layers=1,
        aux_feature_keys=("types",),
        equivariance_check=True,
        check_probability=1.0,
        tensor_validation_check=True,
    )

    feature = embedding(batch)

    assert feature.validate() is feature
