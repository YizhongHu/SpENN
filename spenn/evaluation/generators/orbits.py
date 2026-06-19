"""Orbit generators for deterministic transform validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from spenn.data.batch import ElectronBatch
from spenn.data.permutation import Permutation, select_nonidentity_permutations
from spenn.evaluation.bundle import GeneratedConfigurations
from spenn.evaluation.protocols import EvaluationContext


class PermutationOrbitGenerator:
    """Generate paired original/permuted electron configurations."""

    name = "permutation_orbit"

    def __init__(
        self,
        *,
        base_generator: object,
        permutations: Sequence[torch.Tensor | Sequence[int] | Permutation] | None = None,
        n_permutations: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_generator = base_generator
        self.permutations = None if permutations is None else tuple(_as_permutation(item) for item in permutations)
        self.n_permutations = None if n_permutations is None else int(n_permutations)
        self.seed = seed

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return a paired batch with sample shape ``[orbit, 2]``."""

        base = self.base_generator.generate(model=model, context=context)
        flat = base.batch.flatten_samples()
        permutations = self.permutations
        if permutations is None:
            max_count = self.n_permutations if self.n_permutations is not None else 1
            permutations = tuple(
                select_nonidentity_permutations(
                    n_particles=flat.n_electrons,
                    fraction=1.0,
                    max_count=max_count,
                    seed=self.seed,
                )
            )
        original, transformed = _permutation_pairs(flat, permutations)
        paired = _stack_pair_batch(original, transformed)
        metadata = _orbit_metadata(flat.batch_size, len(permutations), device=flat.device)
        metadata.update(
            {
                "permutation": torch.tensor(
                    [permutation.image for permutation in permutations],
                    device=flat.device,
                    dtype=torch.long,
                ).repeat_interleave(flat.batch_size, dim=0),
                "permutation_parity": torch.tensor(
                    [permutation.sign for permutation in permutations],
                    device=flat.device,
                    dtype=flat.dtype,
                ).repeat_interleave(flat.batch_size, dim=0),
                "base_metadata": base.metadata,
            }
        )
        return GeneratedConfigurations(batch=paired, metadata=metadata)


class RotationOrbitGenerator:
    """Generate paired original/rotated electron configurations."""

    name = "rotation_orbit"

    def __init__(
        self,
        *,
        base_generator: object,
        n_rotations: int,
        seed: int,
    ) -> None:
        self.base_generator = base_generator
        self.n_rotations = int(n_rotations)
        self.seed = int(seed)

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return a paired batch with random orthogonal rotations."""

        if self.n_rotations <= 0:
            raise ValueError("RotationOrbitGenerator requires n_rotations > 0")
        base = self.base_generator.generate(model=model, context=context)
        flat = base.batch.flatten_samples()
        rotations = _rotation_matrices(
            flat.spatial_dim,
            self.n_rotations,
            seed=self.seed,
            device=flat.device,
            dtype=flat.dtype,
        )
        original, transformed = _rotation_pairs(flat, rotations)
        paired = _stack_pair_batch(original, transformed)
        metadata = _orbit_metadata(flat.batch_size, self.n_rotations, device=flat.device)
        metadata.update(
            {
                "rotation_id": metadata["transform_id"],
                "rotation_matrix": rotations.repeat_interleave(flat.batch_size, dim=0),
                "base_metadata": base.metadata,
            }
        )
        return GeneratedConfigurations(batch=paired, metadata=metadata)


class ExchangeOrbitGenerator:
    """Generate paired original/spatially exchanged Hooke configurations."""

    name = "exchange_orbit"

    def __init__(
        self,
        *,
        base_generator: object,
        exchange: Literal["opposite_spin_pair", "same_spin_pair"] = "opposite_spin_pair",
    ) -> None:
        self.base_generator = base_generator
        self.exchange = exchange

    def generate(self, *, model: torch.nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        """Return paired configurations with one spatial coordinate exchange."""

        base = self.base_generator.generate(model=model, context=context)
        flat = base.batch.flatten_samples()
        pair = _exchange_pair(flat, self.exchange)
        transformed_positions = flat.positions.clone()
        left, right = pair
        transformed_positions[:, left, :] = flat.positions[:, right, :]
        transformed_positions[:, right, :] = flat.positions[:, left, :]
        transformed = ElectronBatch(
            positions=transformed_positions,
            system=flat.system,
            nuclear_positions=flat.nuclear_positions,
            nuclear_charges=flat.nuclear_charges,
            spins=flat.spins,
            aux=dict(flat.aux),
        )
        paired = _stack_pair_batch(flat, transformed)
        metadata = {
            "base_sample_index": torch.arange(flat.batch_size, device=flat.device),
            "exchange_id": torch.zeros(flat.batch_size, device=flat.device, dtype=torch.long),
            "exchange_pair": torch.tensor(pair, device=flat.device, dtype=torch.long).expand(flat.batch_size, 2),
            "orbit_id": torch.arange(flat.batch_size, device=flat.device),
            "expected_sign_parity": torch.ones(flat.batch_size, device=flat.device, dtype=flat.dtype),
            "base_metadata": base.metadata,
        }
        return GeneratedConfigurations(batch=paired, metadata=metadata)


def _as_permutation(value: torch.Tensor | Sequence[int] | Permutation) -> Permutation:
    if isinstance(value, Permutation):
        return value
    if isinstance(value, torch.Tensor):
        return Permutation(tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist()))
    return Permutation(tuple(int(item) for item in value))


def _permutation_pairs(batch: ElectronBatch, permutations: Sequence[Permutation]) -> tuple[ElectronBatch, ElectronBatch]:
    originals = []
    transformed = []
    for permutation in permutations:
        originals.append(batch)
        transformed.append(batch.permute(permutation))
    if not originals:
        empty_positions = batch.positions[:0]
        empty = ElectronBatch(
            positions=empty_positions,
            system=batch.system,
            nuclear_positions=batch.nuclear_positions,
            nuclear_charges=batch.nuclear_charges,
            spins=None if batch.spins is None else batch.spins[:0],
            aux={},
        )
        return empty, empty
    return _concat_batches(originals), _concat_batches(transformed)


def _rotation_pairs(batch: ElectronBatch, rotations: torch.Tensor) -> tuple[ElectronBatch, ElectronBatch]:
    originals = []
    transformed = []
    for rotation in rotations:
        originals.append(batch)
        transformed.append(
            ElectronBatch(
                positions=torch.einsum("bij,kj->bik", batch.positions, rotation),
                system=batch.system,
                nuclear_positions=_rotate_optional(batch.nuclear_positions, rotation),
                nuclear_charges=batch.nuclear_charges,
                spins=batch.spins,
                aux=dict(batch.aux),
            )
        )
    return _concat_batches(originals), _concat_batches(transformed)


def _concat_batches(batches: Sequence[ElectronBatch]) -> ElectronBatch:
    first = batches[0]
    positions = torch.cat([batch.positions for batch in batches], dim=0)
    spins = None if first.spins is None else torch.cat([batch.spins for batch in batches if batch.spins is not None], dim=0)
    nuclear_positions = _concat_optional_sampled([batch.nuclear_positions for batch in batches], first.batch_size, tail_rank=2)
    nuclear_charges = _concat_optional_sampled([batch.nuclear_charges for batch in batches], first.batch_size, tail_rank=1)
    return ElectronBatch(
        positions=positions,
        system=first.system,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        spins=spins,
        aux={},
    )


def _stack_pair_batch(original: ElectronBatch, transformed: ElectronBatch) -> ElectronBatch:
    positions = torch.stack((original.positions, transformed.positions), dim=1)
    spins = None
    if original.spins is not None and transformed.spins is not None:
        spins = torch.stack((original.spins, transformed.spins), dim=1)
    return ElectronBatch(
        positions=positions,
        system=original.system,
        nuclear_positions=_stack_optional_pair(original.nuclear_positions, transformed.nuclear_positions, original.batch_size, tail_rank=2),
        nuclear_charges=_stack_optional_pair(original.nuclear_charges, transformed.nuclear_charges, original.batch_size, tail_rank=1),
        spins=spins,
        aux={},
    )


def _orbit_metadata(n_base: int, n_transforms: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "base_sample_index": torch.arange(n_base, device=device).repeat(n_transforms),
        "transform_id": torch.arange(n_transforms, device=device, dtype=torch.long).repeat_interleave(n_base),
        "permutation_id": torch.arange(n_transforms, device=device, dtype=torch.long).repeat_interleave(n_base),
        "orbit_id": torch.arange(n_base * n_transforms, device=device),
    }


def _rotation_matrices(
    spatial_dim: int,
    count: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrices = []
    for _ in range(count):
        if spatial_dim == 2:
            angle = 2.0 * torch.pi * torch.rand((), generator=generator, dtype=dtype)
            matrix = torch.stack(
                (
                    torch.stack((torch.cos(angle), -torch.sin(angle))),
                    torch.stack((torch.sin(angle), torch.cos(angle))),
                )
            )
        else:
            random_matrix = torch.randn(spatial_dim, spatial_dim, generator=generator, dtype=dtype)
            q, r = torch.linalg.qr(random_matrix)
            signs = torch.sign(torch.diag(r)).clamp(min=-1.0, max=1.0)
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            matrix = q * signs
            if torch.linalg.det(matrix) < 0:
                matrix[:, 0] = -matrix[:, 0]
        matrices.append(matrix.to(device=device, dtype=dtype))
    return torch.stack(matrices)


def _rotate_optional(value: torch.Tensor | None, rotation: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.einsum("...j,kj->...k", value, rotation)


def _concat_optional_sampled(values: Sequence[torch.Tensor | None], sample_size: int, *, tail_rank: int) -> torch.Tensor | None:
    if not values or values[0] is None:
        return None
    tensors = [_as_sampled(value, sample_size=sample_size, tail_rank=tail_rank) for value in values]
    return torch.cat(tensors, dim=0)


def _stack_optional_pair(
    original: torch.Tensor | None,
    transformed: torch.Tensor | None,
    sample_size: int,
    *,
    tail_rank: int,
) -> torch.Tensor | None:
    if original is None and transformed is None:
        return None
    if original is None or transformed is None:
        raise ValueError("paired batch requires both original and transformed optional tensors")
    return torch.stack(
        (
            _as_sampled(original, sample_size=sample_size, tail_rank=tail_rank),
            _as_sampled(transformed, sample_size=sample_size, tail_rank=tail_rank),
        ),
        dim=1,
    )


def _as_sampled(value: torch.Tensor | None, *, sample_size: int, tail_rank: int) -> torch.Tensor:
    if value is None:
        raise ValueError("expected tensor, got None")
    if value.ndim == tail_rank:
        return value.expand(sample_size, *value.shape)
    if value.ndim == tail_rank + 1 and int(value.shape[0]) == sample_size:
        return value
    raise ValueError(f"cannot align optional tensor with shape {tuple(value.shape)} to {sample_size} samples")


def _exchange_pair(batch: ElectronBatch, exchange: str) -> tuple[int, int]:
    if batch.n_electrons < 2:
        raise ValueError("ExchangeOrbitGenerator requires at least two electrons")
    if batch.spins is None:
        return 0, 1
    spins = batch.spins.reshape(batch.batch_size, batch.n_electrons)[0]
    want_same = exchange == "same_spin_pair"
    for left in range(batch.n_electrons):
        for right in range(left + 1, batch.n_electrons):
            same = bool(spins[left].item() == spins[right].item())
            if same == want_same:
                return left, right
    raise ValueError(f"could not find {exchange!r} in generated batch")


__all__ = [
    "ExchangeOrbitGenerator",
    "PermutationOrbitGenerator",
    "RotationOrbitGenerator",
]
