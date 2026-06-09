"""Unit tests for Metropolis sampling helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.sampling import MALASampler
from spenn.sampling.metropolis import MetropolisSampler
from spenn.sampling.moves import GaussianMove


class ShiftMove(nn.Module):
    def propose(
        self,
        walkers: Walkers,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del generator
        positions = walkers.positions + 1.0
        log_q_ratio = torch.zeros(walkers.batch_size, device=walkers.device, dtype=walkers.dtype)
        return positions, log_q_ratio


class NoGradLinearModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        assert not torch.is_grad_enabled()
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class QuadraticLogAbsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_position_grad = False

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if torch.is_grad_enabled() and batch.positions.requires_grad:
            self.saw_position_grad = True
        logabs = -0.5 * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_metropolis_sampler_uses_logabs_ratio_and_caches_values() -> None:
    sampler = MetropolisSampler(move=ShiftMove(), n_walkers=3, n_electrons=2, spatial_dim=1, dtype=torch.float64)
    walkers = sampler.initialize(n_walkers=3, device="cpu")
    spins = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros_like(walkers.positions), spins=spins, aux=walkers.aux)

    stepped = sampler.step(NoGradLinearModel(), walkers)

    assert torch.equal(stepped.positions, torch.ones_like(stepped.positions))
    assert torch.equal(stepped.spins, spins)
    assert torch.equal(stepped.logabs, torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.equal(stepped.sign, torch.ones(3, dtype=torch.float64))
    assert torch.equal(stepped.aux["accepted"], torch.ones(3, dtype=torch.bool))
    assert sampler.acceptance_rate == 1.0


def test_metropolis_initialize_builds_spins_from_partition() -> None:
    sampler = MetropolisSampler(n_walkers=5, n_electrons=2, spatial_dim=3, n_up=1, n_down=1, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (5, 2, 3)
    assert walkers.spins is not None
    assert torch.equal(walkers.spins, torch.tensor([[1.0, -1.0]], dtype=torch.float64).expand(5, -1))


def test_metropolis_initialize_without_partition_has_no_spins() -> None:
    sampler = MetropolisSampler(n_walkers=4, n_electrons=2, spatial_dim=2, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (4, 2, 2)
    assert walkers.spins is None


def test_gaussian_single_electron_move_preserves_shape_and_changes_one_electron_per_walker() -> None:
    torch.manual_seed(0)
    positions = torch.zeros(5, 3, 2, dtype=torch.float64)
    walkers = Walkers(positions=positions)
    move = GaussianMove(step_size=0.1, move_all=False)

    proposals, log_q_ratio = move.propose(walkers)
    changed = (proposals != positions).any(dim=-1)

    assert proposals.shape == positions.shape
    assert log_q_ratio.shape == (5,)
    assert torch.allclose(log_q_ratio, torch.zeros(5, dtype=torch.float64))
    assert torch.equal(changed.sum(dim=1), torch.ones(5, dtype=torch.int64))


def _tiny_sampler() -> MetropolisSampler:
    return MetropolisSampler(
        n_walkers=8,
        burn_in=3,
        n_steps=2,
        proposal_scale=0.3,
        seed=123,
        n_electrons=2,
        spatial_dim=1,
        n_up=1,
        n_down=1,
        dtype=torch.float64,
    )


def test_sampler_local_rng_does_not_mutate_global_state() -> None:
    before = torch.get_rng_state()
    sampler = _tiny_sampler()
    sampler.collect_samples(NoGradLinearModel())
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_sampler_seed_is_reproducible_across_instances() -> None:
    model = NoGradLinearModel()
    first, _ = _tiny_sampler().collect_samples(model)
    second, _ = _tiny_sampler().collect_samples(model)

    assert torch.equal(first.positions, second.positions)


def test_collect_samples_burns_in_once_and_advances_chain() -> None:
    sampler = _tiny_sampler()
    model = NoGradLinearModel()

    first, _ = sampler.collect_samples(model)
    assert sampler.has_burned_in is True
    second, _ = sampler.collect_samples(model)

    # The chain advanced (no re-burn, no reset) so the second draw differs.
    assert not torch.equal(first.positions, second.positions)


def test_collect_samples_reset_restarts_the_chain() -> None:
    model = NoGradLinearModel()
    sampler = _tiny_sampler()

    sampler.collect_samples(model)
    reset_draw, _ = sampler.collect_samples(model, reset=True)
    fresh_draw, _ = _tiny_sampler().collect_samples(model)

    # reset=True re-seeds, so it matches a fresh sampler's first draw.
    assert torch.equal(reset_draw.positions, fresh_draw.positions)


def test_sampler_state_dict_roundtrip_continues_same_chain() -> None:
    model = NoGradLinearModel()
    sampler = _tiny_sampler()
    sampler.collect_samples(model)
    state = sampler.state_dict()

    resumed = _tiny_sampler()
    resumed.load_state_dict(state)

    expected, _ = sampler.collect_samples(model)
    actual, _ = resumed.collect_samples(model)

    assert resumed.has_burned_in is True
    assert torch.equal(expected.positions, actual.positions)


def test_initialize_rejects_mismatched_device() -> None:
    import pytest

    sampler = _tiny_sampler()
    with pytest.raises(ValueError, match="reset"):
        sampler.initialize(device="meta")


def test_mala_sampler_uses_logabs_gradients_and_caches_valid_walkers() -> None:
    torch.manual_seed(0)
    model = QuadraticLogAbsModel()
    sampler = MALASampler(proposal_scale=0.05, n_walkers=4, n_electrons=2, spatial_dim=1, dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros(4, 2, 1, dtype=torch.float64))

    stepped = sampler.step(model, walkers)

    assert model.saw_position_grad
    assert stepped.positions.shape == walkers.positions.shape
    assert stepped.logabs is not None and stepped.logabs.shape == (4,)
    assert stepped.sign is not None and stepped.sign.shape == (4,)
    assert stepped.aux["accepted"].shape == (4,)
    assert stepped.aux["log_accept_ratio"].shape == (4,)
    assert torch.isfinite(stepped.logabs).all()
    assert 0.0 <= sampler.acceptance_rate <= 1.0
