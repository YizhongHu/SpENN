"""Tests for MCMC-backed evaluation generators."""

from __future__ import annotations

from pathlib import Path

import torch

from spenn.data.batch import ElectronBatch, Walkers
from spenn.evaluation.generators import MCMCGenerator
from spenn.evaluation.protocols import EvaluationContext


class RecordingSampler:
    def collect_samples(self, model, *, device=None):
        positions = torch.zeros(2, 1, 3, dtype=torch.float64)
        walkers = Walkers(positions=positions)
        return walkers, {"acceptance_rate": 1.0}


def test_mcmc_generator_seed_does_not_mutate_global_torch_rng() -> None:
    torch.manual_seed(999)
    before = torch.get_rng_state()
    generator = MCMCGenerator(sampler=RecordingSampler(), seed=123)
    context = EvaluationContext(
        namespace="eval",
        artifact_level="metrics_only",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=123,
        run_dir=Path("/tmp"),
        task_output_dir=Path("/tmp"),
        metadata={},
    )

    generated = generator.generate(model=None, context=context)

    assert isinstance(generated.batch, ElectronBatch)
    assert generated.metadata["seed"] == 123
    torch.testing.assert_close(torch.get_rng_state(), before)
