"""Runtime seed behavior for configured runs."""

from __future__ import annotations

from omegaconf import OmegaConf

import spenn.run as run_module
from spenn.dependencies import require_torch


def test_runtime_seed_seeds_torch_rng_for_torch_runner() -> None:
    """``runtime.seed`` makes Torch draws reproducible before runner construction."""

    torch = require_torch(feature="runtime seed test")
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 1234},
            "runner": {"_target_": "spenn.runner.Train"},
        }
    )

    torch.manual_seed(9999)
    run_module._seed_runtime_rngs(cfg)
    first = torch.rand(4)

    torch.manual_seed(9999)
    run_module._seed_runtime_rngs(cfg)
    second = torch.rand(4)

    assert torch.equal(first, second)
