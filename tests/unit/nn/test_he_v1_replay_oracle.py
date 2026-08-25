"""Pinned CPU-float64 He-v1 cusp replay fixtures."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from tpen.data.batch import ElectronBatch, pairwise_distances
from tpen.nn import ElectronElectronCusp


# Provenance for every numeric constant below:
# source SHA: 418accf153368aab45586dc2a2cc97c18472691c
# command: srun --exclusive --ntasks=1 --cpus-per-task=4 env
#          PYTHONPATH=<detached-checkout> <lock-provisioned-venv>/bin/python
#          <external-oracle.py> --source-sha <source-sha>
#          --source-root <detached-checkout> --expected-venv <venv>
#          --lock-path <detached-checkout>/uv.lock --output <fixture.json>
# Slurm: job 41499410, test, kozinsky_lab, 4 CPU, 32 GiB, 00:30:00;
# execution host holy8a24101.rc.fas.harvard.edu; CPU, torch.float64, seed 0.
# Driver SHA256: 68983576e123f5bab5ffd4e144457d07ac14a4242fad70f09366d359d1aacebf.
# Durable log: reference-receipt-41499410.log (its facility location is retained
# in the Task Orchestrator receipt, rather than committed to the repository).
# Equality standard for all fixtures: bitwise float.hex().  The reference and
# assertion both execute CPU torch 2.12.0+cpu in float64, so no tolerance is
# needed and a tolerance would hide the replay change this test exists to catch.
REFERENCE_SOURCE_SHA: Final = "418accf153368aab45586dc2a2cc97c18472691c"
REFERENCE_JOB_ID: Final = 41499410

ORDINARY_SOFTENED_DISTANCE: Final = 1.5
ORDINARY_SOFTENED_DISTANCE_HEX: Final = "0x1.8000000000000p+0"
ORDINARY_CUSP_VALUE: Final = 0.3
ORDINARY_CUSP_VALUE_HEX: Final = "0x1.3333333333333p-2"

NEAR_COALESCENCE_SOFTENED_DISTANCE: Final = 1.414213562373095e-12
NEAR_COALESCENCE_SOFTENED_DISTANCE_HEX: Final = "0x1.8e10d3a69204bp-40"
NEAR_COALESCENCE_CUSP_VALUE: Final = 7.071067811855475e-13
NEAR_COALESCENCE_CUSP_VALUE_HEX: Final = "0x1.8e10d3a68f99cp-41"

TAIL_SOFTENED_DISTANCE: Final = 12.0
TAIL_SOFTENED_DISTANCE_HEX: Final = "0x1.8000000000000p+3"
TAIL_CUSP_VALUE: Final = 0.46153846153846156
TAIL_CUSP_VALUE_HEX: Final = "0x1.d89d89d89d89ep-2"


@pytest.mark.parametrize(
    (
        "positions",
        "expected_distance",
        "expected_distance_hex",
        "expected_cusp",
        "expected_cusp_hex",
    ),
    [
        (
            [[0.25, -0.5, 0.125], [-0.75, 0.5, -0.375]],
            ORDINARY_SOFTENED_DISTANCE,
            ORDINARY_SOFTENED_DISTANCE_HEX,
            ORDINARY_CUSP_VALUE,
            ORDINARY_CUSP_VALUE_HEX,
        ),
        (
            [[0.0, 0.0, 0.0], [6.0e-13, 8.0e-13, 0.0]],
            NEAR_COALESCENCE_SOFTENED_DISTANCE,
            NEAR_COALESCENCE_SOFTENED_DISTANCE_HEX,
            NEAR_COALESCENCE_CUSP_VALUE,
            NEAR_COALESCENCE_CUSP_VALUE_HEX,
        ),
        (
            [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            TAIL_SOFTENED_DISTANCE,
            TAIL_SOFTENED_DISTANCE_HEX,
            TAIL_CUSP_VALUE,
            TAIL_CUSP_VALUE_HEX,
        ),
    ],
    ids=("ordinary", "near_coalescence", "tail"),
)
def test_he_v1_cusp_replays_the_pinned_v031_cpu_float64_oracle(
    positions: list[list[float]],
    expected_distance: float,
    expected_distance_hex: str,
    expected_cusp: float,
    expected_cusp_hex: str,
) -> None:
    """Pin the executed softened e-e distance and cusp contribution together."""

    torch.manual_seed(0)
    batch = ElectronBatch(
        positions=torch.tensor([positions], device="cpu", dtype=torch.float64),
        spins=torch.tensor([[1, -1]], device="cpu"),
    )
    cusp = ElectronElectronCusp(trainable_range=True).to(
        device="cpu", dtype=torch.float64
    )
    distance = pairwise_distances(batch.positions, eps=cusp.eps)[0, 0, 1, 0].item()
    value = cusp.envelope_value(batch)[0].item()

    assert distance.hex() == expected_distance_hex
    assert distance.hex() == expected_distance.hex()
    assert value.hex() == expected_cusp_hex
    assert value.hex() == expected_cusp.hex()
