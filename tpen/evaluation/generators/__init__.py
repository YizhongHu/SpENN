"""Evaluation configuration generators."""

from __future__ import annotations

from tpen.evaluation.generators.atom import HeliumRadialGridGenerator
from tpen.evaluation.generators.hooke import (
    CuspGridGenerator,
    HookeOrbitalGenerator,
    StratifiedGeometryGenerator,
    TailGridGenerator,
)
from tpen.evaluation.generators.mcmc import MCMCGenerator
from tpen.evaluation.generators.orbits import (
    ExchangeOrbitGenerator,
    PermutationOrbitGenerator,
    RotationOrbitGenerator,
)
from tpen.evaluation.generators.trajectory import (
    SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY,
    TRAJECTORY_METADATA_KEY,
    TrajectoryMCMCGenerator,
)

__all__ = [
    "SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY",
    "TRAJECTORY_METADATA_KEY",
    "CuspGridGenerator",
    "ExchangeOrbitGenerator",
    "HookeOrbitalGenerator",
    "HeliumRadialGridGenerator",
    "MCMCGenerator",
    "PermutationOrbitGenerator",
    "RotationOrbitGenerator",
    "StratifiedGeometryGenerator",
    "TailGridGenerator",
    "TrajectoryMCMCGenerator",
]
