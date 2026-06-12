"""Shared fixtures for the pair_validation study-script tests.

These tests exercise experiments-owned code; the only spenn surface allowed
is the torch-free checkpoint hash helper that evaluate_selected.py shares
with the Checkpoint callback (physics stays with the Evaluate runner).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parents[1]
# Appended (not prepended) so study modules never shadow stdlib modules —
# select.py shares its name with the stdlib select module.
if str(STUDY_DIR) not in sys.path:
    sys.path.append(str(STUDY_DIR))


@pytest.fixture(scope="session")
def select_mod():
    """Load select.py by path; ``import select`` would hit the stdlib module."""

    import collect  # ensures select.py's `from collect import ...` resolves

    assert collect is not None
    spec = importlib.util.spec_from_file_location("study_select", STUDY_DIR / "select.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["study_select"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    """A small two-axis manifest mirroring the real protocol shape."""

    manifest = {
        "study": {"name": "test_study_v1", "purpose": "validation_scan", "sector": "singlet"},
        "train_config": "experiments/hooke/configs/benchmark/pair_train.yaml",
        "grid": {
            "runtime.seed": [3, 9],
            "optimizer_params.lr": [1.0e-3, 3.0e-3],
            "model_params.channels": [8, 32],
            "model_params.layers": [1],
            "model_params.gate_activation": ["silu"],
        },
        "seed_key": "runtime.seed",
        "validation": {
            "metric": "validation/energy",
            "aggregate": "median",
            "checkpoint": "final",
            "failed_run_value": float("inf"),
        },
        "eligibility": {
            "require": [
                "checks/data_integrity/passed",
                "checks/gradient/passed",
                "checks/equivariance/full_model/passed",
            ],
            "local_energy_finite_fraction": 1.0,
        },
        "selection": {
            "absolute_energy_floor": 1.0e-4,
            "margin": {"stderr_multiplier": 2.0, "seed_iqr_fraction": 0.25},
            "require_all_seeds": True,
            "tie_breakers": [
                "validation/energy_variance",
                "validation_energy_iqr",
                "validation/energy_stderr",
                "geometry_warning_count",
                "model_params.channels",
                "runtime/wall_time_sec",
            ],
        },
        "diagnostic_fields": {
            "sampler_geometry": [
                "validation/sampler/radius_mean",
                "validation/sampler/radius_q99",
                "validation/sampler/radius_max",
                "validation/sampler/electron_distance_q01",
                "validation/sampler/electron_distance_min",
                "validation/sampler/position_rms",
            ]
        },
        "geometry_flags": {"electron_distance_q01_min": 1.0e-3},
        "final_evaluation": {
            "study_name": "test_study_final_v1",
            "eval_config": "experiments/hooke/configs/benchmark/pair_final_eval.yaml",
            "training_seeds": [100, 101],
            "eval_seeds": [100000, 100001],
            "allow_validation_seed_reuse": False,
            "checkpoint_loading": "structured_checkpoint",
            "sampler": {
                "n_walkers": 4096,
                "burn_in": 100,
                "n_steps": 50,
                "proposal_scale": 0.35,
            },
        },
    }
    path = tmp_path / "manifest.yaml"
    # sort_keys=False keeps the grid axis declaration order, which defines
    # the config_id field order.
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


