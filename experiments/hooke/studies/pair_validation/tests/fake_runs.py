"""Tiny fake run-directory builder matching the SpENN run-output contract.

Experiments-owned test helper; must not import ``spenn``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def make_run_dir(
    root: Path,
    *,
    seed: int,
    lr: float = 1.0e-3,
    channels: int = 8,
    gate: str = "silu",
    energy: float = 2.0,
    energy_variance: float = 0.5,
    finite_fraction: float = 1.0,
    status: str = "completed",
    checks_passed: bool = True,
    with_validation: bool = True,
    with_geometry: bool = True,
    electron_distance_q01: float = 0.5,
    wall_time: float = 10.0,
) -> Path:
    """Write one fake run directory and return its path."""

    run_dir = root / f"run_lr{lr:g}_c{channels}_{gate}_seed{seed}"
    run_dir.mkdir(parents=True)

    (run_dir / "metadata.json").write_text(
        json.dumps({"git_commit": "deadbeef", "run_id": run_dir.name}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    resolved = {
        "study": {"name": "test_study_v1", "config_id": None},
        "runtime": {"seed": seed},
        "optimizer_params": {"lr": lr},
        "model_params": {"channels": channels, "layers": 1, "gate_activation": gate},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")

    records = [
        {"step": 2, "namespace": "checks/data_integrity", "metrics": {"passed": checks_passed}},
        {"step": 2, "namespace": "checks/gradient", "metrics": {"passed": True}},
        {"step": 2, "namespace": "checks/equivariance/full_model", "metrics": {"passed": True}},
        {"step": 0, "namespace": "runtime", "metrics": {"wall_time_sec": wall_time}},
    ]
    if with_validation:
        records.append(
            {
                "step": 2,
                "namespace": "validation",
                "metrics": {
                    "energy": energy,
                    "energy_variance": energy_variance,
                    "energy_stderr": 0.1,
                    "local_energy_finite_fraction": finite_fraction,
                },
            }
        )
        sampler_metrics = {
            "acceptance_rate": 0.7,
            "n_walkers": 2048,
            "burn_in": 500,
            "n_steps": 200,
            "proposal_scale": 0.35,
            "seed": 114514,
            "n_electrons": 2,
        }
        if with_geometry:
            sampler_metrics.update(
                {
                    "radius_mean": 1.2,
                    "radius_q99": 3.0,
                    "radius_max": 4.0,
                    "electron_distance_q01": electron_distance_q01,
                    "electron_distance_min": 0.3,
                    "position_rms": 1.0,
                }
            )
        records.append({"step": 2, "namespace": "validation/sampler", "metrics": sampler_metrics})
    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return run_dir
