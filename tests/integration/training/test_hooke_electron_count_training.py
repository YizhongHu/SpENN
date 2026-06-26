"""Integration coverage for Hooke smoke training across small electron counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from spenn.run import run_from_config

CONFIG = Path(__file__).resolve().parents[1] / "artifacts" / "hooke" / "pair_train.yaml"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("n_electrons", "n_up", "n_down"),
    [
        (0, 0, 0),
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 1),
    ],
)
def test_hooke_models_initialize_and_train_for_small_electron_counts(
    tmp_path: Path,
    n_electrons: int,
    n_up: int,
    n_down: int,
) -> None:
    cfg = OmegaConf.load(CONFIG)
    cfg.run.root = str(tmp_path)
    cfg.experiment.name = f"hooke_{n_electrons}_electron_smoke"
    cfg.experiment.sector = f"n{n_electrons}"
    cfg.experiment.run_name = f"hooke_{n_electrons}_electron_train"
    cfg.system.n_particles = n_electrons
    cfg.system.spin.n_up = n_up
    cfg.system.spin.n_down = n_down
    cfg.sampler.n_walkers = 4
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    cfg.trainer.max_steps = 1

    exit_code = run_from_config(cfg, config_path=str(CONFIG), command="pytest")

    assert exit_code == 0
    run_dirs = list((tmp_path / cfg.experiment.name / cfg.experiment.sector).iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_records = [record for record in records if record.get("namespace") == "train"]
    sampler_records = [record for record in records if record.get("namespace") == "train/sampler"]
    assert len(train_records) == 1
    assert len(sampler_records) == 1
    assert sampler_records[0]["metrics"]["n_walkers"] == 4

    train_metrics = train_records[0]["metrics"]
    assert train_metrics["local_energy_n_total"] == 4
    assert train_metrics["loss_has_grad"] is (n_electrons > 0)
    assert train_metrics["optimizer_step"] is (n_electrons > 0)
