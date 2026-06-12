"""Tests for the pair_validation final evaluator script (experiments-owned)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import evaluate_selected

STUDY_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def configs(tmp_path: Path) -> dict[str, Path]:
    """Minimal train/eval config stubs declaring the run-dir identity."""

    train_config = tmp_path / "pair_train.yaml"
    train_config.write_text(
        yaml.safe_dump({"experiment": {"name": "hooke_pair_benchmark", "sector": "pair"}}),
        encoding="utf-8",
    )
    eval_config = tmp_path / "pair_final_eval.yaml"
    eval_config.write_text(
        yaml.safe_dump({"experiment": {"name": "hooke_pair_benchmark", "sector": "pair"}}),
        encoding="utf-8",
    )
    return {"train": train_config, "eval": eval_config}


@pytest.fixture()
def selected_config_path(tmp_path: Path, configs: dict[str, Path]) -> Path:
    payload = {
        "study": "test_study_v1",
        "train_config": str(configs["train"]),
        "selected": {
            "config_id": "lr=0.001_channels=8_layers=1_gate_activation=silu",
            "optimizer_params.lr": "0.001",
            "model_params.channels": "8",
            "model_params.layers": "1",
            "model_params.gate_activation": "silu",
        },
        "overrides": [
            "optimizer_params.lr=0.001",
            "model_params.channels=8",
            "model_params.layers=1",
            "model_params.gate_activation=silu",
        ],
    }
    path = tmp_path / "selected_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _patched_manifest(manifest_path: Path, tmp_path: Path, configs: dict[str, Path], **overrides) -> Path:
    """Rewrite the fixture manifest to point at the stub eval config."""

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    manifest["final_evaluation"]["eval_config"] = str(configs["eval"])
    manifest["final_evaluation"].update(overrides)
    path = tmp_path / "manifest_final.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _run_dry(tmp_path, manifest_path, configs, selected_config_path):
    output_dir = tmp_path / "reports"
    exit_code = evaluate_selected.main(
        [
            "--manifest", str(manifest_path),
            "--selected-config", str(selected_config_path),
            "--run-root", str(tmp_path / "outputs"),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    return output_dir


def test_dry_run_writes_outputs(tmp_path, manifest_path, configs, selected_config_path) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    assert (output_dir / "final_eval_commands.sh").is_file()
    assert (output_dir / "final_eval_manifest.yaml").is_file()
    assert (output_dir / "final_eval_inputs.csv").is_file()
    # Dry-run must not execute anything: no run directories, no run table.
    assert not (tmp_path / "outputs").exists()
    assert not (output_dir / "final_eval_runs.csv").exists()


def test_commands_use_selected_hyperparameters(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    commands = (output_dir / "final_eval_commands.sh").read_text(encoding="utf-8")
    assert "model_params.channels=8" in commands
    assert "model_params.gate_activation=silu" in commands
    assert "optimizer_params.lr=0.001" in commands  # training stage
    assert str(configs["eval"]) in commands
    # Final seeds, paired index-wise; validation seeds (3, 9) never appear.
    assert "runtime.seed=100" in commands
    assert "runtime.seed=100000" in commands
    assert "runtime.seed=3" not in commands
    assert "runtime.seed=9" not in commands
    # Eval stage restores the paired checkpoint with the manifest sampler.
    assert "checkpoints/latest.pt" in commands
    assert "sampler_params.n_walkers=4096" in commands


def test_eval_commands_omit_training_only_overrides(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    with open(output_dir / "final_eval_inputs.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # one row per (training seed, eval seed) pair
    assert rows[0]["training_seed"] == "100"
    assert rows[0]["eval_seed"] == "100000"
    assert rows[0]["checkpoint"].endswith("final_train_seed100/checkpoints/latest.pt")

    # The eval command rebuilds the architecture but never passes the lr.
    plan = evaluate_selected.build_plan(
        evaluate_selected.load_manifest(manifest),
        evaluate_selected.load_selected_config(selected_config_path),
        run_root=tmp_path / "outputs",
        train_config=configs["train"],
        eval_config=configs["eval"],
    )
    for entry in plan:
        assert not any(arg.startswith("optimizer_params.") for arg in entry["eval_command"])
        assert any(arg == "model_params.channels=8" for arg in entry["eval_command"])
        assert any(arg.startswith("evaluation.training_seed=") for arg in entry["eval_command"])


def test_final_manifest_records_provenance(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    with open(output_dir / "final_eval_manifest.yaml", encoding="utf-8") as handle:
        final_manifest = yaml.safe_load(handle)
    assert final_manifest["study"]["source_validation_study"] == "test_study_v1"
    assert final_manifest["selected"]["config_id"] == "lr=0.001_channels=8_layers=1_gate_activation=silu"
    assert final_manifest["final_training_seeds"] == [100, 101]
    assert final_manifest["final_eval_seeds"] == [100000, 100001]
    assert final_manifest["final_eval_sampler"]["n_walkers"] == 4096
    assert "selection_report" in final_manifest
    assert "exact_reference" in final_manifest


def test_validation_seed_reuse_is_rejected(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    # Seed 3 is a validation grid seed; reusing it must fail loudly...
    manifest = _patched_manifest(
        manifest_path, tmp_path, configs, training_seeds=[3, 101], eval_seeds=[100000, 100001]
    )
    with pytest.raises(ValueError, match="reuse validation seeds"):
        evaluate_selected.final_evaluation_policy(evaluate_selected.load_manifest(manifest))

    # ...unless the manifest explicitly allows it.
    (tmp_path / "allowed").mkdir(exist_ok=True)
    manifest_allowed = _patched_manifest(
        manifest_path,
        tmp_path / "allowed",
        configs,
        training_seeds=[3, 101],
        eval_seeds=[100000, 100001],
        allow_validation_seed_reuse=True,
    )
    policy = evaluate_selected.final_evaluation_policy(
        evaluate_selected.load_manifest(manifest_allowed)
    )
    assert policy["training_seeds"] == [3, 101]


def test_script_delegates_physics_and_never_uses_wandb() -> None:
    source = (STUDY_DIR / "evaluate_selected.py").read_text(encoding="utf-8")
    # Physics diagnostics belong to the Evaluate runner, not this script.
    assert "EnergyEvaluation" not in source
    assert "import spenn" not in source
    assert "import torch" not in source
    # Local outputs are authoritative; the script never talks to W&B.
    assert "import wandb" not in source


def _make_final_eval_run(
    root: Path,
    *,
    training_seed: int,
    eval_seed: int,
    energy: float = 2.001,
    status: str = "completed",
) -> Path:
    run_dir = (
        root
        / "test_study_final_v1"
        / "hooke_pair_benchmark"
        / "pair"
        / f"final_eval_seed{training_seed}_eval{eval_seed}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"git_commit": "deadbeef", "wandb_run_id": ""}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    resolved = {
        "study": {"name": "test_study_final_v1", "config_id": "lr=0.001_channels=8_layers=1_gate_activation=silu"},
        "runtime": {"seed": eval_seed},
        "evaluation": {
            "checkpoint": f".../final_train_seed{training_seed}/checkpoints/latest.pt",
            "training_seed": training_seed,
        },
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    records = [
        {
            "step": 0,
            "namespace": "eval",
            "metrics": {
                "energy": energy,
                "energy_stderr": 0.001,
                "energy_variance": 0.05,
                "energy_error": energy - 2.0,
                "energy_abs_error": abs(energy - 2.0),
            },
        },
        {
            "step": 0,
            "namespace": "eval/sampler",
            "metrics": {
                "acceptance_rate": 0.7,
                "radius_mean": 1.2,
                "radius_q99": 3.0,
                "electron_distance_q01": 0.5,
            },
        },
        {"step": 0, "namespace": "runtime", "metrics": {"wall_time_sec": 100.0}},
    ]
    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return run_dir


def test_collect_mode_writes_final_summary(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    run_root = tmp_path / "outputs"
    _make_final_eval_run(run_root, training_seed=100, eval_seed=100000, energy=2.001)
    _make_final_eval_run(run_root, training_seed=101, eval_seed=100001, energy=2.003)
    # A training run in the same tree must be skipped (no evaluation.checkpoint).
    train_dir = run_root / "test_study_final_v1" / "hooke_pair_benchmark" / "pair" / "final_train_seed100"
    train_dir.mkdir(parents=True)
    (train_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (train_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"runtime": {"seed": 100}}), encoding="utf-8"
    )

    output_dir = tmp_path / "reports"
    exit_code = evaluate_selected.main(
        [
            "--manifest", str(manifest),
            "--run-root", str(run_root),
            "--output-dir", str(output_dir),
            "--collect",
        ]
    )

    assert exit_code == 0
    with open(output_dir / "final_benchmark_summary.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # eval runs only; the training run is skipped
    assert rows[0]["config_id"] == "lr=0.001_channels=8_layers=1_gate_activation=silu"
    assert rows[0]["training_seed"] == "100"
    assert rows[0]["eval_seed"] == "100000"
    assert float(rows[0]["eval/energy"]) == pytest.approx(2.001)
    assert float(rows[0]["eval/energy_abs_error"]) == pytest.approx(0.001)
    assert rows[0]["status"] == "completed"
    assert rows[0]["git/sha"] == "deadbeef"

    assert (output_dir / "final_benchmark_summary.json").is_file()
    report = (output_dir / "final_benchmark_report.md").read_text(encoding="utf-8")
    assert "Median eval/energy" in report
    assert "W&B is visualization only" in report
