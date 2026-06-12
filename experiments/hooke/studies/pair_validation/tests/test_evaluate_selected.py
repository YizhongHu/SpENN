"""Tests for the pair_validation final evaluator script (experiments-owned)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import evaluate_selected
from spenn.callback.checkpoint import model_config_hash

STUDY_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def configs(tmp_path: Path) -> dict[str, Path]:
    """Minimal train/eval config stubs declaring run-dir identity and model."""

    train_config = tmp_path / "pair_train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "hooke_pair_benchmark", "sector": "pair"},
                "model_params": {"channels": 4, "layers": 1, "gate_activation": "silu"},
                "optimizer_params": {"lr": 3.0e-4},
                # Interpolations mirror the real config: the planned model spec
                # must resolve through the selected overrides.
                "model": {
                    "_target_": "spenn.nn.SpENNWaveFunction",
                    "channels": "${model_params.channels}",
                    "layers": "${model_params.layers}",
                    "gate": "${model_params.gate_activation}",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    eval_config = tmp_path / "pair_final_eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "hooke_pair_benchmark", "sector": "pair"},
                "run": {"root": "outputs", "run_id": None},
                "study": {"name": None, "config_id": None},
                "runtime": {"seed": 0},
                "model_params": {"channels": 4, "layers": 1, "gate_activation": "silu"},
                "model": {
                    "_target_": "spenn.nn.SpENNWaveFunction",
                    "channels": "${model_params.channels}",
                },
                "sampler_params": {
                    "n_walkers": 8192,
                    "burn_in": 1000,
                    "n_steps": 500,
                    "proposal_scale": 0.35,
                },
                "evaluation": {"checkpoint": "???", "training_seed": None},
            },
            sort_keys=False,
        ),
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


def _generated_eval_config(output_dir: Path, training_seed: int, eval_seed: int) -> dict:
    path = output_dir / f"final_eval_config_seed{training_seed}_eval{eval_seed}.yaml"
    assert path.is_file()
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_dry_run_writes_outputs(tmp_path, manifest_path, configs, selected_config_path) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    assert (output_dir / "final_eval_commands.sh").is_file()
    assert (output_dir / "final_eval_manifest.yaml").is_file()
    assert (output_dir / "final_eval_inputs.csv").is_file()
    # One self-contained generated eval config per (training seed, eval seed).
    assert (output_dir / "final_eval_config_seed100_eval100000.yaml").is_file()
    assert (output_dir / "final_eval_config_seed101_eval100001.yaml").is_file()
    # Dry-run must not execute anything: no run directories, no run table.
    assert not (tmp_path / "outputs").exists()
    assert not (output_dir / "final_eval_runs.csv").exists()


def test_commands_use_selected_hyperparameters(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    commands = (output_dir / "final_eval_commands.sh").read_text(encoding="utf-8")
    # Training stage: selected hyperparameters and fresh training seeds.
    assert "model_params.channels=8" in commands
    assert "model_params.gate_activation=silu" in commands
    assert "optimizer_params.lr=0.001" in commands
    assert "runtime.seed=100" in commands
    # Validation seeds (3, 9) never appear.
    assert "runtime.seed=3" not in commands
    assert "runtime.seed=9" not in commands
    # Eval stage: each command runs one generated, self-contained config.
    assert "final_eval_config_seed100_eval100000.yaml" in commands

    generated = _generated_eval_config(output_dir, 100, 100000)
    # Everything eval-side is baked into the generated config.
    assert generated["runtime"]["seed"] == 100000
    assert generated["sampler_params"]["n_walkers"] == 4096  # manifest sampler
    assert generated["study"]["name"] == "test_study_final_v1"
    assert generated["study"]["config_id"] == "lr=0.001_channels=8_layers=1_gate_activation=silu"
    assert generated["evaluation"]["checkpoint"].endswith(
        "final_train_seed100/checkpoints/latest.pt"
    )
    assert generated["evaluation"]["training_seed"] == 100


def test_generated_eval_config_carries_explicit_model_spec(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    generated = _generated_eval_config(output_dir, 100, 100000)
    # The model spec is the resolved training spec (selected overrides applied),
    # not an interpolation template and not inferred from checkpoint keys.
    assert generated["model"] == {
        "_target_": "spenn.nn.SpENNWaveFunction",
        "channels": 8,
        "layers": 1,
        "gate": "silu",
    }
    # The knob block stays consistent with the explicit spec.
    assert generated["model_params"]["channels"] == 8
    # Training-only hyperparameters never enter the eval config.
    assert "optimizer_params" not in generated
    # Strict, hash-verified structured loading.
    assert generated["evaluation"]["checkpoint_strict"] is True
    assert generated["evaluation"]["allow_model_config_mismatch"] is False
    assert generated["evaluation"]["expected_model_config_hash"] == model_config_hash(
        generated["model"]
    )
    assert generated["checkpoint_loading"]["mode"] == "structured_checkpoint"
    assert generated["checkpoint_loading"]["model_config_hash_verified"] is True


def test_expected_hash_matches_training_command_resolution(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    """The baked hash equals what the planned training command will checkpoint."""

    from spenn.run import load_config  # the same loader run.py uses
    from omegaconf import OmegaConf

    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    output_dir = _run_dry(tmp_path, manifest, configs, selected_config_path)

    with open(output_dir / "final_eval_inputs.csv", encoding="utf-8", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))
    train_command = (output_dir / "final_eval_commands.sh").read_text(encoding="utf-8")
    assert "runtime.seed=100" in train_command

    # Replay the training command's config resolution exactly as run.py does.
    cfg = load_config(
        str(configs["train"]),
        [
            "runtime.seed=100",
            "optimizer_params.lr=0.001",
            "model_params.channels=8",
            "model_params.layers=1",
            "model_params.gate_activation=silu",
        ],
    )
    training_model = OmegaConf.to_container(OmegaConf.select(cfg, "model"), resolve=True)
    assert row["source_model_config_hash"] == model_config_hash(training_model)

    generated = _generated_eval_config(output_dir, 100, 100000)
    assert generated["evaluation"]["expected_model_config_hash"] == model_config_hash(
        training_model
    )


def _make_train_run(
    root: Path, *, study_name: str, training_seed: int, model_spec: dict
) -> Path:
    """Fake completed training run with a resolved config (legacy workaround)."""

    run_dir = (
        root / study_name / "hooke_pair_benchmark" / "pair" / f"final_train_seed{training_seed}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"runtime": {"seed": training_seed}, "model": model_spec}),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"git_commit": "trainsha"}), encoding="utf-8"
    )
    return run_dir


def test_legacy_workaround_copies_model_spec_from_resolved_config(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(
        manifest_path, tmp_path, configs, checkpoint_loading="legacy_resolved_config_workaround"
    )
    model_spec = {"_target_": "spenn.nn.SpENNWaveFunction", "channels": 8, "gate": "silu"}
    run_root = tmp_path / "outputs"
    for seed in (100, 101):
        _make_train_run(
            run_root, study_name="test_study_final_v1", training_seed=seed, model_spec=model_spec
        )

    output_dir = tmp_path / "reports"
    exit_code = evaluate_selected.main(
        [
            "--manifest", str(manifest),
            "--selected-config", str(selected_config_path),
            "--run-root", str(run_root),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0

    generated = _generated_eval_config(output_dir, 100, 100000)
    # Explicit model spec copied verbatim from the training resolved config.
    assert generated["model"] == model_spec
    # Legacy checkpoints carry no hash: loading stays strict but unverified.
    assert generated["evaluation"]["expected_model_config_hash"] is None
    assert generated["evaluation"]["checkpoint_strict"] is True
    assert generated["checkpoint_loading"]["mode"] == "legacy_resolved_config_workaround"
    assert generated["checkpoint_loading"]["model_config_hash_verified"] is False
    assert generated["checkpoint_loading"]["source_resolved_config"].endswith(
        "final_train_seed100/resolved_config.yaml"
    )

    with open(output_dir / "final_eval_manifest.yaml", encoding="utf-8") as handle:
        final_manifest = yaml.safe_load(handle)
    assert final_manifest["checkpoint_loading"]["mode"] == "legacy_resolved_config_workaround"
    assert final_manifest["checkpoint_loading"]["strict"] is True
    assert final_manifest["checkpoint_loading"]["model_config_hash_verified"] is False
    assert final_manifest["runs"][0]["source_train_git_sha"] == "trainsha"
    # The hash of the copied spec is still recorded for future verification.
    assert final_manifest["runs"][0]["source_model_config_hash"] == model_config_hash(model_spec)


def test_legacy_workaround_requires_existing_training_runs(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(
        manifest_path, tmp_path, configs, checkpoint_loading="legacy_resolved_config_workaround"
    )
    with pytest.raises(ValueError, match="needs completed training runs"):
        evaluate_selected.build_plan(
            evaluate_selected.load_manifest(manifest),
            evaluate_selected.load_selected_config(selected_config_path),
            run_root=tmp_path / "outputs",
            train_config=configs["train"],
            eval_config=configs["eval"],
            output_dir=tmp_path / "reports",
        )


def test_unknown_checkpoint_loading_mode_is_rejected(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    manifest = _patched_manifest(manifest_path, tmp_path, configs, checkpoint_loading="trust_me")
    with pytest.raises(ValueError, match="checkpoint_loading"):
        evaluate_selected.final_evaluation_policy(evaluate_selected.load_manifest(manifest))


def test_structured_mode_prefers_existing_training_resolved_config(
    tmp_path, manifest_path, configs, selected_config_path
) -> None:
    # When a training run already exists, its resolved config is authoritative
    # even in structured mode (and the hash pin comes from it).
    manifest = _patched_manifest(manifest_path, tmp_path, configs)
    model_spec = {"_target_": "spenn.nn.SpENNWaveFunction", "channels": 8, "gate": "silu"}
    run_root = tmp_path / "outputs2"
    _make_train_run(
        run_root, study_name="test_study_final_v1", training_seed=100, model_spec=model_spec
    )

    plan = evaluate_selected.build_plan(
        evaluate_selected.load_manifest(manifest),
        evaluate_selected.load_selected_config(selected_config_path),
        run_root=run_root,
        train_config=configs["train"],
        eval_config=configs["eval"],
        output_dir=tmp_path / "reports",
    )
    assert plan[0]["eval_config"]["model"] == model_spec
    assert plan[0]["source_resolved_config"].endswith("resolved_config.yaml")
    assert plan[0]["eval_config"]["evaluation"]["expected_model_config_hash"] == (
        model_config_hash(model_spec)
    )
    # Seed 101 has no run yet: spec falls back to the planned resolution.
    assert plan[1]["source_resolved_config"] is None
    assert plan[1]["eval_config"]["model"]["channels"] == 8


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
    assert final_manifest["checkpoint_loading"]["mode"] == "structured_checkpoint"
    assert final_manifest["checkpoint_loading"]["model_config_hash_verified"] is True
    run = final_manifest["runs"][0]
    assert run["source_checkpoint_path"].endswith("final_train_seed100/checkpoints/latest.pt")
    assert run["final_eval_config_path"].endswith("final_eval_config_seed100_eval100000.yaml")

    with open(output_dir / "final_eval_inputs.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # one row per (training seed, eval seed) pair
    assert rows[0]["training_seed"] == "100"
    assert rows[0]["eval_seed"] == "100000"
    assert rows[0]["source_checkpoint_path"].endswith("final_train_seed100/checkpoints/latest.pt")
    assert rows[0]["source_model_config_hash"]
    assert rows[0]["final_eval_config_path"].endswith("final_eval_config_seed100_eval100000.yaml")


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
    assert "spenn.diagnostics" not in source
    assert "import torch" not in source
    # The only spenn surface used is the torch-free checkpoint hash helper,
    # shared with the Checkpoint callback so the pairing hash has one owner.
    assert "from spenn.callback.checkpoint import model_config_hash" in source
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
