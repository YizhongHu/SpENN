"""Tests for the pair_validation collector script (experiments-owned)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fake_runs import make_run_dir

import collect

REAL_MANIFEST = Path(__file__).resolve().parents[1] / "manifest.yaml"


def test_real_manifest_loads() -> None:
    manifest = collect.load_manifest(REAL_MANIFEST)
    assert manifest["study"]["name"] == "hooke_pair_validation_v1"
    assert manifest["seed_key"] == "runtime.seed"
    assert manifest["validation"]["metric"] == "validation/energy"
    assert manifest["validation"]["failed_run_value"] == float("inf")
    # The selection rule must not use the exact reference energy.
    assert "reference_energy" not in REAL_MANIFEST.read_text(encoding="utf-8")


def test_config_id_generation_is_deterministic(manifest_path: Path) -> None:
    manifest = collect.load_manifest(manifest_path)
    values = {
        "optimizer_params.lr": 1.0e-3,
        "model_params.channels": 32,
        "model_params.layers": 1,
        "model_params.gate_activation": "silu",
        "runtime.seed": 3,
    }
    config_id = collect.config_id_from_values(manifest, values)
    assert config_id == "lr=0.001_channels=32_layers=1_gate_activation=silu"
    # Stable across calls and independent of the seed value.
    values["runtime.seed"] = 9
    assert collect.config_id_from_values(manifest, values) == config_id


def test_collector_reads_fake_run_dir(tmp_path: Path, manifest_path: Path) -> None:
    manifest = collect.load_manifest(manifest_path)
    run_dir = make_run_dir(tmp_path / "runs", seed=3, energy=2.25)

    row = collect.collect_run(run_dir, manifest)

    assert row["status"] == "completed"
    assert row["study_name"] == "test_study_v1"
    assert row["config_id"] == "lr=0.001_channels=8_layers=1_gate_activation=silu"
    assert row["runtime.seed"] == 3
    assert row["validation/energy"] == 2.25
    assert row["validation/sampler/radius_q99"] == 3.0
    assert row["checks/data_integrity/passed"] is True
    assert row["git/sha"] == "deadbeef"


def test_collector_handles_failed_and_incomplete_runs(tmp_path: Path, manifest_path: Path) -> None:
    manifest = collect.load_manifest(manifest_path)
    failed = make_run_dir(
        tmp_path / "runs",
        seed=3,
        status="failed",
        with_validation=False,
        exception_type="OutOfMemoryError",
        exception_message="CUDA out of memory",
    )
    incomplete = make_run_dir(tmp_path / "runs", seed=9, status="running", with_validation=False)

    failed_row = collect.collect_run(failed, manifest)
    incomplete_row = collect.collect_run(incomplete, manifest)

    assert failed_row["status"] == "failed"
    assert failed_row["failure_reason"] == "OutOfMemoryError"
    assert incomplete_row["status"] == "incomplete"
    assert incomplete_row["failure_reason"] == "missing validation metric validation/energy"
    assert "validation/energy" not in failed_row


def test_completed_status_requires_validation_metric(tmp_path: Path, manifest_path: Path) -> None:
    manifest = collect.load_manifest(manifest_path)
    run_dir = make_run_dir(tmp_path / "runs", seed=3, status="completed", with_validation=False)
    assert collect.collect_run(run_dir, manifest)["status"] == "incomplete"


def test_csv_cell_formats_floats_for_readable_tables() -> None:
    assert collect._csv_cell(2.0) == "2.0"
    assert collect._csv_cell(2.123456789012345) == "2.12345678901"
    assert collect._csv_cell(1.0e-8) == "1e-08"
    assert collect._csv_cell(float("inf")) == "inf"


def test_collector_writes_runs_csv_and_jsonl(tmp_path: Path, manifest_path: Path) -> None:
    runs_root = tmp_path / "runs"
    make_run_dir(runs_root, seed=3, energy=2.0)
    make_run_dir(runs_root, seed=9, energy=2.5)
    make_run_dir(runs_root, seed=3, lr=3.0e-3, status="failed", with_validation=False)
    output_dir = tmp_path / "results"

    exit_code = collect.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-root",
            str(runs_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    with open(output_dir / "runs.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"completed", "failed"}
    completed = [row for row in rows if row["status"] == "completed"]
    assert {row["validation/energy"] for row in completed} == {"2.0", "2.5"}
    # Required columns from the study protocol are all present.
    for column in (
        "run_dir",
        "status",
        "failure_reason",
        "study_name",
        "config_id",
        "runtime.seed",
        "optimizer_params.lr",
        "model_params.channels",
        "model_params.layers",
        "model_params.gate_activation",
        "validation/energy",
        "validation/energy_stderr",
        "validation/energy_variance",
        "validation/local_energy_finite_fraction",
        "validation/sampler/acceptance_rate",
        "validation/sampler/n_walkers",
        "validation/sampler/burn_in",
        "validation/sampler/n_steps",
        "validation/sampler/proposal_scale",
        "validation/sampler/seed",
        "validation/sampler/radius_mean",
        "validation/sampler/radius_q99",
        "validation/sampler/radius_max",
        "validation/sampler/electron_distance_q01",
        "validation/sampler/electron_distance_min",
        "validation/sampler/position_rms",
        "runtime/wall_time_sec",
        "git/sha",
        "wandb/run_id",
    ):
        assert column in rows[0], f"missing column {column}"

    jsonl_rows = [
        json.loads(line)
        for line in (output_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(jsonl_rows) == 3


def test_study_scripts_do_not_import_spenn() -> None:
    """Experiments code stays decoupled from the spenn package."""

    import ast

    study_dir = Path(__file__).resolve().parents[1]
    for script in ("collect.py", "select.py"):
        tree = ast.parse((study_dir / script).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module != "spenn" and not module.startswith("spenn."), (
                    f"{script} imports {module}"
                )
