"""Tests for the Hooke pair validation post-processing study scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "experiments" / "hooke" / "studies" / "pair_validation"
MANIFEST = STUDY_DIR / "manifest.yaml"
FINAL_EVAL_CONFIG = ROOT / "experiments" / "hooke" / "configs" / "benchmark" / "pair_final_eval.yaml"


def _load_script(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pair_validation_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_script("collect")
select = _load_script("select")
evaluate_selected = _load_script("evaluate_selected")


def test_collector_normalizes_filters_and_preserves_raw_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    completed = _fake_run(run_root, "completed", study_name="hooke_pair_validation_v1", status="completed")
    failed = _fake_run(run_root, "failed", study_name="hooke_pair_validation_v1", status="failed")
    missing_metrics = _fake_run(
        run_root,
        "missing-metrics",
        study_name="hooke_pair_validation_v1",
        status="completed",
        write_metrics=False,
    )
    missing_validation = _fake_run(
        run_root,
        "missing-validation",
        study_name="hooke_pair_validation_v1",
        status="completed",
        include_validation=False,
    )
    other = _fake_run(run_root, "other", study_name="other_study", status="completed")
    before = (completed / "resolved_config.yaml").read_text()

    rows = collect.collect_runs(manifest_path=MANIFEST, run_root=run_root, output_dir=tmp_path / "reports")

    assert (tmp_path / "reports" / "runs.csv").exists()
    assert (tmp_path / "reports" / "runs.jsonl").exists()
    assert {Path(row["run_dir"]).name for row in rows} == {
        "completed",
        "failed",
        "missing-metrics",
        "missing-validation",
    }
    statuses = {Path(row["run_dir"]).name: row["status"] for row in rows}
    assert statuses == {
        "completed": "completed",
        "failed": "failed",
        "missing-metrics": "missing_metrics",
        "missing-validation": "missing_validation",
    }
    assert rows[0]["wandb/run_id"] is None
    assert (completed / "resolved_config.yaml").read_text() == before

    with (tmp_path / "reports" / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert set(collect.REQUIRED_COLUMNS) <= set(fieldnames or [])

    rows_with_other = collect.collect_runs(
        manifest_path=MANIFEST,
        run_root=run_root,
        output_dir=tmp_path / "reports-other",
        allow_other_studies=True,
    )
    assert any(Path(row["run_dir"]) == other for row in rows_with_other)


def test_selector_groups_failed_seeds_and_writes_outputs(tmp_path: Path) -> None:
    runs = [
        _run_row(seed=3, channels=8, energy=1.0, config_id="small"),
        _run_row(seed=9, channels=8, energy=1.1, config_id="small"),
        _run_row(seed=11, channels=8, status="failed", energy=None, config_id="small"),
        _run_row(seed=3, channels=32, energy=1.2, config_id="large"),
        _run_row(seed=9, channels=32, energy=1.2, config_id="large"),
        _run_row(seed=11, channels=32, energy=1.2, config_id="large"),
    ]
    runs_csv = _write_runs_csv(tmp_path, runs)

    selected = select.select_runs(manifest_path=MANIFEST, runs_path=runs_csv, output_dir=tmp_path / "selection")

    assert selected["selected"]["config_id"] == "large"
    with (tmp_path / "selection" / "selection.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = {row["config_id"]: row for row in csv.DictReader(handle)}
    assert rows["small"]["n_failed"] == "1"
    assert rows["small"]["median validation/energy"] == "1.1"
    assert (tmp_path / "selection" / "selection.csv").exists()
    assert (tmp_path / "selection" / "selected_config.yaml").exists()
    assert "selection margin" in (tmp_path / "selection" / "selection_report.md").read_text().lower()


def test_selector_refuses_all_failed_or_ineligible_study(tmp_path: Path) -> None:
    runs_csv = _write_runs_csv(
        tmp_path,
        [
            _run_row(seed=3, channels=8, status="failed", energy=None),
            _run_row(seed=9, channels=8, status="missing_validation", energy=None),
            _run_row(seed=11, channels=8, status="completed", energy=1.0, finite_fraction=0.5),
        ],
    )

    with pytest.raises(ValueError, match="no candidate has a finite eligible"):
        select.select_runs(manifest_path=MANIFEST, runs_path=runs_csv, output_dir=tmp_path / "selection")


def test_selector_rejects_forbidden_exact_reference_metric(tmp_path: Path) -> None:
    manifest = OmegaConf.load(MANIFEST)
    manifest.selection.metric = "validation/energy_abs_error"
    manifest_path = tmp_path / "manifest.yaml"
    OmegaConf.save(manifest, manifest_path, resolve=True)
    runs_csv = _write_runs_csv(tmp_path, [_run_row(seed=3, channels=8, energy=1.0)])

    with pytest.raises(ValueError, match="forbidden"):
        select.select_runs(manifest_path=manifest_path, runs_path=runs_csv, output_dir=tmp_path / "selection")


@pytest.mark.parametrize("raw", ["true", "false", "1", "0", "1.0", "0.0"])
def test_boolean_parser_accepts_required_forms(raw: str) -> None:
    assert select.parse_bool(raw) is (raw in {"true", "1", "1.0"})


def test_clear_lower_median_energy_wins_outside_margin(tmp_path: Path) -> None:
    winner = _select_two_candidates(
        tmp_path,
        [_run_row(seed=3, channels=8, energy=1.0), _run_row(seed=9, channels=8, energy=1.0), _run_row(seed=11, channels=8, energy=1.0)],
        [_run_row(seed=3, channels=32, energy=1.01), _run_row(seed=9, channels=32, energy=1.01), _run_row(seed=11, channels=32, energy=1.01)],
    )
    assert winner == "ch8"


def test_within_margin_uses_variance_tie_breaker(tmp_path: Path) -> None:
    winner = _select_two_candidates(
        tmp_path,
        [_run_row(seed=seed, channels=8, energy=1.0, variance=0.2) for seed in (3, 9, 11)],
        [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1) for seed in (3, 9, 11)],
    )
    assert winner == "ch32"


def test_if_variance_tied_lower_seed_iqr_wins(tmp_path: Path) -> None:
    winner = _select_two_candidates(
        tmp_path,
        [
            _run_row(seed=3, channels=8, energy=0.99995, variance=0.1),
            _run_row(seed=9, channels=8, energy=1.0, variance=0.1),
            _run_row(seed=11, channels=8, energy=1.00005, variance=0.1),
        ],
        [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1) for seed in (3, 9, 11)],
    )
    assert winner == "ch32"


def test_if_iqr_tied_lower_stderr_wins(tmp_path: Path) -> None:
    winner = _select_two_candidates(
        tmp_path,
        [_run_row(seed=seed, channels=8, energy=1.0, variance=0.1, stderr=0.02) for seed in (3, 9, 11)],
        [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)],
    )
    assert winner == "ch32"


def test_if_stderr_tied_fewer_geometry_warnings_wins(tmp_path: Path) -> None:
    noisy = [_run_row(seed=seed, channels=8, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)]
    for row in noisy:
        row["validation/sampler/electron_distance_q01"] = "1.0e-8"
    clean = [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)]
    winner = _select_two_candidates(tmp_path, noisy, clean)
    assert winner == "ch32"


def test_if_geometry_tied_smaller_model_wins(tmp_path: Path) -> None:
    winner = _select_two_candidates(
        tmp_path,
        [_run_row(seed=seed, channels=8, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)],
        [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)],
    )
    assert winner == "ch8"


def test_if_model_tied_lower_wall_time_wins(tmp_path: Path) -> None:
    fast = [
        _run_row(seed=seed, channels=8, gate="silu", energy=1.0, variance=0.1, stderr=0.01, wall_time=10.0)
        for seed in (3, 9, 11)
    ]
    slow = [
        _run_row(seed=seed, channels=8, gate="sigmoid", energy=1.0, variance=0.1, stderr=0.01, wall_time=20.0)
        for seed in (3, 9, 11)
    ]
    runs_csv = _write_runs_csv(tmp_path, [*fast, *slow])
    selected = select.select_runs(manifest_path=MANIFEST, runs_path=runs_csv, output_dir=tmp_path / "selection")
    assert selected["selected"]["hyperparameters"]["model_params.gate_activation"] == "silu"


def test_geometry_warning_logic_reports_unknown_or_suspicious_values() -> None:
    manifest = OmegaConf.to_container(OmegaConf.load(MANIFEST), resolve=True)
    row = {
        "validation/sampler/radius_q99": "",
        "validation/sampler/radius_max": "inf",
        "validation/sampler/electron_distance_q01": "",
        "validation/sampler/position_rms": "nan",
    }

    warnings = select.geometry_warnings(row, manifest)

    assert "validation/sampler/radius_q99 missing" in warnings
    assert "validation/sampler/radius_max nonfinite" in warnings
    assert "validation/sampler/electron_distance_q01 missing" in warnings
    assert "validation/sampler/position_rms nonfinite" in warnings


def test_evaluate_selected_dry_run_writes_load_configs_and_inputs(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)

    plan = evaluate_selected.generate_final_evaluation(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        run_root=tmp_path / "outputs",
        output_dir=tmp_path / "reports",
    )

    assert (tmp_path / "reports" / "final_eval_commands.sh").exists()
    assert (tmp_path / "reports" / "final_eval_manifest.yaml").exists()
    assert (tmp_path / "reports" / "final_eval_inputs.csv").exists()
    assert len(plan["inputs"]) == 10
    eval_config = OmegaConf.load(plan["inputs"][0]["eval_config"])
    eval_config_raw = OmegaConf.to_container(eval_config, resolve=False)
    assert eval_config.load.mode == "model_only"
    assert str(eval_config.load.path).endswith("checkpoints/latest.json")
    assert "latest.pt" not in str(eval_config.load.path)
    assert eval_config_raw["runner"]["model"] == "${model}"
    assert "load_model_checkpoint" not in Path(plan["inputs"][0]["eval_config"]).read_text()


def test_evaluate_selected_rejects_validation_seed_reuse(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)
    manifest = OmegaConf.load(MANIFEST)
    manifest.final_evaluation.training_seeds = [3]
    manifest.final_evaluation.eval_seeds = [100000]
    manifest_path = tmp_path / "manifest.yaml"
    OmegaConf.save(manifest, manifest_path, resolve=True)

    with pytest.raises(ValueError, match="reuse validation seeds"):
        evaluate_selected.generate_final_evaluation(
            manifest_path=manifest_path,
            selected_config_path=selected_config,
            run_root=tmp_path / "outputs",
            output_dir=tmp_path / "reports",
        )


def test_evaluate_selected_collect_writes_final_summary_files(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)
    plan = evaluate_selected.generate_final_evaluation(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        run_root=tmp_path / "outputs",
        output_dir=tmp_path / "reports",
    )
    first = plan["inputs"][0]
    _fake_eval_run(Path(first["eval_run_dir"]), training_seed=first["training_seed"], eval_seed=first["eval_seed"])

    summary = evaluate_selected.collect_final_outputs(inputs=plan["inputs"], output_dir=tmp_path / "reports")

    assert summary[0]["eval/energy"] == pytest.approx(2.01)
    assert (tmp_path / "reports" / "final_eval_runs.csv").exists()
    assert (tmp_path / "reports" / "final_benchmark_summary.csv").exists()
    assert (tmp_path / "reports" / "final_benchmark_summary.json").exists()
    assert (tmp_path / "reports" / "final_benchmark_report.md").exists()


def test_pair_final_eval_template_uses_pr81_load_contract() -> None:
    text = FINAL_EVAL_CONFIG.read_text()
    cfg = OmegaConf.load(FINAL_EVAL_CONFIG)
    raw = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.load.mode == "model_only"
    assert cfg.load.strict is True
    assert cfg.load.allow_protocol_mismatch is False
    assert raw["runner"]["model"] == "${model}"
    assert cfg.runner.diagnostics[0]._target_ == "spenn.diagnostics.EnergyEvaluation"
    assert "load_model_checkpoint" not in text


def test_readme_documents_reproducibility_contract() -> None:
    text = (STUDY_DIR / "README.md").read_text()
    for section in (
        "Overview",
        "Prerequisites",
        "Validation Scan Protocol",
        "How To Launch Training Scan",
        "How Validation Runs At Train End",
        "How To Collect Results",
        "How To Select The Winning Config",
        "Tie-Breaker Rule",
        "How To Generate Final Evaluation Commands",
        "How To Run Final Evaluation",
        "How To Summarize Final Evaluation Results",
        "Output Files",
        "W&B Role",
        "Reproducibility Notes",
    ):
        assert f"## {section}" in text
    assert "W&B is visualization only" in text
    assert "Validation does not use exact reference energy" in text
    assert "collect.py" in text
    assert "select.py" in text
    assert "evaluate_selected.py" in text


def _fake_run(
    root: Path,
    name: str,
    *,
    study_name: str,
    status: str,
    seed: int = 3,
    write_metrics: bool = True,
    include_validation: bool = True,
) -> Path:
    run_dir = root / "hooke_pair_benchmark" / "pair" / name
    run_dir.mkdir(parents=True)
    cfg = {
        "study": {"name": study_name, "config_id": "fake"},
        "runtime": {"seed": seed},
        "optimizer_params": {"lr": 0.001},
        "model_params": {"channels": 8, "layers": 1, "gate_activation": "silu"},
        "system": {"n_particles": 2, "spin": {"n_up": 1, "n_down": 1}},
    }
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "resolved_config.yaml", resolve=True)
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"status": status, "git_commit": "abc123"}), encoding="utf-8")
    (run_dir / "run_start.json").write_text(json.dumps({"git": {"sha": "abc123"}}), encoding="utf-8")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.json").write_text(json.dumps({"checkpoint_dir": "step_000001"}), encoding="utf-8")
    if status == "failed":
        (run_dir / "error.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    if write_metrics:
        records = []
        if include_validation:
            records.append(
                {
                    "step": 1,
                    "namespace": "validation",
                    "metrics": {
                        "energy": 1.0,
                        "energy_stderr": 0.01,
                        "energy_variance": 0.1,
                        "local_energy_finite_fraction": 1.0,
                    },
                }
            )
        records.extend(
            [
                {
                    "step": 1,
                    "namespace": "validation/sampler",
                    "metrics": {
                        "acceptance_rate": 0.6,
                        "n_walkers": 16,
                        "burn_in": 2,
                        "n_steps": 3,
                        "proposal_scale": 0.35,
                        "seed": 114514,
                        "n_electrons": 2,
                        "radius_mean": 1.0,
                        "radius_q99": 2.0,
                        "radius_max": 2.5,
                        "electron_distance_q01": 0.2,
                        "electron_distance_min": 0.1,
                        "position_rms": 1.2,
                    },
                },
                {"step": 1, "namespace": "checks/data_integrity", "metrics": {"passed": True}},
                {"step": 1, "namespace": "checks/gradient", "metrics": {"passed": True}},
                {"step": 1, "namespace": "checks/equivariance/full_model", "metrics": {"passed": True}},
                {"step": 1, "namespace": "runtime", "metrics": {"wall_time_sec": 10.0}},
            ]
        )
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")
    return run_dir


def _run_row(
    *,
    seed: int,
    channels: int,
    energy: float | None,
    status: str = "completed",
    config_id: str | None = None,
    gate: str = "silu",
    variance: float = 0.1,
    stderr: float = 0.0,
    finite_fraction: float = 1.0,
    wall_time: float = 10.0,
) -> dict[str, Any]:
    row = {
        "run_dir": f"/runs/ch{channels}/seed{seed}",
        "status": status,
        "study_name": "hooke_pair_validation_v1",
        "config_id": config_id or f"ch{channels}",
        "runtime.seed": seed,
        "optimizer_params.lr": 0.001,
        "model_params.channels": channels,
        "model_params.layers": 1,
        "model_params.gate_activation": gate,
        "system.n_particles": 2,
        "system.n_electrons": 2,
        "system.spin.n_up": 1,
        "system.spin.n_down": 1,
        "validation/energy": "" if energy is None else energy,
        "validation/energy_stderr": stderr,
        "validation/energy_variance": variance,
        "validation/local_energy_finite_fraction": finite_fraction,
        "validation/sampler/radius_q99": 2.0,
        "validation/sampler/radius_max": 2.5,
        "validation/sampler/electron_distance_q01": 0.2,
        "validation/sampler/position_rms": 1.2,
        "checks/data_integrity/passed": "1.0",
        "checks/gradient/passed": "true",
        "checks/equivariance/full_model/passed": "1",
        "runtime/wall_time_sec": wall_time,
        "git/sha": "abc123",
        "checkpoint/latest_path": f"/runs/ch{channels}/seed{seed}/checkpoints/latest.json",
    }
    return row


def _write_runs_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "runs.csv"
    columns = list(collect.REQUIRED_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _select_two_candidates(tmp_path: Path, first: list[dict[str, Any]], second: list[dict[str, Any]]) -> str:
    runs_csv = _write_runs_csv(tmp_path, first + second)
    selected = select.select_runs(manifest_path=MANIFEST, runs_path=runs_csv, output_dir=tmp_path / "selection")
    return str(selected["selected"]["config_id"])


def _write_selected_config(tmp_path: Path) -> Path:
    path = tmp_path / "selected_config.yaml"
    selected = {
        "study": {
            "name": "hooke_pair_validation_v1",
            "source_runs": str(tmp_path / "runs.csv"),
            "selection_report": str(tmp_path / "selection_report.md"),
        },
        "selection": {"selected_config_id": "ch8", "metric": "validation/energy"},
        "selected": {
            "config_id": "ch8",
            "hyperparameters": {
                "optimizer_params.lr": 0.001,
                "model_params.channels": 8,
                "model_params.layers": 1,
                "model_params.gate_activation": "silu",
            },
            "validation_runs": [
                {
                    "run_dir": "/runs/seed3",
                    "training_seed": 3,
                    "checkpoint_path": "/runs/seed3/checkpoints/latest.json",
                    "git_sha": "abc123",
                }
            ],
        },
    }
    OmegaConf.save(OmegaConf.create(selected), path, resolve=True)
    return path


def _fake_eval_run(run_dir: Path, *, training_seed: int, eval_seed: int) -> None:
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "study": {"name": "hooke_pair_validation_v1", "config_id": "ch8"},
                "evaluation": {"training_seed": training_seed},
                "runtime": {"seed": eval_seed},
            }
        ),
        run_dir / "resolved_config.yaml",
        resolve=True,
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"status": "completed", "git_commit": "abc123"}), encoding="utf-8")
    (run_dir / "run_start.json").write_text(json.dumps({"git": {"sha": "abc123"}}), encoding="utf-8")
    records = [
        {
            "step": 0,
            "namespace": "eval",
            "metrics": {
                "energy": 2.01,
                "energy_stderr": 0.02,
                "energy_variance": 0.3,
                "energy_error": 0.01,
                "energy_abs_error": 0.01,
            },
        },
        {
            "step": 0,
            "namespace": "eval/sampler",
            "metrics": {
                "acceptance_rate": 0.7,
                "radius_mean": 1.0,
                "radius_q99": 2.0,
                "electron_distance_q01": 0.2,
            },
        },
        {"step": 0, "namespace": "runtime", "metrics": {"wall_time_sec": 11.0}},
    ]
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
