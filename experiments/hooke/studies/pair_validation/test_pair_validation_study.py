"""Tests for the Hooke pair validation/final-benchmark study scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parent
MANIFEST = STUDY_DIR / "manifest.yaml"
STUDY_TRAIN_CONFIG = STUDY_DIR / "configs" / "pair_train.yaml"
STUDY_EVAL_CONFIG = STUDY_DIR / "configs" / "pair_eval.yaml"
METHODS = STUDY_DIR / "methods.md"
CONFIGS_README = ROOT / "experiments" / "hooke" / "configs" / "README.md"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))


def _load_script(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pair_validation_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study_manifest = _load_script("study_manifest")
collect = _load_script("collect")
select = _load_script("select")
plan_final = _load_script("plan_final")
collect_final = _load_script("collect_final")
orchestrate = _load_script("orchestrate")
sync_reports = _load_script("sync_reports")
plot_final = _load_script("plot_final")


def test_manifest_uses_phase_schema_and_one_train_eval_config() -> None:
    manifest = OmegaConf.load(MANIFEST)
    raw_manifest = OmegaConf.to_container(manifest, resolve=False)

    assert manifest.study.name == "hooke_pair_validation"
    assert manifest.study.version == "v2"
    assert manifest.configs.train == "experiments/hooke/studies/pair_validation/configs/pair_train.yaml"
    assert manifest.configs.eval == "experiments/hooke/studies/pair_validation/configs/pair_eval.yaml"
    assert manifest.paths.report_root == "experiments/hooke/studies/pair_validation/reports"
    assert manifest.phases.validation_train.run_root == (
        "experiments/hooke/studies/pair_validation/reports/01_train/outputs"
    )
    assert manifest.phases.validation_train.slurm_log_dir == (
        "experiments/hooke/studies/pair_validation/reports/01_train/slurm_logs"
    )
    assert manifest.phases.final_train.run_root == (
        "experiments/hooke/studies/pair_validation/reports/04_final_train/outputs"
    )
    assert manifest.phases.final_train.slurm_log_dir == (
        "experiments/hooke/studies/pair_validation/reports/04_final_train/slurm_logs"
    )
    assert manifest.phases.final_eval.run_root == (
        "experiments/hooke/studies/pair_validation/reports/05_final_eval/outputs"
    )
    assert manifest.phases.final_eval.slurm_log_dir == (
        "experiments/hooke/studies/pair_validation/reports/05_final_eval/slurm_logs"
    )
    assert (ROOT / manifest.configs.train).exists()
    assert (ROOT / manifest.configs.eval).exists()
    assert manifest.profiles.test.device == "cpu"
    assert manifest.profiles.test.slurm.partition == "test"
    assert manifest.profiles.test.slurm.timeout_min == 15
    assert manifest.profiles.gpu_test.device == "cuda"
    assert manifest.profiles.gpu_test.slurm.partition == "gpu_test"
    assert manifest.profiles.gpu_test.slurm.timeout_min == 15
    assert not (ROOT / "experiments" / "hooke" / "configs" / "benchmark" / "pair_final_eval.yaml").exists()
    assert not (STUDY_DIR / "launch_array.sh").exists()
    assert not (STUDY_DIR / "evaluate_selected.py").exists()

    assert manifest.phases.validation_train.orchestrator == "train"
    assert manifest.phases.validation_train.mode == "cartesian"
    validation_fixed = raw_manifest["phases"]["validation_train"]["overrides"]["fixed"]
    assert validation_fixed["study.name"] == "${study.name}"
    assert validation_fixed["study.version"] == "${study.version}"
    assert manifest.phases.validation_train.overrides.fixed["study.phase"] == "validation_train"
    assert manifest.phases.final_train.orchestrator == "train"
    assert manifest.phases.final_train.mode == "cartesian"
    assert manifest.phases.final_eval.orchestrator == "eval"
    assert manifest.phases.final_eval.mode == "rows"
    assert manifest.phases.validation_train.smoke.overlay
    assert manifest.phases.final_train.smoke.overlay
    assert manifest.phases.final_eval.smoke.overlay
    assert manifest.selection.source_phase == "validation_train"
    assert manifest.final_evaluation.source_phase == "final_train"


def test_pair_validation_tests_live_with_experiment_code() -> None:
    test_path = Path(__file__).resolve()

    assert test_path.is_relative_to(ROOT / "experiments")
    assert not test_path.is_relative_to(ROOT / "tests")


def test_study_scripts_do_not_hardcode_manifest_names_or_directories() -> None:
    scripts = (
        "collect.py",
        "collect_final.py",
        "orchestrate.py",
        "plan_final.py",
        "select.py",
        "study_manifest.py",
        "sync_reports.py",
    )
    forbidden = ("hooke_pair", "reports/hooke", "outputs/hooke", "slurm_logs/hooke")

    for script in scripts:
        text = (STUDY_DIR / script).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{script} hardcodes {pattern!r}"


def test_sync_reports_keeps_eval_latest_checkpoints_only(tmp_path: Path) -> None:
    source = tmp_path / "reports"
    destination = tmp_path / "snapshot"
    train_run = source / "01_train" / "outputs" / "full" / "ch8" / "seed=3"
    eval_run = (
        source
        / "05_final_eval"
        / "outputs"
        / "full"
        / "ch8"
        / "train_seed=100_eval_seed=100000"
    )
    _write_checkpointed_report_run(train_run)
    _write_checkpointed_report_run(eval_run)
    slurm_dir = source / "01_train" / "slurm_logs"
    slurm_dir.mkdir(parents=True)
    (slurm_dir / "22707654_0.out").write_text("log", encoding="utf-8")
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    summary = sync_reports.sync_reports(
        source=source,
        destination=destination,
        checkpoint_roots=(source / "05_final_eval" / "outputs",),
    )

    assert summary.copied_files == 7
    assert summary.skipped_checkpoint_files == 7
    assert summary.skipped_slurm_log_files == 1
    assert "copied_mb" in summary.to_dict()
    assert "copied_bytes" not in summary.to_dict()
    assert not (destination / "stale.txt").exists()
    assert (destination / "01_train" / "outputs" / "full" / "ch8" / "seed=3" / "status.json").exists()
    assert not (
        destination
        / "01_train"
        / "outputs"
        / "full"
        / "ch8"
        / "seed=3"
        / "checkpoints"
        / "latest.json"
    ).exists()
    assert not (
        destination
        / "01_train"
        / "outputs"
        / "full"
        / "ch8"
        / "seed=3"
        / "checkpoints"
        / "step_000002"
        / "model.pt"
    ).exists()
    assert (
        destination
        / "05_final_eval"
        / "outputs"
        / "full"
        / "ch8"
        / "train_seed=100_eval_seed=100000"
        / "checkpoints"
        / "latest.json"
    ).exists()
    assert (
        destination
        / "05_final_eval"
        / "outputs"
        / "full"
        / "ch8"
        / "train_seed=100_eval_seed=100000"
        / "checkpoints"
        / "step_000002"
        / "model.pt"
    ).exists()
    assert not (
        destination
        / "05_final_eval"
        / "outputs"
        / "full"
        / "ch8"
        / "train_seed=100_eval_seed=100000"
        / "checkpoints"
        / "step_000001"
        / "model.pt"
    ).exists()
    assert not (destination / "01_train" / "slurm_logs" / "22707654_0.out").exists()

    dry_destination = tmp_path / "dry-snapshot"
    dry_summary = sync_reports.sync_reports(
        source=source,
        destination=dry_destination,
        checkpoint_roots=(source / "05_final_eval" / "outputs",),
        dry_run=True,
    )
    assert dry_summary.copied_files == summary.copied_files
    assert not dry_destination.exists()


def test_sync_reports_derives_eval_checkpoint_roots_from_manifest(tmp_path: Path) -> None:
    manifest = OmegaConf.load(MANIFEST)
    manifest.paths.report_root = str(tmp_path / "manifest-reports")
    manifest.phases.validation_train.run_root = "${paths.report_root}/01_train/outputs"
    manifest.phases.final_train.run_root = "${paths.report_root}/04_final_train/outputs"
    manifest.phases.final_eval.run_root = "${paths.report_root}/05_final_eval/outputs"
    manifest_path = tmp_path / "manifest.yaml"
    OmegaConf.save(manifest, manifest_path, resolve=True)
    source = tmp_path / "actual-reports"

    roots = sync_reports.eval_checkpoint_roots(
        study_manifest.load_yaml(manifest_path),
        manifest_path,
        source,
    )

    assert roots == ((source / "05_final_eval" / "outputs").resolve(),)


def test_collector_normalizes_filters_and_preserves_raw_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    completed = _fake_run(run_root, "completed", study_name="hooke_pair_validation", status="completed")
    failed = _fake_run(run_root, "failed", study_name="hooke_pair_validation", status="failed")
    missing_metrics = _fake_run(
        run_root,
        "missing-metrics",
        study_name="hooke_pair_validation",
        status="completed",
        write_metrics=False,
    )
    missing_validation = _fake_run(
        run_root,
        "missing-validation",
        study_name="hooke_pair_validation",
        status="completed",
        include_validation=False,
    )
    smoke_completed = _fake_run(
        run_root,
        "smoke/completed",
        study_name="hooke_pair_validation",
        status="completed",
    )
    other = _fake_run(run_root, "other", study_name="other_study", status="completed")
    before = (completed / "resolved_config.yaml").read_text()

    rows = collect.collect_runs(manifest_path=MANIFEST, run_root=run_root, output_dir=tmp_path / "reports")

    assert (tmp_path / "reports" / "runs.csv").exists()
    assert (tmp_path / "reports" / "runs.jsonl").exists()
    assert len(rows) == 4
    assert all(Path(row["run_dir"]) != smoke_completed for row in rows)
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
    failed_row = next(row for row in rows if Path(row["run_dir"]).name == "failed")
    assert failed_row["status/current_event"] == "exception"
    assert failed_row["status/exception_type"] == "RuntimeError"
    assert failed_row["status/exception_message"] == "synthetic failure"
    assert (completed / "resolved_config.yaml").read_text() == before

    with (tmp_path / "reports" / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        csv_rows = {Path(row["run_dir"]).name: row for row in reader}
    assert set(collect.REQUIRED_COLUMNS) <= set(fieldnames or [])
    assert csv_rows["failed"]["status/exception_message"] == "synthetic failure"

    rows_with_other = collect.collect_runs(
        manifest_path=MANIFEST,
        run_root=run_root,
        output_dir=tmp_path / "reports-other",
        allow_other_studies=True,
        include_smoke=True,
    )
    assert any(Path(row["run_dir"]) == other for row in rows_with_other)
    assert any(Path(row["run_dir"]) == smoke_completed for row in rows_with_other)


@pytest.mark.integration
def test_local_smoke_pipeline_runs_collects_selects_and_plans(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from spenn.run import run_from_config

    run_root = tmp_path / "outputs"
    reports = tmp_path / "reports"
    manifest_path = _write_local_smoke_manifest(tmp_path)

    cfg = OmegaConf.load(STUDY_TRAIN_CONFIG)
    cfg.run.root = str(run_root)
    cfg.run.timezone = "UTC"
    cfg.terminal.enabled = False
    cfg.runtime.seed = 3
    cfg.runtime.device = "cpu"
    cfg.study = {"name": "hooke_pair_validation", "version": "v2", "phase": "validation_train", "config_id": "local_smoke"}
    cfg.optimizer_params.lr = 0.01
    cfg.model_params.channels = 4
    cfg.model_params.layers = 1
    cfg.model_params.gate_activation = "silu"
    cfg.training.max_steps = 1
    cfg.sampler_params.n_walkers = 4
    cfg.sampler_params.burn_in = 1
    cfg.sampler_params.n_steps = 1
    cfg.validation_sampler_params.n_walkers = 4
    cfg.validation_sampler_params.burn_in = 1
    cfg.validation_sampler_params.n_steps = 1
    cfg.checkpoint.keep_last = 1

    assert run_from_config(cfg, config_path=str(STUDY_TRAIN_CONFIG), command="pytest local pipeline smoke") == 0

    rows = collect.collect_runs(manifest_path=manifest_path, run_root=run_root)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["checkpoint/latest_path"]

    selected = select.select_runs(manifest_path=manifest_path, runs_path=reports / "02_collect" / "runs.csv")
    assert selected["selected"]["config_id"] == "local_smoke"

    plan = plan_final.plan_final(
        manifest_path=manifest_path,
        selected_config_path=reports / "03_select" / "selected_config.yaml",
        phase="final_train",
    )
    assert len(plan["jobs"]) == 1
    assert "runtime.seed=100" in plan["jobs"][0]["overrides"]
    assert "model_params.channels=4" in plan["jobs"][0]["overrides"]


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
    assert selected["selected"]["model_params"]["channels"] == 32
    with (tmp_path / "selection" / "selection.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = {row["config_id"]: row for row in csv.DictReader(handle)}
    assert rows["small"]["n_failed"] == "1"
    assert rows["small"]["median validation/energy"] == "1.1"
    assert (tmp_path / "selection" / "selection.jsonl").exists()
    assert (tmp_path / "selection" / "selected_config.yaml").exists()
    report = (tmp_path / "selection" / "selection_report.md").read_text()
    assert "Selection report:" in report
    assert "selection margin" in report.lower()
    assert "selection.jsonl" in report
    assert "| rank | config_id | lr | channels | layers | gate activation | median energy | stderr | IQR | failures | geometry warnings |" in report
    assert "| energy rank | selected |" not in report
    assert "| tie-break rank | selected |" not in report
    assert report.index("`large`") < report.index("`small`")


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


def test_tie_breakers_and_geometry_warnings(tmp_path: Path) -> None:
    noisy = [_run_row(seed=seed, channels=8, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)]
    for row in noisy:
        row["validation/sampler/electron_distance_q01"] = "1.0e-8"
    clean = [_run_row(seed=seed, channels=32, energy=1.0, variance=0.1, stderr=0.01) for seed in (3, 9, 11)]
    runs_csv = _write_runs_csv(tmp_path, noisy + clean)

    selected = select.select_runs(manifest_path=MANIFEST, runs_path=runs_csv, output_dir=tmp_path / "selection")

    assert selected["selected"]["config_id"] == "ch32"
    manifest = OmegaConf.to_container(OmegaConf.load(MANIFEST), resolve=True)
    warnings = select.geometry_warnings(
        {
            "validation/sampler/radius_q99": "",
            "validation/sampler/radius_max": "inf",
            "validation/sampler/electron_distance_q01": "",
            "validation/sampler/position_rms": "nan",
        },
        manifest,
    )
    assert "validation/sampler/radius_q99 missing" in warnings
    assert "validation/sampler/radius_max nonfinite" in warnings
    assert "validation/sampler/electron_distance_q01 missing" in warnings
    assert "validation/sampler/position_rms nonfinite" in warnings


def test_plan_final_train_and_final_eval_jobs(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)

    final_train = plan_final.plan_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        phase="final_train",
        output_dir=tmp_path / "reports",
    )

    assert (tmp_path / "reports" / "final_train_manifest.yaml").exists()
    assert (tmp_path / "reports" / "final_train_jobs.jsonl").exists()
    assert len(final_train["jobs"]) == 10
    assert "optimizer_params.lr=0.001" in final_train["jobs"][0]["overrides"]
    assert "model_params.channels=8" in final_train["jobs"][0]["overrides"]
    assert final_train["jobs"][0]["run_id"] == "full/ch8/seed=100"
    assert "run.layout=flat" in final_train["jobs"][0]["overrides"]

    checkpoint_csv = _write_final_train_runs_csv(tmp_path)
    final_eval = plan_final.plan_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        final_train_runs_path=checkpoint_csv,
        phase="final_eval",
        output_dir=tmp_path / "reports",
    )
    smoke_eval = plan_final.plan_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        final_train_runs_path=checkpoint_csv,
        phase="smoke_eval",
        output_dir=tmp_path / "reports",
    )

    assert (tmp_path / "reports" / "final_eval_manifest.yaml").exists()
    assert (tmp_path / "reports" / "final_eval_jobs.jsonl").exists()
    assert len(final_eval["jobs"]) == 10
    assert final_eval["jobs"][0]["run_id"] == "full/ch8/train_seed=100_eval_seed=100000"
    assert smoke_eval["jobs"][0]["run_id"] == "smoke/ch8/train_seed=100_eval_seed=100000"
    assert "run.layout=flat" in final_eval["jobs"][0]["overrides"]
    assert "run.layout=flat" in smoke_eval["jobs"][0]["overrides"]
    assert [(job["train_seed"], job["eval_seed"]) for job in final_eval["jobs"]] == [
        (100 + index, 100000 + index) for index in range(10)
    ]
    assert "load.mode=model_only" in final_eval["jobs"][0]["overrides"]
    assert any(item.startswith("load.path=") for item in final_eval["jobs"][0]["overrides"])
    assert "runtime.seed=100000" in final_eval["jobs"][0]["overrides"]
    assert "evaluation.training_seed=100" in final_eval["jobs"][0]["overrides"]
    assert "runtime.seed=100001" in final_eval["jobs"][1]["overrides"]
    assert "evaluation.training_seed=101" in final_eval["jobs"][1]["overrides"]
    assert not any(
        job["train_seed"] == 100 and job["eval_seed"] == 100001 for job in final_eval["jobs"]
    )
    assert len(smoke_eval["jobs"]) == 1
    assert smoke_eval["jobs"][0]["train_seed"] == 100
    assert smoke_eval["jobs"][0]["eval_seed"] == 100000
    assert "sampler_params.n_walkers=128" in smoke_eval["jobs"][0]["overrides"]


def test_collect_final_writes_tables_summary_and_report(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)
    train_root = tmp_path / "final_train"
    eval_root = tmp_path / "final_eval"
    checkpoint = _fake_final_train_run(train_root, seed=100)
    eval_run = _fake_eval_run(eval_root, checkpoint_path=checkpoint, training_seed=100, eval_seed=100000)
    smoke_checkpoint = _fake_final_train_run(train_root / "smoke", seed=101)
    _fake_eval_run(eval_root / "smoke", checkpoint_path=smoke_checkpoint, training_seed=101, eval_seed=100001)
    (eval_run / "error.json").write_text(
        json.dumps({"status": "failed", "message": "stale retry artifact"}),
        encoding="utf-8",
    )

    result = collect_final.collect_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        final_train_root=train_root,
        final_eval_root=eval_root,
        output_dir=tmp_path / "reports",
    )

    assert result["summary"]["final_eval_completed"] == 1
    assert result["summary"]["final_eval_failed_or_incomplete"] == 0
    assert result["summary"]["energy_mean"] == pytest.approx(2.01)
    row = result["final_eval_runs"][0]
    assert row["eval/virial_residual"] == pytest.approx(2.0 * 0.7 - 2.0 * 0.8 + 0.5)
    assert row["artifact/pair_distance_probe_exists"] is True
    assert (tmp_path / "reports" / "final_train_runs.csv").exists()
    assert (tmp_path / "reports" / "final_eval_runs.jsonl").exists()
    assert (tmp_path / "reports" / "final_benchmark_summary.csv").exists()
    assert (tmp_path / "reports" / "final_benchmark_summary.json").exists()
    report = (tmp_path / "reports" / "final_benchmark_report.md").read_text()
    assert "Final benchmark report:" in report
    assert "load.mode" in report

    with_smoke = collect_final.collect_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        final_train_root=train_root,
        final_eval_root=eval_root,
        output_dir=tmp_path / "reports-with-smoke",
        include_smoke=True,
    )
    assert with_smoke["summary"]["final_eval_completed"] == 2


def test_collect_final_strict_artifacts_fails_on_missing_required_artifact(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)
    train_root = tmp_path / "final_train"
    eval_root = tmp_path / "final_eval"
    checkpoint = _fake_final_train_run(train_root, seed=100)
    eval_run = _fake_eval_run(eval_root, checkpoint_path=checkpoint, training_seed=100, eval_seed=100000)
    (eval_run / "diagnostics" / "pair_distance_probe" / "probe.csv").unlink()

    with pytest.raises(FileNotFoundError, match="pair_distance_probe"):
        collect_final.collect_final(
            manifest_path=MANIFEST,
            selected_config_path=selected_config,
            final_train_root=train_root,
            final_eval_root=eval_root,
            output_dir=tmp_path / "reports",
            strict_artifacts=True,
        )


def test_plot_final_writes_tables_plots_and_report_from_collected_files(tmp_path: Path) -> None:
    selected_config = _write_selected_config(tmp_path)
    train_root = tmp_path / "final_train"
    eval_root = tmp_path / "final_eval"
    checkpoint = _fake_final_train_run(train_root, seed=100)
    _fake_eval_run(eval_root, checkpoint_path=checkpoint, training_seed=100, eval_seed=100000)
    collect_final.collect_final(
        manifest_path=MANIFEST,
        selected_config_path=selected_config,
        final_train_root=train_root,
        final_eval_root=eval_root,
        output_dir=tmp_path / "reports",
    )

    result = plot_final.plot_final(manifest_path=MANIFEST, final_eval_dir=tmp_path / "reports")

    assert (tmp_path / "reports" / "tables" / "energy_reference.csv").exists()
    assert (tmp_path / "reports" / "tables" / "artifact_summary.csv").exists()
    assert (tmp_path / "reports" / "plots" / "energy_by_run.png").exists()
    assert (tmp_path / "reports" / "plots" / "probe_pair_distance_logabs.png").exists()
    report = (tmp_path / "reports" / "final_benchmark_report.md").read_text()
    assert "Hooke pair final benchmark report" in report
    assert "Position-Exchange Check" in report
    assert "| train_seed | eval_seed | energy | stderr | reference | error | abs_error | kinetic | harmonic_trap | electron_electron | virial_residual | virial_rel |" in report
    assert "![Energy by run](plots/energy_by_run.png)" in report
    assert "![Energy by run](plots/energy_by_run.png)\n\n![Energy error by run]" in report
    assert "- plots/energy_by_run.png" not in report
    assert "![Pair-distance probe logabs](plots/probe_pair_distance_logabs.png)" in report
    assert "| quantity | mean | median | min | max |" in report
    assert "| run | contract | max_abs_error | mean_abs_error | failure_count | nonfinite_count |" in report
    assert "\n\n![Pair-distance probe local energy](plots/probe_pair_distance_local_energy.png)" in report
    assert "![Pair-distance probe logabs](plots/probe_pair_distance_logabs.png)\n\n![Pair-distance probe relative abs psi]" in report
    assert "- tables/energy_components_and_virial.csv" not in report
    assert "- tables/exchange_summary.csv" not in report
    assert result["warnings"] == []


def test_report_markdown_table_formats_numeric_columns() -> None:
    table = plot_final._markdown_table(
        [
            {"label": "large", "regular": 2.011085943, "tiny": 8.881784197e-16, "mixed": 2.011085943},
            {"label": "huge", "regular": 121.3793763, "tiny": 1.776356839e-15, "mixed": 0.0034819705},
            {"label": "small", "regular": 0.097190397, "tiny": 0.0, "mixed": 0.0},
        ]
    )

    assert "| large | 2.011 | 8.88e-16 | 2011.08e-03 |" in table
    assert "| huge | 121.3 | 17.76e-16 | 3.48e-03 |" in table
    assert "| small | 0.09719 | 0.00e-16 | 0.00e-03 |" in table


def test_orchestrator_jobs_and_dry_run_contracts(tmp_path: Path) -> None:
    manifest = study_manifest.load_yaml(MANIFEST)
    validation_jobs = orchestrate.phase_jobs(manifest=manifest, kind="train", phase="validation_train")
    smoke_jobs = orchestrate.phase_jobs(manifest=manifest, kind="train", phase="smoke_train")
    selected_config = _write_selected_config(tmp_path)
    final_jobs = orchestrate.phase_jobs(
        manifest=manifest,
        kind="train",
        phase="final_train",
        selected_config_path=selected_config,
    )

    assert len(validation_jobs) == 144
    assert len(smoke_jobs) == 1
    assert smoke_jobs[0]["base_config"] == validation_jobs[0]["base_config"]
    assert validation_jobs[0]["run_id"].startswith("full/")
    assert smoke_jobs[0]["run_id"].startswith("smoke/")
    assert "/seed=" in validation_jobs[0]["run_id"]
    assert "run.layout=flat" in validation_jobs[0]["overrides"]
    assert "run.layout=flat" in smoke_jobs[0]["overrides"]
    assert "training.max_steps=2" in smoke_jobs[0]["overrides"]
    assert len(final_jobs) == 10

    command = study_manifest.command_for_job(validation_jobs[0], device="cpu")
    assert command[:4] == ["python", "-u", "run.py", "--config"]
    assert "experiments/hooke/studies/pair_validation/configs/pair_train.yaml" in command
    assert "runtime.device=cpu" in command

    options = study_manifest.slurm_options(manifest, phase="validation_train", profile="gpu", job_count=144)
    assert options["job_name"] == "hooke-pv-validation-train"
    assert options["log_dir"] == "experiments/hooke/studies/pair_validation/reports/01_train/slurm_logs"
    assert options["partition"] == "kozinsky_gpu,seas_gpu"
    assert options["gres"] == "gpu:1"
    assert options["array_parallelism"] == 144

    smoke_cpu = study_manifest.slurm_options(manifest, phase="validation_train", profile="test", job_count=1)
    assert smoke_cpu["partition"] == "test"
    assert "gres" not in smoke_cpu
    assert smoke_cpu["timeout_min"] == 15

    smoke_gpu = study_manifest.slurm_options(manifest, phase="validation_train", profile="gpu_test", job_count=1)
    assert smoke_gpu["partition"] == "gpu_test"
    assert smoke_gpu["gres"] == "gpu:1"
    assert smoke_gpu["timeout_min"] == 15


def test_orchestrate_py_help_and_dry_run_contract(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(STUDY_DIR / "orchestrate.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "usage:" in help_result.stdout
    assert "--kind" in help_result.stdout
    assert "Run train or eval jobs." in help_result.stdout
    assert "Execution backend." in help_result.stdout
    assert "examples:" in help_result.stdout

    result = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "orchestrate.py"),
            "--kind",
            "train",
            "--backend",
            "local",
            "--phase",
            "smoke_train",
            "--profile",
            "cpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "run.py --config" in result.stdout
    assert "experiments/hooke/studies/pair_validation/configs/pair_train.yaml" in result.stdout
    assert "training.max_steps=2" in result.stdout
    assert not (STUDY_DIR / "orchestrate.sh").exists()
    assert not (STUDY_DIR / "orchestrate_train_local.sh").exists()


def test_smoke_slurm_uses_test_partitions(tmp_path: Path) -> None:
    cpu_result = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "orchestrate.py"),
            "--kind",
            "train",
            "--backend",
            "slurm",
            "--phase",
            "smoke_train",
            "--profile",
            "cpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cpu_result.returncode == 0, cpu_result.stderr
    assert "Slurm launch plan" in cpu_result.stdout
    assert "slurm profile: test" in cpu_result.stdout
    assert "partition: test" in cpu_result.stdout
    assert "--partition=test" in cpu_result.stdout
    assert "gres:" not in cpu_result.stdout
    assert "--gres" not in cpu_result.stdout
    assert "--profile cpu" in cpu_result.stdout

    gpu_result = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "orchestrate.py"),
            "--kind",
            "train",
            "--backend",
            "slurm",
            "--phase",
            "smoke_train",
            "--profile",
            "gpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert gpu_result.returncode == 0, gpu_result.stderr
    assert "Slurm launch plan" in gpu_result.stdout
    assert "slurm profile: gpu_test" in gpu_result.stdout
    assert "partition: gpu_test" in gpu_result.stdout
    assert "gres: gpu:1" in gpu_result.stdout
    assert "--partition=gpu_test" in gpu_result.stdout
    assert "--gres=gpu:1" in gpu_result.stdout
    assert "--profile gpu" in gpu_result.stdout


def test_orchestrate_missing_eval_jobs_prints_planning_hint(tmp_path: Path) -> None:
    missing_jobs = tmp_path / "reports" / "smoke_eval_jobs.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "orchestrate.py"),
            "--kind",
            "eval",
            "--backend",
            "slurm",
            "--manifest",
            str(MANIFEST),
            "--phase",
            "smoke_eval",
            "--jobs",
            str(missing_jobs),
            "--profile",
            "gpu",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert f"jobs file not found: {missing_jobs}" in result.stderr
    assert "plan_final.py" in result.stderr
    assert "--phase smoke_eval" in result.stderr
    assert "final_train_runs.csv" in result.stderr


def test_pair_eval_template_uses_pr81_load_contract() -> None:
    text = STUDY_EVAL_CONFIG.read_text()
    cfg = OmegaConf.load(STUDY_EVAL_CONFIG)
    raw = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.load.mode == "model_only"
    assert cfg.load.strict is True
    assert cfg.load.allow_protocol_mismatch is False
    assert raw["runner"]["model"] == "${model}"
    assert cfg.runner.diagnostics[0]._target_ == "spenn.diagnostics.EnergyEvaluation"
    assert "load_model_checkpoint" not in text


def test_study_configs_use_wandb_study_provenance() -> None:
    train = OmegaConf.to_container(OmegaConf.load(STUDY_TRAIN_CONFIG), resolve=False)
    eval_cfg = OmegaConf.to_container(OmegaConf.load(STUDY_EVAL_CONFIG), resolve=False)

    for cfg, phase in ((train, "validation_train"), (eval_cfg, "final_eval")):
        assert cfg["study"]["version"] is None
        assert cfg["study"]["phase"] == phase
        assert cfg["wandb"]["group"] == "${study.name}_${study.version}_${study.phase}"
        assert "${study.name}" in cfg["wandb"]["tags"]
        assert "${study.version}" in cfg["wandb"]["tags"]
        assert "${study.phase}" in cfg["wandb"]["tags"]


def test_readme_documents_phase_flow() -> None:
    text = (STUDY_DIR / "README.md").read_text()
    for section in (
        "Phase Flow",
        "Smoke Train",
        "Validation Scan",
        "Collect",
        "Select",
        "Final Planning",
        "Final Benchmark",
        "Sync Reports",
        "Outputs To Keep",
    ):
        assert f"## {section}" in text
    assert "01_train/outputs/smoke/<config_id>/seed=<seed>" in text
    assert "Only `orchestrate.py` launches SpENN" in text
    assert "orchestrate.py --help" in text
    assert "orchestrate_train_local.sh" not in text
    assert "orchestrate_eval_slurm.sh" not in text
    assert "plan_final.py" in text
    assert "collect_final.py" in text
    assert "plot_final.py" in text
    assert "experiments/hooke/studies/pair_validation/configs/pair_train.yaml" in text
    assert "experiments/hooke/studies/pair_validation/configs/pair_eval.yaml" in text
    assert "evaluate_selected.py" not in text
    assert "launch_array.sh" not in text


def test_methods_documents_experiment_protocol() -> None:
    text = METHODS.read_text()

    assert "Validation is used only for selection" in text
    assert "Only `orchestrate.py` launches SpENN" in text
    assert "orchestrate.py --help" in text
    assert "runtime.seed: [3, 9, 11]" in text
    assert "phase-local overrides.sweep" in text
    assert "experiments/hooke/studies/pair_validation/configs/" in text
    assert "legacy test/reference configs" in text
    assert "plan_final.py" in text
    assert "collect_final.py" in text
    assert "mode: model_only" in text
    assert "Reproducibility" in text


def test_hooke_configs_readme_marks_legacy_test_configs() -> None:
    text = CONFIGS_README.read_text()

    assert "# Legacy Hooke Test Configs" in text
    assert "legacy cheap VMC training test config" in text
    assert "legacy benchmark-shaped test/reference config" in text
    assert "legacy CPU preflight test config" in text


def _write_local_smoke_manifest(tmp_path: Path) -> Path:
    manifest = OmegaConf.load(MANIFEST)
    manifest.phases.validation_train.overrides.sweep["runtime.seed"] = [3]
    manifest.phases.validation_train.overrides.sweep["optimizer_params.lr"] = [0.01]
    manifest.phases.validation_train.overrides.sweep["model_params.channels"] = [4]
    manifest.phases.validation_train.overrides.sweep["model_params.layers"] = [1]
    manifest.phases.validation_train.overrides.sweep["model_params.gate_activation"] = ["silu"]
    manifest.phases.final_train.overrides.sweep["runtime.seed"] = [100]
    manifest.final_evaluation.eval_seeds = [100000]
    manifest.final_evaluation.sampler.n_walkers = 4
    manifest.final_evaluation.sampler.burn_in = 1
    manifest.final_evaluation.sampler.n_steps = 1
    manifest.paths.report_root = str(tmp_path / "reports")
    manifest.phases.validation_train.run_root = str(tmp_path / "reports" / "01_train" / "outputs")
    manifest.phases.validation_train.slurm_log_dir = str(tmp_path / "reports" / "01_train" / "slurm_logs")
    manifest.phases.final_train.run_root = str(tmp_path / "reports" / "04_final_train" / "outputs")
    manifest.phases.final_train.slurm_log_dir = str(tmp_path / "reports" / "04_final_train" / "slurm_logs")
    manifest.phases.final_eval.run_root = str(tmp_path / "reports" / "05_final_eval" / "outputs")
    manifest.phases.final_eval.slurm_log_dir = str(tmp_path / "reports" / "05_final_eval" / "slurm_logs")
    path = tmp_path / "manifest.yaml"
    OmegaConf.save(manifest, path, resolve=True)
    return path


def _write_checkpointed_report_run(run_dir: Path) -> None:
    checkpoint_root = run_dir / "checkpoints"
    old_checkpoint = checkpoint_root / "step_000001"
    latest_checkpoint = checkpoint_root / "step_000002"
    old_checkpoint.mkdir(parents=True)
    latest_checkpoint.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint_root / "latest.json").write_text(
        json.dumps({"checkpoint_dir": latest_checkpoint.name}),
        encoding="utf-8",
    )
    (old_checkpoint / "model.pt").write_text("old", encoding="utf-8")
    (old_checkpoint / "manifest.json").write_text("old manifest", encoding="utf-8")
    (latest_checkpoint / "model.pt").write_text("latest", encoding="utf-8")
    (latest_checkpoint / "manifest.json").write_text("latest manifest", encoding="utf-8")


def _fake_run(
    root: Path,
    name: str,
    *,
    study_name: str,
    status: str,
    study_version: str = "v2",
    study_phase: str = "validation_train",
    seed: int = 3,
    write_metrics: bool = True,
    include_validation: bool = True,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    cfg = {
        "study": {"name": study_name, "version": study_version, "phase": study_phase, "config_id": "fake"},
        "runtime": {"seed": seed},
        "optimizer_params": {"lr": 0.001},
        "model_params": {"channels": 8, "layers": 1, "gate_activation": "silu"},
        "system": {"n_particles": 2, "spin": {"n_up": 1, "n_down": 1}},
    }
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "resolved_config.yaml", resolve=True)
    status_record = {"status": status}
    metadata_record = {"status": status, "git_commit": "abc123"}
    if status == "failed":
        status_record.update(
            {
                "current_event": "exception",
                "exception_type": "RuntimeError",
                "exception_message": "synthetic failure",
            }
        )
        metadata_record.update(
            {
                "exception_type": "RuntimeError",
                "exception_message": "synthetic failure",
            }
        )
    (run_dir / "status.json").write_text(json.dumps(status_record), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata_record), encoding="utf-8")
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
    return {
        "run_dir": f"/runs/ch{channels}/seed{seed}",
        "status": status,
        "study_name": "hooke_pair_validation",
        "study_version": "v2",
        "study_phase": "validation_train",
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


def _write_selected_config(tmp_path: Path) -> Path:
    path = tmp_path / "selected_config.yaml"
    selected = {
        "study": {
            "name": "hooke_pair_validation",
            "version": "v2",
            "source_phase": "validation_train",
            "source_runs": str(tmp_path / "runs.csv"),
            "selection_report": str(tmp_path / "selection_report.md"),
        },
        "selection": {"selected_config_id": "ch8", "metric": "validation/energy"},
        "selected": {
            "config_id": "ch8",
            "optimizer_params": {"lr": 0.001},
            "model_params": {"channels": 8, "layers": 1, "gate_activation": "silu"},
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


def _write_final_train_runs_csv(tmp_path: Path) -> Path:
    path = tmp_path / "final_train_runs.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collect.REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for seed in range(100, 110):
            checkpoint = tmp_path / "outputs" / "hooke_pair_validation" / f"seed={seed}" / "checkpoints" / "latest.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"checkpoint_dir": "step_000001"}), encoding="utf-8")
            row = _run_row(seed=seed, channels=8, energy=1.0, config_id="ch8")
            row["study_name"] = "hooke_pair_validation"
            row["study_version"] = "v2"
            row["study_phase"] = "final_train"
            row["checkpoint/latest_path"] = str(checkpoint)
            writer.writerow({key: row.get(key, "") for key in collect.REQUIRED_COLUMNS})
    return path


def _fake_final_train_run(root: Path, *, seed: int) -> Path:
    run_dir = _fake_run(
        root,
        f"seed={seed}",
        study_name="hooke_pair_validation",
        study_phase="final_train",
        status="completed",
        seed=seed,
    )
    return run_dir / "checkpoints" / "latest.json"


def _fake_eval_run(root: Path, *, checkpoint_path: Path, training_seed: int, eval_seed: int) -> Path:
    run_dir = root / f"train_seed={training_seed}_eval_seed={eval_seed}"
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "study": {"name": "hooke_pair_validation", "version": "v2", "phase": "final_eval", "config_id": "ch8"},
                "evaluation": {"training_seed": training_seed},
                "runtime": {"seed": eval_seed},
                "load": {"path": str(checkpoint_path)},
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
                "reference_energy": 2.0,
                "energy_error": 0.01,
                "energy_abs_error": 0.01,
                "energy_term_kinetic": 0.7,
                "energy_term_harmonic_trap": 0.8,
                "energy_term_electron_electron": 0.5,
                "local_energy_finite_fraction": 1.0,
                "local_energy_q001": 1.8,
                "local_energy_q01": 1.85,
                "local_energy_q05": 1.9,
                "local_energy_q50": 2.0,
                "local_energy_q95": 2.1,
                "local_energy_q99": 2.15,
                "local_energy_q999": 2.2,
                "local_energy_nonfinite_count": 0,
                "local_energy_error_q001": -0.2,
                "local_energy_error_q01": -0.15,
                "local_energy_error_q05": -0.1,
                "local_energy_error_q50": 0.0,
                "local_energy_error_q95": 0.1,
                "local_energy_error_q99": 0.15,
                "local_energy_error_q999": 0.2,
                "local_energy_error_mean": 0.01,
                "local_energy_abs_error_mean": 0.05,
                "probe_pair_distance/local_energy_max_abs_error": 0.2,
                "probe_pair_distance/local_energy_q95_abs_error": 0.15,
                "probe_pair_distance/nonfinite_count": 0,
                "probe_center_of_mass/local_energy_max_abs_error": 0.2,
                "probe_center_of_mass/local_energy_q95_abs_error": 0.15,
                "probe_center_of_mass/nonfinite_count": 0,
                "checks/exchange/logabs_max_abs_error": 0.0,
                "checks/exchange/logabs_mean_abs_error": 0.0,
                "checks/exchange/sign_failure_count": 0,
                "checks/exchange/nonfinite_count": 0,
                "checks/rotation/logabs_max_abs_error": 0.0,
                "checks/rotation/logabs_mean_abs_error": 0.0,
                "checks/rotation/local_energy_max_abs_error": 0.0,
                "checks/rotation/local_energy_mean_abs_error": 0.0,
                "checks/rotation/nonfinite_count": 0,
                "checks/trace_equivariance/max_abs_error": 0.0,
                "checks/trace_equivariance/mean_abs_error": 0.0,
                "checks/trace_equivariance/failure_count": 0,
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
    _write_fake_diagnostics(run_dir)
    return run_dir


def _write_fake_diagnostics(run_dir: Path) -> None:
    diagnostics = run_dir / "diagnostics"
    energy = diagnostics / "energy"
    pair = diagnostics / "pair_distance_probe"
    com = diagnostics / "center_of_mass_probe"
    exchange = diagnostics / "exchange"
    rotation = diagnostics / "rotation"
    trace = diagnostics / "trace_equivariance"
    for directory in (energy, pair, com, exchange, rotation, trace):
        directory.mkdir(parents=True, exist_ok=True)

    _write_csv(
        energy / "sampled_eval_table.csv",
        [
            {
                "sample_index": 0,
                "local_energy": 2.0,
                "local_energy_error": 0.0,
                "kinetic_energy": 0.7,
                "harmonic_trap_energy": 0.8,
                "electron_electron_energy": 0.5,
                "electron_distance": 1.0,
                "center_of_mass_radius": 0.0,
                "radius_e1": 0.5,
                "radius_e2": 0.5,
                "position_norm_max": 0.5,
                "logabs": -0.1,
                "sign": 1.0,
                "finite": True,
            }
        ],
    )
    probe_rows = [
        {
            "probe_index": index,
            "pair_distance": 0.1 + index * 0.1,
            "center_of_mass_radius": 0.0,
            "direction_id": 0,
            "model_logabs": -0.1 * index,
            "model_sign": 1.0,
            "model_relative_abs_psi": 1.0 - 0.1 * index,
            "model_local_energy": 2.0 + 0.01 * index,
            "model_local_energy_error": 0.01 * index,
            "kinetic_energy": 0.7,
            "harmonic_trap_energy": 0.8,
            "electron_electron_energy": 0.5,
            "finite": True,
            "exact_logabs": -0.1 * index,
            "exact_relative_abs_psi": 1.0 - 0.1 * index,
            "exact_local_energy": 2.0,
            "aligned_logabs_error": 0.0,
            "relative_abs_psi_error": 0.0,
        }
        for index in range(3)
    ]
    _write_csv(pair / "probe.csv", probe_rows)
    _write_csv(
        com / "probe.csv",
        [
            {
                **row,
                "center_of_mass_radius": row["pair_distance"],
                "pair_distance": 1.0,
            }
            for row in probe_rows
        ],
    )
    _write_jsonl(exchange / "trace.jsonl", [{"contract": "symmetric_spatial_singlet", "logabs_abs_error": 0.0}])
    _write_jsonl(rotation / "trace.jsonl", [{"check_type": "spatial_rotation", "logabs_abs_error": 0.0}])
    _write_jsonl(trace / "trace.jsonl", [{"check_type": "semantic_trace_equivariance", "max_abs_error": 0.0}])
    (diagnostics / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_dir": str(run_dir),
                "artifacts": [
                    {"name": "sampled_eval_table", "kind": "csv", "path": "diagnostics/energy/sampled_eval_table.csv", "enabled": True, "expected": False, "exists": True, "readable": True, "rows": 1},
                    {"name": "pair_distance_probe", "kind": "csv", "path": "diagnostics/pair_distance_probe/probe.csv", "enabled": True, "expected": True, "exists": True, "readable": True, "rows": 3},
                    {"name": "center_of_mass_probe", "kind": "csv", "path": "diagnostics/center_of_mass_probe/probe.csv", "enabled": True, "expected": True, "exists": True, "readable": True, "rows": 3},
                    {"name": "exchange_trace", "kind": "jsonl", "path": "diagnostics/exchange/trace.jsonl", "enabled": True, "expected": True, "exists": True, "readable": True, "rows": 1},
                    {"name": "rotation_trace", "kind": "jsonl", "path": "diagnostics/rotation/trace.jsonl", "enabled": True, "expected": True, "exists": True, "readable": True, "rows": 1},
                    {"name": "trace_equivariance_trace", "kind": "jsonl", "path": "diagnostics/trace_equivariance/trace.jsonl", "enabled": True, "expected": True, "exists": True, "readable": True, "rows": 1},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
