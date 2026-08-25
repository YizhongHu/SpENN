"""Deterministic render and failure-loud tests for diagnostic reporting."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image

STUDY_DIR = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_report_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load("diagnostic_report")
plan = report.plan_stage

PLAN_ID = "P1"
COLLECT_ID = "C1"
EVAL_SHA = "e" * 40
CHECKPOINTS = ("step_025000", "step_050000")
CHECKPOINT_SHA = {"step_025000": "a" * 64, "step_050000": "b" * 64}
FACTOR_ARMS = report.EXPECTED_FACTOR_ARMS
DIAGNOSTIC_TASKS = (
    "he_radial_profiles",
    "he_en_numerical_atlas",
    "he_ee_ideal_vs_executed_numerical_atlas",
    "he_one_electron_tail_atlas",
    "he_center_of_mass_tail_atlas",
    "he_angular_shell_atlas",
    "full_model_antisymmetry",
    "spatial_exchange_symmetry",
    "rotation_consistency",
    "trace_equivariance",
    "feature_trace",
    "readout_trace",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planned_row(
    checkpoint: str,
    *,
    row_id: str,
    profile: str,
    protocol: str,
    comparison_kind: str,
    seed: int,
    tasks: list[str],
    factor_arm: dict[str, Any] | None = None,
    n_walkers: int = 2,
    n_draws: int = 2,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "kind": "diagnostic_eval",
        "stage": "03_eval",
        "profile": profile,
        "task_names": tasks,
        "protocol": protocol,
        "comparison_kind": comparison_kind,
        "seed": seed,
        "n_walkers": n_walkers,
        "n_draws": n_draws,
        "burn_in": 1,
        "stride": 2,
        "record_capacity": n_walkers * n_draws,
        "diagnostic_samples": n_walkers,
        "factor_arm": factor_arm,
        "checkpoint_label": checkpoint,
        "checkpoint_step": 25000 if checkpoint == "step_025000" else 50000,
        "checkpoint_model_sha256": CHECKPOINT_SHA[checkpoint],
        "checkpoint_manifest_sha256": "c" * 64,
        "checkpoint_complete_sha256": "d" * 64,
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_source_git_sha": "f" * 40,
        "checkpoint_source_tpen_version": "0.3.1",
        "checkpoint_source_dir": f"/facility/{checkpoint}",
        "resources": {
            "partition": "gpu_test",
            "stratum": "a100_mig",
            "constraint": None,
            "timeout_min": 120,
            "cpus": 4,
            "mem_gb": 32,
            "gpus": 1,
        },
    }


def _planned_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for protocol, comparison, seeds in (
            ("primary_256x4096", "primary_headline", range(1000, 1004)),
            ("long_1024x1024", "long_chain_diagnostic", range(2000, 2004)),
        ):
            for seed in seeds:
                rows.append(
                    _planned_row(
                        checkpoint,
                        row_id=f"{checkpoint}-{protocol}-seed{seed:04d}",
                        profile="retained_energy",
                        protocol=protocol,
                        comparison_kind=comparison,
                        seed=seed,
                        tasks=["retained_energy"],
                    )
                )
        for protocol, comparison, seed in (
            ("burn_in_50", "burn_in_sensitivity", 3000),
            ("burn_in_200", "burn_in_sensitivity", 3001),
            ("stride_10", "stride_sensitivity", 3100),
            ("stride_40", "stride_sensitivity", 3101),
        ):
            rows.append(
                _planned_row(
                    checkpoint,
                    row_id=f"{checkpoint}-{protocol}-seed{seed:04d}",
                    profile="retained_energy",
                    protocol=protocol,
                    comparison_kind=comparison,
                    seed=seed,
                    tasks=["retained_energy"],
                )
            )
        rows.append(
            _planned_row(
                checkpoint,
                row_id=f"{checkpoint}-factor-common-configuration",
                profile="common_factor_response",
                protocol="factor_common_configuration",
                comparison_kind="common_configuration",
                seed=5000,
                tasks=["common_factor_response"],
                n_draws=1,
            )
        )
        for index, arm in enumerate(FACTOR_ARMS):
            parameter, scale = report._factor_coordinate(arm)
            coordinates = {
                "label": arm,
                "b_ee": scale if parameter == "b_ee" else 1.0,
                "c_electron_nucleus": scale if parameter == "c_electron_nucleus" else 1.0,
                "d_electron_nucleus": scale if parameter == "d_electron_nucleus" else 1.0,
            }
            rows.append(
                _planned_row(
                    checkpoint,
                    row_id=f"{checkpoint}-factor-reequilibrated-{arm}",
                    profile="reequilibrated_energy",
                    protocol=f"factor_reequilibrated/{arm}",
                    comparison_kind="re_equilibrated",
                    seed=4000 + index,
                    tasks=["reequilibrated_energy"],
                    factor_arm=coordinates,
                )
            )
        rows.append(
            _planned_row(
                checkpoint,
                row_id=f"{checkpoint}-checkpoint-diagnostics",
                profile="checkpoint_diagnostics",
                protocol="checkpoint_diagnostics",
                comparison_kind="checkpoint_diagnostics",
                seed=6000,
                tasks=list(DIAGNOSTIC_TASKS),
                n_draws=1,
            )
        )
    assert len(rows) == 42
    return rows


def _artifact(
    path: Path,
    *,
    row_id: str,
    task: str,
    name: str,
    kind: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "namespace": f"he_v1_diagnostic_v1/{task}",
        "name": name,
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_receipt(row: dict[str, Any], *, available: bool = True) -> dict[str, Any]:
    status = "available" if available else "unresolved"
    statistics = (
        {
            "tau_int": 1.2,
            "ess": 3.3333333333333335,
            "mcse": 0.02,
            "mean": -2.90 + (0.002 if row["checkpoint_label"] == "step_025000" else 0.001),
            "variance": 0.04,
        }
        if available
        else None
    )
    return {
        "stage": "03_eval",
        "run_id": row["row_id"],
        "attempt_id": PLAN_ID,
        "checkpoint_sha256": row["checkpoint_model_sha256"],
        "config_sha256": "9" * 64,
        "observable": "local_energy",
        "evaluator_id": "he-v1-diagnostic-v1",
        "status": status,
        "recorded_at_utc": "2026-08-24T00:00:00Z",
        "estimator_id": "pooled_geyer_ips",
        "estimator_version": "1",
        "tau_convention": "tau = 1 + 2 sum rho",
        "reason": None if available else "fixture plateau unresolved",
        "warnings": [],
        "chains": [],
        "shape": {
            "walker_count": row["n_walkers"],
            "draw_count": row["n_draws"],
            "total_draws": row["record_capacity"],
            "draw_stride": row["stride"],
            "burn_in_draws": row["burn_in"],
        },
        "plateau": {"plateau_reached": available, "truncation_lag": 1, "pair_count": 1, "max_lag": 1},
        "mixing": {"r_hat": 1.0, "n_split_chains": 4, "draws_per_split_chain": 1, "reason": None},
        "statistics": statistics,
    }


def _trajectory_artifacts(
    run_dir: Path,
    row: dict[str, Any],
    *,
    task: str,
    available: bool,
    primary: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = run_dir / task
    task_dir.mkdir(parents=True)
    receipt_path = task_dir / "trajectory_statistics.jsonl"
    receipt = _trajectory_receipt(row, available=available)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [
        _artifact(
            receipt_path,
            row_id=row["row_id"],
            task=task,
            name="local_energy_trajectory_statistics",
            kind="trajectory_statistics_sidecar",
            metadata={"status": "available" if available else "unresolved"},
        )
    ]
    metrics = {
        "local_energy_mean": -2.89,
        "local_energy_stderr": 0.01,
        "local_energy_trajectory_statistics_available": available,
        "sampler_trajectory_retained_draw_acceptance_rate_mean": 0.51,
    }
    if available:
        statistics = receipt["statistics"]
        assert statistics is not None
        metrics.update(
            {
                "local_energy_mcse": statistics["mcse"],
                "local_energy_ess": statistics["ess"],
                "local_energy_tau_int": statistics["tau_int"],
                "local_energy_trajectory_mean": statistics["mean"],
                "local_energy_trajectory_variance": statistics["variance"],
                "local_energy_stderr_iid": math.sqrt(
                    statistics["variance"] / row["record_capacity"]
                ),
                "local_energy_mcse_inflation": statistics["mcse"]
                / math.sqrt(statistics["variance"] / row["record_capacity"]),
            }
        )
    if primary:
        sampled = task_dir / "sampled_eval_table.csv"
        energy_values = [-3.0, -2.9, -2.8, -2.7]
        _write_csv(
            sampled,
            ["sample_index", "draw_index", "walker_index", "local_energy", "finite"],
            [
                {
                    "sample_index": index,
                    "draw_index": index // row["n_walkers"],
                    "walker_index": index % row["n_walkers"],
                    "local_energy": energy,
                    "finite": True,
                }
                for index, energy in enumerate(energy_values)
            ],
        )
        sampled_artifact = _artifact(
            sampled,
            row_id=row["row_id"],
            task=task,
            name="sampled_eval_table",
            kind="csv",
            metadata={
                "rows": row["record_capacity"],
                "truncated": False,
                "selection": "complete_draw_walker_grid",
            },
        )
        artifacts.append(sampled_artifact)
        conditioned = task_dir / "conditioned_local_energy.json"
        conditioned.write_text(
            json.dumps(
                _conditioned_payload(row, sampled_artifact),
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            _artifact(
                conditioned,
                row_id=row["row_id"],
                task=task,
                name="conditioned_local_energy",
                kind="json",
                metadata={"source_csv_sha256": sampled_artifact["sha256"]},
            )
        )
        sampler = task_dir / "sampler_trajectory_diagnostics.json"
        sampler.write_text(
            json.dumps(
                {
                    "schema": "sampler_trajectory_diagnostics/v1",
                    "n_walkers": row["n_walkers"],
                    "draw_stride": row["stride"],
                    "sampler_burn_in": row["burn_in"],
                    "retained_draws": [{}, {}],
                    "discarded_draws": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            _artifact(
                sampler,
                row_id=row["row_id"],
                task=task,
                name="sampler_trajectory_diagnostics",
                kind="sampler_trajectory_diagnostics",
                metadata={"retained_draw_count": row["n_draws"]},
            )
        )
    return artifacts, [
        {"namespace": f"he_v1_diagnostic_v1/{task}", "step": 0, "metrics": metrics}
    ]


def _conditioned_payload(row: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    bins = [
        _condition_bin("underflow", "underflow", "-inf", 0.01, 0, 0.0),
        _condition_bin("range_000", "range", 0.01, 0.1, 2, 0.02),
        _condition_bin("range_001", "range", 0.1, 1.0, 2, 0.02),
        _condition_bin("overflow", "overflow", 1.0, "inf", 0, 0.0),
        _condition_bin("nonfinite", "nonfinite", None, None, 0, 0.0),
    ]
    pass_receipt = {
        "csv_sha256": sampled["sha256"],
        "byte_count": sampled["bytes"],
        "row_count": row["record_capacity"],
    }
    return {
        "schema": "conditioned_local_energy/v1",
        "source": {
            "trajectory_record_schema": "trajectory_records/v1",
            "csv_sha256": sampled["sha256"],
            "byte_count": sampled["bytes"],
            "row_count": row["record_capacity"],
            "draw_count": row["n_draws"],
            "walker_count": row["n_walkers"],
            "two_pass_identity_confirmed": True,
            "statistics_pass": pass_receipt,
            "rare_events_pass": pass_receipt,
        },
        "estimator": {"headline_estimator": False},
        "configuration": {
            "range_edges": {
                quantity: [0.01, 0.1, 1.0]
                for quantity in report.REQUIRED_CONDITION_QUANTITIES
            },
            "deviation_ccdf_thresholds": [0.01, 0.1, 1.0],
        },
        "global": {
            "finite_local_energy_count": row["record_capacity"],
            "nonfinite_local_energy_count": 0,
            "second_moment_about_mean": 0.04,
        },
        "range_conditioned": {
            quantity: {
                "predeclared_edges": [0.01, 0.1, 1.0],
                "structural_bins": ["underflow", "overflow", "nonfinite"],
                "bins": bins,
                "reconciliation": {
                    "finite_count_sum": row["record_capacity"],
                    "global_finite_count": row["record_capacity"],
                    "probability_sum": 1.0,
                    "second_moment_contribution_sum": 0.04,
                    "global_second_moment": 0.04,
                },
            }
            for quantity in report.REQUIRED_CONDITION_QUANTITIES
        },
        "rare_events": {
            "absolute_deviation_ccdf": [
                {"threshold": 0.01, "count": 4, "probability_over_finite_local_energy": 1.0},
                {"threshold": 0.1, "count": 2, "probability_over_finite_local_energy": 0.5},
                {"threshold": 1.0, "count": 1, "probability_over_finite_local_energy": 0.25},
            ]
        },
    }


def _condition_bin(
    bin_id: str,
    kind: str,
    lower: Any,
    upper: Any,
    finite_count: int,
    contribution: float,
) -> dict[str, Any]:
    return {
        "id": bin_id,
        "kind": kind,
        "lower": lower,
        "upper": upper,
        "observables": {"local_energy": {"finite_count": finite_count}},
        "variance_attribution": {
            "probability": finite_count / 4,
            "second_moment_contribution": contribution,
        },
    }


def _common_factor_artifacts(
    run_dir: Path,
    row: dict[str, Any],
    *,
    comparison_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task = "common_factor_response"
    task_dir = run_dir / task
    path = task_dir / "factor_response.csv"
    fields = [
        "arm",
        "sample_index",
        "comparison_kind",
        "parameter_scales",
        "local_energy",
        "delta_local_energy_from_baseline",
        "delta_logabs_from_baseline",
        "finite",
    ]
    records = []
    for arm_index, arm in enumerate(FACTOR_ARMS):
        parameter, scale = report._factor_coordinate(arm)
        scales = {
            "b_ee": scale if parameter == "b_ee" else 1.0,
            "c_electron_nucleus": scale if parameter == "c_electron_nucleus" else 1.0,
            "d_electron_nucleus": scale if parameter == "d_electron_nucleus" else 1.0,
        }
        for sample in range(row["record_capacity"]):
            records.append(
                {
                    "arm": arm,
                    "sample_index": sample,
                    "comparison_kind": comparison_kind,
                    "parameter_scales": json.dumps(scales, sort_keys=True),
                    "local_energy": -2.9 + 0.001 * arm_index,
                    "delta_local_energy_from_baseline": 0.001 * arm_index,
                    "delta_logabs_from_baseline": 0.002 * arm_index,
                    "finite": True,
                }
            )
    _write_csv(path, fields, records)
    artifact = _artifact(
        path,
        row_id=row["row_id"],
        task=task,
        name="factor_response_common_configuration",
        kind="csv",
        metadata={
            "comparison_kind": "common_configuration",
            "baseline_label": "baseline",
            "rows": row["record_capacity"] * len(FACTOR_ARMS),
            "arm_count": len(FACTOR_ARMS),
            "configuration_count": row["record_capacity"],
            "selection": "complete_common_configuration_grid",
            "model_state_restored": True,
        },
    )
    return [artifact], [
        {
            "namespace": f"he_v1_diagnostic_v1/{task}",
            "step": 0,
            "metrics": {"comparison_is_common_configuration": True},
        }
    ]


def _diagnostic_artifacts(
    run_dir: Path,
    row: dict[str, Any],
    *,
    cusp_available: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    radial_dir = run_dir / "he_radial_profiles"
    radial = radial_dir / "electron_nucleus_radial_profile.csv"
    _write_csv(radial, ["measurement_index", "radius", "radial_dlogabs", "available"], [{"measurement_index": 0, "radius": 0.001, "radial_dlogabs": -2.0, "available": cusp_available}])
    artifacts.append(_artifact(radial, row_id=row["row_id"], task="he_radial_profiles", name="electron_nucleus_radial_profile", kind="csv", metadata={"rows": 1}))
    cusp_metrics: dict[str, Any] = {"cusp_available": cusp_available}
    if cusp_available:
        cusp_metrics.update({"cusp_expected_slope": -2.0, "cusp_one_sided_slope_mean": -1.99})
    metrics.append({"namespace": "he_v1_diagnostic_v1/he_radial_profiles", "step": 0, "metrics": cusp_metrics})

    en_fields = [
        "realized_physical_coordinate",
        "is_exact_zero_sentinel",
        "executed_electron_nucleus_factor_first_derivative",
        "executed_electron_nucleus_factor_first_derivative_finite",
        "executed_full_logabs_second_derivative",
        "executed_full_logabs_second_derivative_finite",
    ]
    en_rows = [
        {
            "realized_physical_coordinate": radius,
            "is_exact_zero_sentinel": False,
            "executed_electron_nucleus_factor_first_derivative": -2.0 + radius,
            "executed_electron_nucleus_factor_first_derivative_finite": True,
            "executed_full_logabs_second_derivative": 0.02 + radius,
            "executed_full_logabs_second_derivative_finite": True,
        }
        for radius in (1.0e-6, 1.0e-4, 1.0e-2)
    ]
    en = run_dir / "he_en_numerical_atlas" / "helium_en_numerical_atlas.csv"
    _write_csv(en, en_fields, en_rows)
    artifacts.append(_artifact(en, row_id=row["row_id"], task="he_en_numerical_atlas", name="helium_atlas", kind="csv", metadata={"rows": len(en_rows)}))
    metrics.append({"namespace": "he_v1_diagnostic_v1/he_en_numerical_atlas", "step": 0, "metrics": {"atlas_total_count": len(en_rows)}})

    ee_fields = [
        "realized_physical_coordinate",
        "is_exact_zero_sentinel",
        "executed_smoothed_ee_factor_first_derivative",
        "executed_smoothed_ee_factor_first_derivative_finite",
        "executed_hamiltonian_cancellation_abs_sum",
        "executed_hamiltonian_cancellation_abs_sum_finite",
        "executed_hamiltonian_cancellation_residual",
        "executed_hamiltonian_cancellation_residual_finite",
        "executed_hamiltonian_cancellation_ratio",
        "executed_hamiltonian_cancellation_ratio_finite",
    ]
    ee_rows = [
        {
            "realized_physical_coordinate": radius,
            "is_exact_zero_sentinel": False,
            "executed_smoothed_ee_factor_first_derivative": 0.45 + radius,
            "executed_smoothed_ee_factor_first_derivative_finite": True,
            "executed_hamiltonian_cancellation_abs_sum": 1.0 / radius,
            "executed_hamiltonian_cancellation_abs_sum_finite": True,
            "executed_hamiltonian_cancellation_residual": 0.1 + radius,
            "executed_hamiltonian_cancellation_residual_finite": True,
            "executed_hamiltonian_cancellation_ratio": 10.0 / radius,
            "executed_hamiltonian_cancellation_ratio_finite": True,
        }
        for radius in (1.0e-6, 1.0e-4, 1.0e-2)
    ]
    ee_task = "he_ee_ideal_vs_executed_numerical_atlas"
    ee = run_dir / ee_task / "helium_ideal_vs_executed.csv"
    _write_csv(ee, ee_fields, ee_rows)
    artifacts.append(_artifact(ee, row_id=row["row_id"], task=ee_task, name="helium_atlas", kind="csv", metadata={"rows": len(ee_rows)}))
    metrics.append({"namespace": f"he_v1_diagnostic_v1/{ee_task}", "step": 0, "metrics": {"atlas_total_count": len(ee_rows)}})

    for task, prefix in (
        ("he_one_electron_tail_atlas", "executed_full_logabs_one_electron_tail"),
        ("he_center_of_mass_tail_atlas", "executed_full_logabs_center_of_mass_tail"),
        ("he_angular_shell_atlas", "executed_full_logabs_angular_shell_tail"),
    ):
        tail = run_dir / task / "helium_tail.csv"
        tail_rows = [
            {
                "realized_physical_coordinate": radius,
                "executed_full_logabs_value": -1.5 * radius,
                "executed_full_logabs_value_finite": True,
            }
            for radius in (2.0, 4.0, 8.0)
        ]
        _write_csv(tail, ["realized_physical_coordinate", "executed_full_logabs_value", "executed_full_logabs_value_finite"], tail_rows)
        artifacts.append(_artifact(tail, row_id=row["row_id"], task=task, name="helium_atlas", kind="csv", metadata={"rows": len(tail_rows)}))
        metrics.append({"namespace": f"he_v1_diagnostic_v1/{task}", "step": 0, "metrics": {f"{prefix}_available": True, f"{prefix}_slope": -1.5}})

    metric_payloads = {
        "full_model_antisymmetry": {"logabs_max_abs_error": 1.0e-10, "sign_failure_count": 0},
        "spatial_exchange_symmetry": {"triplet_fraction_mean_under_psi_orig_sq": 1.0e-6},
        "rotation_consistency": {"local_energy_max_abs_error": 2.0e-9},
        "trace_equivariance": {"max_abs_error": 3.0e-10, "comparison_error_count": 0},
        "feature_trace": {"feature_nonfinite_count": 0},
        "readout_trace": {"readout_nonfinite_count": 0},
    }
    artifact_names = {
        "full_model_antisymmetry": "transform_records",
        "spatial_exchange_symmetry": "transform_records",
        "rotation_consistency": "transform_records",
        "trace_equivariance": "trace_records",
        "feature_trace": "trace_records",
        "readout_trace": "trace_records",
    }
    for task, payload in metric_payloads.items():
        path = run_dir / task / "records.csv"
        _write_csv(path, ["sample_index", "value"], [{"sample_index": 0, "value": 0.0}])
        artifacts.append(_artifact(path, row_id=row["row_id"], task=task, name=artifact_names[task], kind="csv", metadata={"rows": 1}))
        metrics.append({"namespace": f"he_v1_diagnostic_v1/{task}", "step": 0, "metrics": payload})
    return artifacts, metrics


def _materialize_fixture(
    root: Path,
    *,
    unresolved_primary: bool = False,
    unresolved_factor_baseline: bool = False,
    cusp_available: bool = True,
    common_comparison_kind: str = "common_configuration",
) -> dict[str, Any]:
    rows = _planned_rows()
    production_hash = "7" * 64
    manifest: dict[str, Any] = {
        "schema": plan.PLAN_SCHEMA,
        "study": "he-v1-diagnostic-v1",
        "scale": "smoke",
        "evaluation_git_sha": EVAL_SHA,
        "created_at": "2026-08-24T00:00:00-04:00",
        "production_grid_sha256_before": production_hash,
        "checkpoint_reporting": "report_both_without_selection",
        "production_run_mutation_authorized": False,
        "checkpoints": [
            {"label": checkpoint, "model_sha256": CHECKPOINT_SHA[checkpoint]}
            for checkpoint in CHECKPOINTS
        ],
        "rows": rows,
    }
    identity = dict(manifest)
    identity.pop("created_at")
    manifest["plan_sha256"] = plan.canonical_sha256(identity)
    plan_dir = root / "00_plan" / PLAN_ID
    plan_dir.mkdir(parents=True)
    (plan_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    collected_rows = []
    unresolved_id = f"step_025000-primary_256x4096-seed1000"
    for planned in rows:
        row_id = planned["row_id"]
        run_dir = root / "03_eval" / row_id / PLAN_ID / row_id
        run_dir.mkdir(parents=True)
        profile = planned["profile"]
        if profile in {"retained_energy", "reequilibrated_energy"}:
            unresolved_reequilibrated = (
                unresolved_factor_baseline
                and profile == "reequilibrated_energy"
                and planned["checkpoint_label"] == "step_025000"
                and planned["factor_arm"]["label"] == "baseline"
            )
            available = not (
                (unresolved_primary and row_id == unresolved_id)
                or unresolved_reequilibrated
            )
            artifacts, metric_records = _trajectory_artifacts(
                run_dir,
                planned,
                task=planned["task_names"][0],
                available=available,
                primary=planned["comparison_kind"] == "primary_headline",
            )
        elif profile == "common_factor_response":
            artifacts, metric_records = _common_factor_artifacts(
                run_dir,
                planned,
                comparison_kind=common_comparison_kind,
            )
        else:
            artifacts, metric_records = _diagnostic_artifacts(
                run_dir,
                planned,
                cusp_available=cusp_available,
            )
        metric_records.extend(
            [
                {"namespace": "eval/perf", "step": 0, "metrics": {"wall_time_sec": 2.0}},
                {
                    "namespace": "runtime",
                    "step": 0,
                    "metrics": {
                        "wall_time_sec": 2.5,
                        "peak_memory_mb": 128.0,
                        "cuda_max_memory_allocated_mb": 64.0,
                    },
                },
            ]
        )
        collected_rows.append(
            {
                **{key: planned[key] for key in (
                    "row_id", "checkpoint_label", "checkpoint_model_sha256", "profile",
                    "protocol", "comparison_kind", "factor_arm", "seed", "n_walkers",
                    "n_draws", "burn_in", "stride", "task_names",
                )},
                "config_sha256": "9" * 64,
                "job_id": "123",
                "delivered_device": "NVIDIA A100 MIG",
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "metrics": metric_records,
            }
        )
    collected = {
        "schema": report.collect_stage.COLLECT_SCHEMA,
        "study": manifest["study"],
        "scale": manifest["scale"],
        "plan_attempt_id": PLAN_ID,
        "plan_sha256": manifest["plan_sha256"],
        "launch_attempt_id": "L1",
        "collect_attempt_id": COLLECT_ID,
        "evaluation_git_sha": EVAL_SHA,
        "created_at": "2026-08-24T01:00:00-04:00",
        "checkpoint_reporting": "both_without_selection",
        "selection_policy": "none",
        "production_run_mutation_authorized": False,
        "production_grid_sha256_before": production_hash,
        "production_grid_sha256_after": production_hash,
        "status": "success",
        "n_planned_rows": 42,
        "n_collected_rows": 42,
        "checkpoints": [
            {"checkpoint_label": checkpoint, "checkpoint_model_sha256": CHECKPOINT_SHA[checkpoint]}
            for checkpoint in CHECKPOINTS
        ],
        "rows": collected_rows,
        "errors": [],
    }
    collect_dir = root / "04_collect" / COLLECT_ID
    collect_dir.mkdir(parents=True)
    (collect_dir / "collected.json").write_text(json.dumps(collected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "collected": collected}


def _render(root: Path, attempt: str = "R1") -> dict[str, Any]:
    return report.build_report(
        results_root=root,
        plan_attempt_id=PLAN_ID,
        collect_attempt_id=COLLECT_ID,
        report_attempt_id=attempt,
    )


@pytest.mark.integration
def test_cusp_curvature_figure_has_three_semantic_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    rows = [
        {
            "view": "electron_nucleus",
            "series": "executed_electron_nucleus_factor",
            "available": True,
            "checkpoint_label": "step_025000",
            "radius_bohr": 0.1,
            "first_derivative": -1.9,
        },
        {
            "view": "electron_nucleus",
            "series": "analytic_ideal_cusp_law",
            "available": True,
            "first_derivative": -2.0,
        },
        {
            "view": "electron_electron",
            "series": "executed_smoothed_ee_factor",
            "available": True,
            "checkpoint_label": "step_025000",
            "radius_bohr": 0.1,
            "first_derivative": 0.45,
        },
        {
            "view": "electron_electron",
            "series": "analytic_ideal_cusp_law",
            "available": True,
            "first_derivative": 0.5,
        },
        {
            "view": "curvature",
            "series": "executed_full_logabs",
            "available": True,
            "checkpoint_label": "step_025000",
            "radius_bohr": 0.1,
            "second_derivative": 1.25,
        },
    ]
    figure = report.plot_stage.cusp_curvature_figure(rows)
    try:
        assert tuple(axis.get_title() for axis in figure.axes) == (
            "Electron–nucleus cusp",
            "Electron–electron cusp",
            "Direct executed curvature",
        )
        assert tuple(len(axis.lines) for axis in figure.axes) == (2, 2, 1)
        assert any(
            "No universal Kato curvature target" in text.get_text()
            for text in figure.axes[2].texts
        )
    finally:
        report.plot_stage.pyplot().close(figure)


@pytest.mark.integration
def test_tails_figure_has_two_semantic_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    rows = [
        {
            "view": "one_electron",
            "available": True,
            "checkpoint_label": "step_025000",
            "radius_bohr": 1.0,
            "executed_logabs": -1.0,
            "outer_slope_bohr_inv": -1.0,
        },
        {
            "view": "center_of_mass",
            "available": True,
            "checkpoint_label": "step_025000",
            "radius_bohr": 1.0,
            "executed_logabs": -2.0,
            "outer_slope_bohr_inv": -2.0,
        },
    ]
    figure = report.plot_stage.tails_figure(rows)
    try:
        assert tuple(axis.get_title() for axis in figure.axes) == (
            "One-electron escape",
            "Centre-of-mass escape",
        )
        assert tuple(len(axis.lines) for axis in figure.axes) == (1, 1)
        assert all(axis.get_xlabel() == "Escape radius (bohr)" for axis in figure.axes)
        assert all(axis.get_ylabel() == r"executed $\log|\Psi|$" for axis in figure.axes)
    finally:
        report.plot_stage.pyplot().close(figure)


@pytest.mark.integration
def test_factor_response_figure_has_two_semantic_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    rows = [
        {
            "comparison_basis": "fixed_configuration_paired",
            "status": "available",
            "checkpoint_label": "step_025000",
            "arm_label": "kinetic_minus_10pct",
            "delta_energy_ha": -0.01,
            "delta_uncertainty_ha": 0.001,
        },
        {
            "comparison_basis": "re_equilibrated_independent",
            "status": "available",
            "checkpoint_label": "step_025000",
            "arm_label": "envelope_plus_10pct",
            "delta_energy_ha": 0.02,
            "delta_uncertainty_ha": 0.002,
        },
    ]
    figure = report.plot_stage.factor_response_figure(rows)
    try:
        assert tuple(axis.get_title() for axis in figure.axes) == (
            "Fixed configurations (paired response)",
            "Re-equilibrated chains (independent estimates)",
        )
        assert tuple(len(axis.containers) for axis in figure.axes) == (1, 1)
        assert tuple(
            tuple(label.get_text() for label in axis.get_xticklabels())
            for axis in figure.axes
        ) == (("kinetic −10%",), ("envelope +10%",))
        assert all(
            axis.get_ylabel() == r"$\Delta E_L$ from baseline (Ha)"
            for axis in figure.axes
        )
    finally:
        report.plot_stage.pyplot().close(figure)


@pytest.mark.integration
def test_fixture_render_is_byte_deterministic_and_publication_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    left = tmp_path / "left"
    right = tmp_path / "right"
    _materialize_fixture(left)
    _materialize_fixture(right)
    left_manifest = _render(left)
    right_manifest = _render(right)

    assert len(left_manifest["figures"]) == 27
    assert len(left_manifest["plot_data"]) == 9
    assert len(left_manifest["memory_mapped_record_artifacts"]) == 18
    assert not any(
        item["task"] == "he_angular_shell_atlas"
        for item in left_manifest["memory_mapped_record_artifacts"]
    )
    left_report = left / "05_report" / "R1"
    right_report = right / "05_report" / "R1"
    for relative in [
        *(f"figures/{name}.{suffix}" for name in report.TABLE_NAMES for suffix in ("svg", "pdf", "png")),
        *(f"plot-data/{name}.csv" for name in report.TABLE_NAMES),
        "report.md",
    ]:
        assert (left_report / relative).read_bytes() == (right_report / relative).read_bytes(), relative

    for name in report.TABLE_NAMES:
        svg = left_report / "figures" / f"{name}.svg"
        pdf = left_report / "figures" / f"{name}.pdf"
        png = left_report / "figures" / f"{name}.png"
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg") and root.get("viewBox")
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert b"%%EOF" in pdf.read_bytes()[-32:]
        with Image.open(png) as image:
            assert image.width > 1000 and image.height > 500
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(300.0, abs=0.2)
            assert dpi[1] == pytest.approx(300.0, abs=0.2)
    markdown = (left_report / "report.md").read_text(encoding="utf-8")
    assert "correlation-aware MCSE" in markdown
    assert "Direct second derivatives are descriptive" in markdown
    assert "paired fixed-configuration" in markdown
    assert "memory-mapped record CSVs" in markdown
    assert "selection_policy" not in markdown
    for name in report.TABLE_NAMES:
        assert f"](figures/{name}.svg)" in markdown
        header = (left_report / "plot-data" / f"{name}.csv").read_text(encoding="utf-8").splitlines()[0]
        for key in report.PROVENANCE_KEYS:
            assert key in header


def test_hash_mismatch_fails_before_report_directory_is_created(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    artifact = next(
        artifact
        for row in fixture["collected"]["rows"]
        for artifact in row["artifacts"]
        if artifact["name"] == "sampled_eval_table"
    )
    path = Path(artifact["path"])
    payload = bytearray(path.read_bytes())
    payload[0] = ord("x") if payload[0] != ord("x") else ord("y")
    path.write_bytes(bytes(payload))
    with pytest.raises(report.DiagnosticReportError, match="artifact hash mismatch"):
        _render(tmp_path)
    assert not (tmp_path / "05_report" / "R1").exists()


def test_incomplete_collection_fails_loudly(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    collected = fixture["collected"]
    collected["rows"].pop()
    collected["n_collected_rows"] = 41
    path = tmp_path / "04_collect" / COLLECT_ID / "collected.json"
    path.write_text(json.dumps(collected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(report.DiagnosticReportError, match="collection receipt verification failed"):
        _render(tmp_path)


def test_missing_task_artifact_fails_loudly(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    row = next(
        row
        for row in fixture["collected"]["rows"]
        if row["profile"] == "checkpoint_diagnostics"
    )
    row["artifacts"] = [
        artifact
        for artifact in row["artifacts"]
        if artifact["task"] != "he_angular_shell_atlas"
    ]
    row["artifact_count"] = len(row["artifacts"])
    path = tmp_path / "04_collect" / COLLECT_ID / "collected.json"
    path.write_text(
        json.dumps(fixture["collected"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(report.DiagnosticReportError, match="artifact task coverage changed"):
        _render(tmp_path)


def test_conditioned_reconciliation_corruption_fails_loudly(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    artifact = next(
        artifact
        for row in fixture["collected"]["rows"]
        for artifact in row["artifacts"]
        if artifact["name"] == "conditioned_local_energy"
    )
    path = Path(artifact["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    quantity = report.REQUIRED_CONDITION_QUANTITIES[0]
    payload["range_conditioned"][quantity]["reconciliation"]["probability_sum"] = 0.5
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact["sha256"] = _sha(path)
    artifact["bytes"] = path.stat().st_size
    collected_path = tmp_path / "04_collect" / COLLECT_ID / "collected.json"
    collected_path.write_text(
        json.dumps(fixture["collected"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        report.DiagnosticReportError,
        match="conditioned artifact verification failed",
    ):
        _render(tmp_path)


def test_trajectory_receipt_metric_disagreement_fails_loudly(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    row = next(
        row
        for row in fixture["collected"]["rows"]
        if row["comparison_kind"] == "primary_headline"
    )
    namespace = f"he_v1_diagnostic_v1/{row['task_names'][0]}"
    metrics = next(record["metrics"] for record in row["metrics"] if record["namespace"] == namespace)
    metrics["local_energy_trajectory_mean"] += 0.1
    path = tmp_path / "04_collect" / COLLECT_ID / "collected.json"
    path.write_text(
        json.dumps(fixture["collected"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(report.DiagnosticReportError, match="trajectory receipt/metric mismatch"):
        _render(tmp_path)


@pytest.mark.integration
def test_unresolved_energy_and_unavailable_cusp_remain_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    _materialize_fixture(tmp_path, unresolved_primary=True, cusp_available=False)
    _render(tmp_path)
    output = tmp_path / "05_report" / "R1"
    energy = (output / "plot-data" / "energy_mcse.csv").read_text(encoding="utf-8")
    cusp = (output / "plot-data" / "cusp_curvature.csv").read_text(encoding="utf-8")
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "unresolved" in energy and "unavailable" in energy
    assert "targeted_cusp_summary" in cusp and "false" in cusp and "unavailable" in cusp
    assert "Unavailable, absent, or unresolved metrics" in markdown
    assert "fixture plateau unresolved" in markdown


def test_common_configuration_rows_cannot_be_relabeled(tmp_path: Path) -> None:
    _materialize_fixture(tmp_path, common_comparison_kind="re_equilibrated")
    study = report.read_verified_study(
        tmp_path,
        plan_attempt_id=PLAN_ID,
        collect_attempt_id=COLLECT_ID,
    )
    with pytest.raises(report.DiagnosticReportError, match="common factor row mislabeled"):
        report._factor_rows(study)


@pytest.mark.integration
def test_unresolved_factor_baseline_stays_unavailable_and_renderable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    _materialize_fixture(tmp_path, unresolved_factor_baseline=True)
    _render(tmp_path)
    table = (
        tmp_path / "05_report" / "R1" / "plot-data" / "factor_response.csv"
    ).read_text(encoding="utf-8")
    assert "re_equilibrated_independent" in table
    assert "unresolved" in table
    assert "fixture plateau unresolved" in table
