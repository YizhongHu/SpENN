"""Study-level tests for the pair-stability V2 major/minor grid."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
GRID = CONFIGS / "grid.yaml"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import final_plan  # noqa: E402
import plan  # noqa: E402
import run_utils  # noqa: E402
import select_champions  # noqa: E402


ATTEMPT = "20260623T120000-0400"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _planned_results(tmp_path: Path) -> Path:
    results_root = tmp_path / "results"
    code = plan.main(["--grid", str(GRID), "--results-root", str(results_root), "--attempt-id", ATTEMPT])
    assert code == 0
    return results_root


def test_v2_plan_records_major_minor_scan_manifest(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    grid_attempt = run_utils.grid_attempt_dir(results_root, ATTEMPT)
    manifest = json.loads((grid_attempt / "manifest.json").read_text())

    assert manifest["study"] == "pair_stability_v2"
    assert manifest["config_snapshots"] == {
        "train": "train_config.yaml",
        "validation": "validation_config.yaml",
        "smoke": "smoke_config.yaml",
    }
    assert (grid_attempt / "train_config.yaml").is_file()
    assert (grid_attempt / "validation_config.yaml").is_file()
    assert (grid_attempt / "smoke_config.yaml").is_file()
    assert not (grid_attempt / "pair_stability.yaml").exists()
    assert not (grid_attempt / "pair_validation.yaml").exists()
    assert manifest["grid_schema"] == "major_minor_scan"
    assert manifest["major_axes"] == ["architecture", "normalization"]
    assert manifest["minor_axes"] == ["lr", "channels"]
    assert manifest["scan_seed_axis"] == "seed"
    assert manifest["axis_id_labels"] == {
        "architecture": "arch",
        "normalization": "norm",
        "lr": "lr",
        "channels": "ch",
        "seed": "seed",
    }
    assert manifest["axis_overrides"] == {
        "architecture": "run_parameters.architecture",
        "normalization": "run_parameters.normalization",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert manifest["choice_validation"]["architecture"]["choices_path"] == "choices.architecture"
    assert manifest["choice_validation"]["normalization"]["choices_path"] == "choices.normalization"
    assert [champion["name"] for champion in manifest["champions"]] == ["energy", "stability"]
    assert manifest["champion_kinds"] == ["energy", "stability"]
    assert manifest["champions"][0]["selector"] == "metric_ladder"
    assert manifest["champions"][1]["selector"] == "metric"
    assert manifest["champions"][1]["metric"] == "eval/feature_trace_stability/feature_rms_q95"
    assert manifest["seed_overrides"]["scan_train"] == {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "sampler.seed": "scan_seed",
    }
    assert manifest["seed_overrides"]["validation"] == {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "evaluation.seed": "scan_seed",
    }
    assert manifest["final_seed_sequences"] == {
        "final_train_sampler_seed": {"start": 101, "step": 1},
        "final_train_model_seed": {"start": 1001, "step": 1},
        "final_eval_seed": {"start": 10001, "step": 1},
    }
    assert manifest["final_replicates"] == 2
    assert manifest["n_jobs"] == 32

    grid = OmegaConf.load(GRID)
    jobs = manifest["jobs"]
    assert {job["choices"]["architecture"] for job in jobs} == set(grid.major_grid.architecture)
    assert {job["choices"]["normalization"] for job in jobs} == set(grid.major_grid.normalization)
    assert {float(job["choices"]["lr"]) for job in jobs} == {float(value) for value in grid.minor_grid.lr}
    assert {job["choices"]["channels"] for job in jobs} == {int(value) for value in grid.minor_grid.channels}
    assert {job["choices"]["seed"] for job in jobs} == {int(value) for value in grid.scan_seeds}

    job = next(job for job in jobs if job["run_id"] == "arch-raw_envelope_norm-N0_lr-1e-3_ch-4_seed-0")
    assert job["major_id"] == "arch-raw_envelope_norm-N0"
    assert job["minor_id"] == "lr-1e-3_ch-4"
    assert job["config_id"] == "arch-raw_envelope_norm-N0_lr-1e-3_ch-4"
    assert job["major_choices"] == {"architecture": "raw_envelope", "normalization": "N0"}
    assert job["minor_choices"] == {"lr": 0.001, "channels": 4}
    assert job["scan_seed"] == 0
    assert job["seed_overrides"]["scan_train"] == {
        "run_parameters.seed": 0,
        "runtime.seed": 0,
        "sampler.seed": 0,
    }
    assert "study.name=pair_stability_v2" in job["overrides"]
    assert "experiment.name=pair_stability_v2" in job["overrides"]
    assert "experiment.run_name=pair_stability_v2_train" in job["overrides"]
    assert "runtime.seed=0" in job["overrides"]
    assert "sampler.seed=0" in job["overrides"]


def _write_collection_summary(results_root: Path) -> None:
    rows = []
    for architecture in ("raw_envelope", "hermite_o2_envelope"):
        for normalization in ("N0", "N1"):
            for lr in (1.0e-3, 3.0e-4):
                for channels in (4, 8):
                    for seed in (0, 1):
                        point = {
                            "architecture": architecture,
                            "normalization": normalization,
                            "lr": lr,
                            "channels": channels,
                            "seed": seed,
                        }
                        energy = 2.0 + (0.0 if lr == 3.0e-4 and channels == 8 else 0.2)
                        feature = 0.01 if lr == 1.0e-3 and channels == 4 else 0.03
                        rows.append(
                            {
                                "run_id": run_utils.id_for_axes(
                                    point,
                                    ("architecture", "normalization", "lr", "channels", "seed"),
                                    {
                                        "architecture": "arch",
                                        "normalization": "norm",
                                        "lr": "lr",
                                        "channels": "ch",
                                        "seed": "seed",
                                    },
                                ),
                                "status": "completed",
                                **{key: str(value) for key, value in point.items()},
                                "eval/stratified_geometry/local_energy_mean": str(energy + 0.001 * seed),
                                "eval/feature_trace_stability/feature_rms_q95": str(feature + 0.001 * seed),
                            }
                        )
    collect_dir = results_root / "03_collect" / "C1"
    _write_csv(collect_dir / "summary.csv", rows)
    (collect_dir / "source_grid_attempt.json").write_text(json.dumps({"grid_attempt_id": ATTEMPT}) + "\n")


def test_v2_selects_champions_per_major_and_final_jobs_freeze_minor_values(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)

    result = select_champions.select(
        results_root=results_root,
        collection_attempt_id="C1",
        select_attempt_id="S1",
    )
    report = result["report"]
    assert report["champion_kinds"] == ["energy", "stability"]
    assert [spec["selector"] for spec in report["champion_specs"]] == ["metric_ladder", "metric"]
    assert report["group_by"] == ["architecture", "normalization"]
    assert report["n_champions"] == 8

    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    assert Counter((row["architecture"], row["normalization"]) for row in champions) == {
        ("hermite_o2_envelope", "N0"): 2,
        ("hermite_o2_envelope", "N1"): 2,
        ("raw_envelope", "N0"): 2,
        ("raw_envelope", "N1"): 2,
    }
    assert {row["winner_kind"] for row in champions} == {"energy", "stability"}
    assert {row["minor_id"] for row in champions} == {"lr-1e-3_ch-4", "lr-3e-4_ch-8"}

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
        ]
    )
    assert code == 0

    final_dir = results_root / "05_final_grid" / "F1"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    jobs = [json.loads(path.read_text()) for path in sorted((final_dir / "jobs").glob("*.json"))]
    assert manifest["study"] == "pair_stability_v2"
    assert manifest["final_replicates"] == 2
    assert manifest["n_jobs"] == 16
    assert manifest["axis_overrides"] == {
        "architecture": "run_parameters.architecture",
        "normalization": "run_parameters.normalization",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert len(jobs) == 16
    assert "seed" not in jobs[0]
    assert jobs[0]["choices"]
    assert {job["replicate_index"] for job in jobs} == {0, 1}
    assert {job["winner_kind"] for job in jobs} == {"energy", "stability"}
    assert Counter(job["major_id"] for job in jobs) == {
        "arch-hermite_o2_envelope_norm-N0": 4,
        "arch-hermite_o2_envelope_norm-N1": 4,
        "arch-raw_envelope_norm-N0": 4,
        "arch-raw_envelope_norm-N1": 4,
    }
    assert {job["source_scan_seeds"] for job in jobs} == {"0,1"}
    assert {job["final_train_model_seed"] for job in jobs} == {1001, 1002}
    assert {job["final_eval_seed"] for job in jobs} == {10001, 10002}
    assert jobs[0]["stage_seed_overrides"]["final_train"] == {
        "run_parameters.seed": 1001,
        "runtime.seed": 1001,
        "sampler.seed": 101,
    }
    assert jobs[0]["stage_seed_overrides"]["final_eval"] == {
        "run_parameters.seed": 10001,
        "runtime.seed": 10001,
        "evaluation.seed": 10001,
    }
