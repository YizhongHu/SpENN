"""Study-level tests for the pair-stability V2 major/minor grid."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
GRID = CONFIGS / "grid.yaml"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))


def _load_script(name: str, *, bind_direct: bool = False) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pair_stability_v2_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if bind_direct:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_utils = _load_script("run_utils", bind_direct=True)
plan = _load_script("plan")
select_champions = _load_script("select_champions")
final_plan = _load_script("final_plan")


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


def test_v2_blinding_is_reproducible_by_seed(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for attempt, seed in (("SAME1", 811), ("SAME2", 811), ("DIFF", 812)):
        code = plan.main(
            [
                "--grid",
                str(GRID),
                "--results-root",
                str(results_root),
                "--attempt-id",
                attempt,
                "--blind",
                "--blind-seed",
                str(seed),
            ]
        )
        assert code == 0

    same1 = json.loads((results_root / "00_grid" / "SAME1" / "unblind.json").read_text())
    same2 = json.loads((results_root / "00_grid" / "SAME2" / "unblind.json").read_text())
    diff = json.loads((results_root / "00_grid" / "DIFF" / "unblind.json").read_text())

    assert same1["axes"] == same2["axes"]
    assert same1["axes"] != diff["axes"]


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
    assert manifest["major_axes"] == ["basis", "mechanism"]
    assert manifest["minor_axes"] == ["lr", "channels"]
    assert manifest["scan_seed_axis"] == "seed"
    assert manifest["axis_id_labels"] == {
        "basis": "b",
        "mechanism": "m",
        "lr": "lr",
        "channels": "ch",
        "seed": "seed",
    }
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "mechanism": "run_parameters.mechanism_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert manifest["choice_validation"]["basis"]["choices_path"] == "choices.basis"
    assert manifest["choice_validation"]["mechanism"]["choices_path"] == "choices.mechanism"
    assert [champion["name"] for champion in manifest["champions"]] == ["energy"]
    assert manifest["champion_kinds"] == ["energy"]
    assert manifest["champions"][0]["selector"] == "metric_ladder"
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
    assert manifest["final_replicates"] == 0
    assert manifest["n_jobs"] == 270
    assert manifest["blinding"]["enabled"] is True
    assert manifest["blinding"]["blind_seed"] == 0

    unblind = json.loads((grid_attempt / "unblind.json").read_text())
    assert set(unblind["axes"]) == {"basis", "mechanism"}
    assert set(unblind["axes"]["basis"]["slot_to_value"].values()) == set(OmegaConf.load(GRID).major_grid.basis)
    assert set(unblind["axes"]["mechanism"]["slot_to_value"].values()) == set(OmegaConf.load(GRID).major_grid.mechanism)

    grid = OmegaConf.load(GRID)
    jobs = manifest["jobs"]
    assert {job["choices"]["basis"] for job in jobs} == set(unblind["axes"]["basis"]["slot_to_value"])
    assert {job["choices"]["mechanism"] for job in jobs} == set(unblind["axes"]["mechanism"]["slot_to_value"])
    assert {float(job["choices"]["lr"]) for job in jobs} == {float(value) for value in grid.minor_grid.lr}
    assert {job["choices"]["channels"] for job in jobs} == {int(value) for value in grid.minor_grid.channels}
    assert {job["choices"]["seed"] for job in jobs} == {int(value) for value in grid.scan_seeds}

    job = jobs[0]
    assert job["run_id"].startswith("b-")
    assert "_m-" in job["run_id"]
    assert job["minor_id"].startswith("lr-")
    assert job["minor_choices"]["channels"] == 8
    assert job["scan_seed"] in {0, 1, 2}
    assert job["seed_overrides"]["scan_train"] == {
        "run_parameters.seed": job["scan_seed"],
        "runtime.seed": job["scan_seed"],
        "sampler.seed": job["scan_seed"],
    }
    assert "study.name=pair_stability_v2" in job["overrides"]
    assert "experiment.name=pair_stability_v2" in job["overrides"]
    assert "experiment.run_name=pair_stability_v2_train" in job["overrides"]
    assert f"runtime.seed={job['scan_seed']}" in job["overrides"]
    assert f"sampler.seed={job['scan_seed']}" in job["overrides"]
    assert any(str(override).startswith("run_parameters.basis_slot=B") for override in job["overrides"])
    assert any(str(override).startswith("run_parameters.mechanism_slot=A") for override in job["overrides"])


def _write_collection_summary(results_root: Path) -> None:
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    rows = []
    for job in manifest["jobs"]:
        point = dict(job["choices"])
        lr = float(point["lr"])
        seed = int(point["seed"])
        energy = 2.0 + (0.0 if lr == 3.0e-4 else 0.2)
        feature = 0.01 if lr == 1.0e-3 else 0.03
        rows.append(
            {
                "run_id": job["run_id"],
                "status": "completed",
                **{key: str(value) for key, value in point.items()},
                "major_id": job["major_id"],
                "minor_id": job["minor_id"],
                "config_id": job["config_id"],
                "eval/stratified_geometry/local_energy_mean": str(energy + 0.001 * seed),
                "eval/feature_trace_stability/feature_rms_q95": str(feature + 0.001 * seed),
            }
        )
    collect_dir = results_root / "03_collect" / "C1"
    _write_csv(collect_dir / "summary.csv", rows)
    (collect_dir / "source_grid_attempt.json").write_text(json.dumps({"grid_attempt_id": ATTEMPT}) + "\n")


def test_v2_selects_energy_champions_per_major_and_skips_final_jobs_by_default(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)

    result = select_champions.select(
        results_root=results_root,
        collection_attempt_id="C1",
        select_attempt_id="S1",
    )
    report = result["report"]
    assert report["champion_kinds"] == ["energy"]
    assert [spec["selector"] for spec in report["champion_specs"]] == ["metric_ladder"]
    assert report["group_by"] == ["basis", "mechanism"]
    assert report["n_champions"] == 30

    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    assert len(Counter((row["basis"], row["mechanism"]) for row in champions)) == 30
    assert set(Counter((row["basis"], row["mechanism"]) for row in champions).values()) == {1}
    assert {row["winner_kind"] for row in champions} == {"energy"}
    assert {row["minor_id"] for row in champions} == {"lr-3e-4_ch-8"}

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
    assert manifest["final_replicates"] == 0
    assert manifest["n_jobs"] == 0
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "mechanism": "run_parameters.mechanism_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert jobs == []
