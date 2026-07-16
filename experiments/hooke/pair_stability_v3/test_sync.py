"""Focused integration coverage for the V3-only archive sync stage."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR))
for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        del sys.modules[module_name]


def _load_sync() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pair_stability_v3_sync_test", STUDY_DIR / "sync.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = _load_sync()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(root), *args), check=True, text=True, capture_output=True).stdout.strip()


def _lineage(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create one complete minimal report lineage outside the Git source tree."""

    root = tmp_path / "source"
    root.mkdir()
    for relative, text in {
        "run.py": "print('fixture')\n",
        "pyproject.toml": "[project]\nname = 'fixture'\nversion = '0'\n",
        "uv.lock": "version = 1\n",
        ".python-version": "3.14\n",
        "spenn/__init__.py": "\n",
        "experiments/toolkit/README.txt": "toolkit\n",
        f"{sync.STUDY_RELATIVE}/README.md": "study\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(("git", "init", str(root)), check=True, capture_output=True, text=True)
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    revision = _git(root, "rev-parse", "HEAD")

    results = root / sync.STUDY_RELATIVE / "results"
    paths = {
        "grid": results / "00_grid" / "grid",
        "train": results / "01_train" / "scan" / "train",
        "validation": results / "02_validation" / "scan" / "validation",
        "collect": results / "03_collect" / "collect",
        "select": results / "04_select" / "select",
        "final_grid": results / "05_final_grid" / "final-grid",
        "final_train": results / "06_final_train" / "final-run" / "final-train",
        "final_eval": results / "07_final_eval" / "final-run" / "final-eval",
        "final_collect": results / "08_final_collect" / "final-collect",
        "report": results / "09_final_report" / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True)
        (path / "artifact.txt").write_text(path.name, encoding="utf-8")
        _write_json(path / "metadata.json", {"git_commit": revision})
    _write_json(paths["grid"] / "manifest.json", {})
    _write_json(paths["train"] / "source_grid_attempt.json", {"grid_attempt_dir": str(paths["grid"])})
    _write_json(paths["validation"] / "source_train_attempt.json", {"train_attempt_dir": str(paths["train"])})
    _write_json(paths["collect"] / "source_validation_attempts.json", [{"validation_attempt_dir": str(paths["validation"])}])
    _write_json(paths["select"] / "source_collection_attempt.json", {"collection_attempt_dir": str(paths["collect"])})
    _write_json(paths["final_grid"] / "source_selection_attempt.json", {"selection_attempt_dir": str(paths["select"])})
    _write_json(paths["final_train"] / "source_final_grid_attempt.json", {"final_grid_attempt_dir": str(paths["final_grid"])})
    _write_json(paths["final_eval"] / "source_final_train_attempt.json", {"final_train_attempt_dir": str(paths["final_train"])})
    (paths["final_collect"] / "run_index.csv").write_text("final_run_id\nfinal-run\n", encoding="utf-8")
    (paths["final_collect"] / "manifest.yaml").write_text("final_eval_attempt_id: final-eval\n", encoding="utf-8")
    _write_json(paths["report"] / "final_report.json", {"final_collect_attempt_id": "final-collect"})
    checkpoint = paths["final_train"] / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.pt").write_bytes(b"not copied")
    return root, results, revision


def test_sync_archives_complete_lineage_without_checkpoint_payload(tmp_path: Path) -> None:
    source_root, results_root, revision = _lineage(tmp_path)
    destination = tmp_path / "archive"
    plan = sync.build_archive_plan(
        source_root=source_root,
        destination=destination,
        report_attempt_id="report",
        source_revision=revision,
        max_bytes=10_000_000,
    )

    assert set(plan.stage_counts) == sync.REQUIRED_STAGES
    assert plan.skipped_checkpoint_dirs == 1
    assert all("checkpoints" not in entry.relative_path for entry in plan.result_files)
    attempt = sync.write_dry_run(plan, results_root=results_root, attempt_id="sync")
    transfer = sync.execute_sync(sync_attempt_dir=attempt.directory)

    assert transfer["checkpoint_payload_transferred"] is False
    assert (destination / "SOURCE_REVISION").read_text().strip() == revision
    assert not (destination / sync.STUDY_RELATIVE / "results" / "06_final_train" / "final-run" / "final-train" / "checkpoints").exists()
    assert sync.verify_archive(plan=plan, archive_root=destination)["result_file_count"] == len(plan.result_files)

    too_small = sync.build_archive_plan(
        source_root=source_root,
        destination=tmp_path / "too-small",
        report_attempt_id="report",
        source_revision=revision,
        max_bytes=1,
    )
    with pytest.raises(ValueError, match="exceeds"):
        sync.execute_sync(sync_attempt_dir=sync.write_dry_run(too_small, results_root=results_root, attempt_id="too-small").directory)
