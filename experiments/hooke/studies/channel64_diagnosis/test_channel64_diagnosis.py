"""Tests for the channel-64 diagnosis study."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def load_analyze_module():
    """Load the adjacent analysis script without requiring a package import."""

    module_path = Path(__file__).with_name("analyze.py")
    spec = importlib.util.spec_from_file_location("channel64_diagnosis_analyze", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_diagnosis_identifies_lr_pocket_without_overfit() -> None:
    analyze = load_analyze_module()
    runs = [
        _run("selected32", 32, 0.003, "sigmoid", 3, train=2.0010, valid=2.0012, train_var=0.0020, valid_var=0.0020),
        _run("selected32", 32, 0.003, "sigmoid", 9, train=2.0011, valid=2.0011, train_var=0.0020, valid_var=0.0021),
        _run("selected32", 32, 0.003, "sigmoid", 11, train=2.0012, valid=2.0013, train_var=0.0021, valid_var=0.0020),
        _run("best64", 64, 0.0003, "sigmoid", 3, train=2.0003, valid=2.0002, train_var=0.0026, valid_var=0.0027),
        _run("best64", 64, 0.0003, "sigmoid", 9, train=2.0001, valid=2.0003, train_var=0.0025, valid_var=0.0028),
        _run("best64", 64, 0.0003, "sigmoid", 11, train=2.0002, valid=2.0001, train_var=0.0026, valid_var=0.0027),
    ]
    selection = [
        _selection("selected32", selected=True, channels=32, lr=0.003, energy=2.0012, variance=0.0020, iqr=0.004, stderr=0.0010),
        _selection("best64", selected=False, channels=64, lr=0.0003, energy=2.0002, variance=0.0027, iqr=0.001, stderr=0.0011),
    ]

    diagnosis = analyze.build_diagnosis(runs, analyze.rank_selection_rows(selection))

    assert diagnosis["best64"]["config_id"] == "best64"
    assert diagnosis["best64_inside_selection_margin"] is True
    assert abs(diagnosis["width64_median_validation_minus_train"]) < 0.001
    assert "good LR/activation pocket" in diagnosis["conclusion"]
    assert "rather than classic overfit" in diagnosis["conclusion"]


def test_run_analysis_writes_report_and_tables(tmp_path: Path) -> None:
    analyze = load_analyze_module()
    runs_csv = tmp_path / "runs.csv"
    selection_csv = tmp_path / "selection.csv"
    output_dir = tmp_path / "reports"
    _write_csv(
        runs_csv,
        [
            _run("selected32", 32, 0.003, "sigmoid", 3, train=2.0010, valid=2.0012, train_var=0.0020, valid_var=0.0020),
            _run("best64", 64, 0.0003, "sigmoid", 3, train=2.0003, valid=2.0002, train_var=0.0026, valid_var=0.0027),
        ],
    )
    _write_csv(
        selection_csv,
        [
            _selection("selected32", selected=True, channels=32, lr=0.003, energy=2.0012, variance=0.0020, iqr=0.004, stderr=0.0010),
            _selection("best64", selected=False, channels=64, lr=0.0003, energy=2.0002, variance=0.0027, iqr=0.001, stderr=0.0011),
        ],
    )

    result = analyze.run_analysis(runs_csv=runs_csv, selection_csv=selection_csv, output_dir=output_dir)

    assert Path(result["report"]).is_file()
    assert (output_dir / "tables" / "channel64_candidates.csv").is_file()
    assert "64 channels did have a good LR" in Path(result["report"]).read_text(encoding="utf-8")


def _run(
    config_id: str,
    channels: int,
    lr: float,
    gate: str,
    seed: int,
    *,
    train: float,
    valid: float,
    train_var: float,
    valid_var: float,
) -> dict[str, str]:
    return {
        "config_id": config_id,
        "runtime.seed": str(seed),
        "optimizer_params.lr": str(lr),
        "model_params.channels": str(channels),
        "model_params.layers": "1",
        "model_params.gate_activation": gate,
        "status": "completed",
        "train/energy": str(train),
        "validation/energy": str(valid),
        "train/energy_variance": str(train_var),
        "validation/energy_variance": str(valid_var),
        "train/sampler/acceptance_rate": "0.70",
        "validation/sampler/acceptance_rate": "0.70",
    }


def _selection(
    config_id: str,
    *,
    selected: bool,
    channels: int,
    lr: float,
    energy: float,
    variance: float,
    iqr: float,
    stderr: float,
) -> dict[str, str]:
    return {
        "selected": "true" if selected else "false",
        "config_id": config_id,
        "optimizer_params.lr": str(lr),
        "model_params.channels": str(channels),
        "model_params.layers": "1",
        "model_params.gate_activation": "sigmoid",
        "median validation/energy": str(energy),
        "median_energy_variance": str(variance),
        "energy_iqr": str(iqr),
        "median_energy_stderr": str(stderr),
        "n_failed": "0",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
