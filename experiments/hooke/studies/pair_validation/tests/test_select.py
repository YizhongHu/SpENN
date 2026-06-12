"""Tests for the pair_validation selector script (experiments-owned)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from fake_runs import make_run_dir

import collect


def _full_group(runs_root: Path, *, lr: float, channels: int, energies: dict[int, float], **kwargs):
    """Write one completed run per seed for a single config group."""

    for seed, energy in energies.items():
        make_run_dir(runs_root, seed=seed, lr=lr, channels=channels, energy=energy, **kwargs)


def _collect_table(tmp_path: Path, manifest_path: Path, runs_root: Path):
    manifest = collect.load_manifest(manifest_path)
    rows = [
        collect.collect_run(run_dir, manifest)
        for run_dir in collect.discover_run_dirs([runs_root])
    ]
    output_dir = tmp_path / "results"
    collect.write_outputs(rows, manifest, output_dir)
    return manifest, output_dir / "runs.csv"


def test_selector_computes_median_energy(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 4.0})
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert len(ranked) == 1
    assert ranked[0]["score"] == 3.0  # median of [2.0, 4.0]


def test_failed_seeds_count_as_inf(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    make_run_dir(runs_root, seed=3, lr=1.0e-3, channels=8, energy=2.0)
    make_run_dir(runs_root, seed=9, lr=1.0e-3, channels=8, status="failed", with_validation=False)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    group = ranked[0]
    assert group["n_failed_seeds"] == 1
    # median of [2.0, inf] is inf for an even count under statistics.median's
    # midpoint rule; the group is correctly penalized.
    assert not math.isfinite(group["score"])


def test_missing_seed_counts_as_failed(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    make_run_dir(runs_root, seed=3, lr=1.0e-3, channels=8, energy=2.0)  # seed 9 never ran
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    group = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)[0]

    assert group["n_seeds_expected"] == 2
    assert group["n_failed_seeds"] == 1


def test_ineligible_run_counts_as_failed(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    make_run_dir(runs_root, seed=3, lr=1.0e-3, channels=8, energy=1.0, checks_passed=False)
    make_run_dir(runs_root, seed=9, lr=1.0e-3, channels=8, energy=1.0, finite_fraction=0.9)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    group = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)[0]

    assert group["n_failed_seeds"] == 2


def test_lower_median_energy_wins(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.2})
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 3.0, 9: 3.2})
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.001"
    assert ranked[1]["optimizer_params.lr"] == "0.003"


def test_tie_breakers_applied_deterministically(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    # Same median energy; group A has lower variance and must win.
    _full_group(runs_root, lr=1.0e-3, channels=32, energies={3: 2.0, 9: 2.0}, energy_variance=0.1)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_variance=0.9)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)
    assert ranked[0]["optimizer_params.lr"] == "0.001"

    # Identical variance/spread/wall-time falls through to smaller channels.
    runs_root2 = tmp_path / "runs2"
    _full_group(runs_root2, lr=1.0e-3, channels=32, energies={3: 2.0, 9: 2.0})
    _full_group(runs_root2, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0})
    manifest, runs_csv2 = _collect_table(tmp_path / "second", manifest_path, runs_root2)

    ranked2 = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv2), manifest)
    assert ranked2[0]["model_params.channels"] == "8"
    assert ranked2[0]["decision"]["deciding_tie_breaker"] == "model_params.channels"


def test_selection_margin_value(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    """margin = max(2 * sqrt(stderr_A^2 + stderr_B^2), 0.25 * max(iqr), floor)."""

    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.2}, energy_stderr=0.1)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_stderr=0.1)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)
    margin = select_mod.selection_margin(ranked[0], ranked[1], manifest)

    # stderr term: 2 * sqrt(0.1^2 + 0.1^2); IQR term: 0.25 * 0.1; floor 1e-4.
    assert margin == pytest.approx(2.0 * math.sqrt(0.02))


def test_margin_floor_applies_when_stats_are_tiny(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_stderr=1e-9)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_stderr=1e-9)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert select_mod.selection_margin(ranked[0], ranked[1], manifest) == pytest.approx(1.0e-4)


def test_clear_winner_outside_margin_skips_tie_breakers(
    tmp_path: Path, manifest_path: Path, select_mod
) -> None:
    runs_root = tmp_path / "runs"
    # Tight estimators and a 0.5 Ha gap: a clear win; variance would prefer
    # the loser, proving tie-breakers were never consulted.
    _full_group(
        runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0},
        energy_stderr=0.001, energy_variance=0.9,
    )
    _full_group(
        runs_root, lr=3.0e-3, channels=8, energies={3: 2.5, 9: 2.5},
        energy_stderr=0.001, energy_variance=0.1,
    )
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.001"
    assert ranked[0]["decision"]["tie_set"] == [ranked[0]["config_id"]]
    assert ranked[0]["decision"]["deciding_tie_breaker"] is None


def test_within_margin_uses_variance_tie_breaker(
    tmp_path: Path, manifest_path: Path, select_mod
) -> None:
    runs_root = tmp_path / "runs"
    # B has the (trivially) lower median, but the 1e-4 gap is far inside the
    # stderr margin; A's lower variance must win the tie.
    _full_group(
        runs_root, lr=1.0e-3, channels=8, energies={3: 2.0001, 9: 2.0001},
        energy_stderr=0.01, energy_variance=0.1,
    )
    _full_group(
        runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0},
        energy_stderr=0.01, energy_variance=0.9,
    )
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.001"
    assert len(ranked[0]["decision"]["tie_set"]) == 2
    assert ranked[0]["decision"]["deciding_tie_breaker"] == "validation/energy_variance"


def test_variance_tied_uses_seed_iqr(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    # Equal medians and variances; B's zero seed spread must win.
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 1.9, 9: 2.1})
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0})
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.003"
    assert ranked[0]["decision"]["deciding_tie_breaker"] == "validation_energy_iqr"


def test_iqr_tied_uses_stderr(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_stderr=0.05)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_stderr=0.02)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.003"
    assert ranked[0]["decision"]["deciding_tie_breaker"] == "validation/energy_stderr"


def test_stderr_tied_uses_geometry_warnings(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0})
    # Same energy statistics but one seed samples a near-coalescence tail.
    make_run_dir(runs_root, seed=3, lr=3.0e-3, channels=8, energy=2.0, electron_distance_q01=1.0e-6)
    make_run_dir(runs_root, seed=9, lr=3.0e-3, channels=8, energy=2.0)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.001"
    assert ranked[0]["decision"]["deciding_tie_breaker"] == "geometry_warning_count"


def test_channels_tied_uses_wall_time(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, wall_time=50.0)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, wall_time=5.0)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    ranked = select_mod.rank_groups(select_mod.read_runs_csv(runs_csv), manifest)

    assert ranked[0]["optimizer_params.lr"] == "0.003"
    assert ranked[0]["decision"]["deciding_tie_breaker"] == "runtime/wall_time_sec"


def test_report_documents_margin_and_tie_breakers(
    tmp_path: Path, manifest_path: Path, select_mod
) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_variance=0.1)
    _full_group(runs_root, lr=3.0e-3, channels=8, energies={3: 2.0, 9: 2.0}, energy_variance=0.9)
    _, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)
    output_dir = tmp_path / "selection"

    select_mod.main(
        ["--manifest", str(manifest_path), "--runs", str(runs_csv), "--output-dir", str(output_dir)]
    )

    report = (output_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "selection margin" in report
    assert "within the selection margin" in report
    assert "validation/energy_variance" in report

    with open(output_dir / "selected_config.yaml", encoding="utf-8") as handle:
        selected = yaml.safe_load(handle)
    assert selected["selection"]["deciding_tie_breaker"] == "validation/energy_variance"
    assert len(selected["selection"]["tie_set"]) == 2


def test_selector_writes_outputs(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.1})
    _full_group(runs_root, lr=3.0e-3, channels=32, energies={3: 2.5, 9: 2.6})
    _, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)
    output_dir = tmp_path / "selection"

    exit_code = select_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--runs",
            str(runs_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "selection.csv").is_file()
    report = (output_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "lr=0.001_channels=8_layers=1_gate_activation=silu" in report

    with open(output_dir / "selected_config.yaml", encoding="utf-8") as handle:
        selected = yaml.safe_load(handle)
    assert selected["selected"]["config_id"] == "lr=0.001_channels=8_layers=1_gate_activation=silu"
    assert selected["selected"]["optimizer_params.lr"] == "0.001"
    assert "optimizer_params.lr=0.001" in selected["overrides"]
    assert selected["selection"]["score"] == pytest.approx(2.05)


def test_selector_never_uses_reference_energy(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    """Selection works without any reference energy column or value."""

    runs_root = tmp_path / "runs"
    _full_group(runs_root, lr=1.0e-3, channels=8, energies={3: 2.0, 9: 2.1})
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    rows = select_mod.read_runs_csv(runs_csv)
    assert not any("reference" in column for row in rows for column in row)
    assert not any("energy_error" in column for row in rows for column in row)

    ranked = select_mod.rank_groups(rows, manifest)
    assert math.isfinite(ranked[0]["score"])

    source = (Path(__file__).resolve().parents[1] / "select.py").read_text(encoding="utf-8")
    assert "reference_energy" not in source


def test_report_includes_geometry_and_flags(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    # Suspiciously small electron_distance_q01 on one seed.
    make_run_dir(runs_root, seed=3, lr=1.0e-3, channels=8, energy=2.0, electron_distance_q01=1.0e-6)
    make_run_dir(runs_root, seed=9, lr=1.0e-3, channels=8, energy=2.1)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)
    rows = select_mod.read_runs_csv(runs_csv)

    flags = select_mod.geometry_flags(rows, manifest)
    assert any("electron_distance_q01" in flag and "below" in flag for flag in flags)

    output_dir = tmp_path / "selection"
    select_mod.main(
        ["--manifest", str(manifest_path), "--runs", str(runs_csv), "--output-dir", str(output_dir)]
    )
    report = (output_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "Sampler geometry diagnostics" in report
    assert "radius_q99" in report
    assert "near-coalescence" in report


def test_missing_geometry_is_flagged(tmp_path: Path, manifest_path: Path, select_mod) -> None:
    runs_root = tmp_path / "runs"
    make_run_dir(runs_root, seed=3, lr=1.0e-3, channels=8, energy=2.0, with_geometry=False)
    manifest, runs_csv = _collect_table(tmp_path, manifest_path, runs_root)

    flags = select_mod.geometry_flags(select_mod.read_runs_csv(runs_csv), manifest)

    assert any("radius_q99 missing" in flag for flag in flags)
    # Pair-distance metrics are required here because n_electrons == 2.
    assert any("electron_distance_q01 missing" in flag for flag in flags)
