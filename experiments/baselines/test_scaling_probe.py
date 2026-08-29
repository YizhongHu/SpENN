"""Cluster-free tests for the reusable multi-GPU scaling probe."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from experiments.baselines import scaling_probe
from experiments.baselines.run_deepqmc_scaling_arm import per_device_batch, training_tail
from experiments.baselines.run_ferminet_scaling_arm import summarize_training_tail


DEVICE = r"Starting QMC with (?P<devices>\d+) XLA devices"
BATCH = r"device_batch_size=(?P<batch>\d+)"
ENERGY = r"energy=(?P<energy>-?\d+\.\d+) stderr=(?P<error>\d+\.\d+)"


def _write_log(path: Path, *, devices: int, batch: int, energy: float, error: float, interval: float) -> None:
    lines = [f"2026-08-29T00:00:00.000000Z\tStarting QMC with {devices} XLA devices", f"2026-08-29T00:00:00.000001Z\tdevice_batch_size={batch}"]
    for step in range(0, 501, 100):
        seconds = step * interval
        lines.append(f"2026-08-29T00:00:{seconds:09.6f}Z\tStep {step}")
    lines.append(f"2026-08-29T00:01:00.000000Z\tenergy={energy:.6f} stderr={error:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result(path: Path, log: Path, *, devices: int, energy: float, error: float, interval: float) -> None:
    analysis = scaling_probe.analyse_log(log, expected_devices=devices, device_regex=DEVICE, batch_regex=BATCH, energy_regex=ENERGY)
    path.write_text(
        json.dumps(
            {
                "schema": scaling_probe.SCHEMA,
                "arm": {"backend": "ferminet", "ansatz": "psiformer", "system": "he", "steps": 500, "gpu_count": devices, "total_batch_size": 4096},
                **analysis,
            }
        ),
        encoding="utf-8",
    )


def test_analysis_requires_process_reported_device_count_and_emits_warmup_sweep(tmp_path: Path) -> None:
    log = tmp_path / "arm.log"
    _write_log(log, devices=2, batch=2048, energy=-2.90, error=0.02, interval=0.01)
    result = scaling_probe.analyse_log(log, expected_devices=2, device_regex=DEVICE, batch_regex=BATCH, energy_regex=ENERGY)
    assert result["status"] == "passed"
    assert result["observed_devices"] == 2
    assert result["observed_per_device_batch"] == 2048
    assert set(result["warmup_cuts"]) == {"100", "150", "200", "250", "300", "350", "400"}
    assert result["warmup_cuts"]["100"]["fitted_seconds_per_step"] == 0.01


def test_analysis_rejects_visibility_only_claim(tmp_path: Path) -> None:
    log = tmp_path / "arm.log"
    _write_log(log, devices=1, batch=4096, energy=-2.90, error=0.02, interval=0.01)
    result = scaling_probe.analyse_log(log, expected_devices=4, device_regex=DEVICE, batch_regex=BATCH, energy_regex=ENERGY)
    assert result["status"] == "failed"
    assert "expected 4" in result["errors"][0]


def test_summary_stops_scaling_when_energy_disagrees(tmp_path: Path) -> None:
    one_log, four_log = tmp_path / "one.log", tmp_path / "four.log"
    _write_log(one_log, devices=1, batch=4096, energy=-2.90, error=0.01, interval=0.04)
    _write_log(four_log, devices=4, batch=1024, energy=-2.80, error=0.01, interval=0.012)
    one, four = tmp_path / "one.json", tmp_path / "four.json"
    _result(one, one_log, devices=1, energy=-2.90, error=0.01, interval=0.04)
    _result(four, four_log, devices=4, energy=-2.80, error=0.01, interval=0.012)
    summary = scaling_probe.summarize_ladder([one, four])
    assert summary["ladder"][1]["correctness"] == "failed"
    assert summary["ladder"][1]["scaling"] == "unassessed"
    assert not scaling_probe.correctness_gate_passed(summary)


def test_summary_reports_combined_error_and_efficiency(tmp_path: Path) -> None:
    one_log, two_log = tmp_path / "one.log", tmp_path / "two.log"
    _write_log(one_log, devices=1, batch=4096, energy=-2.90, error=0.02, interval=0.04)
    _write_log(two_log, devices=2, batch=2048, energy=-2.88, error=0.02, interval=0.024)
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    _result(one, one_log, devices=1, energy=-2.90, error=0.02, interval=0.04)
    _result(two, two_log, devices=2, energy=-2.88, error=0.02, interval=0.024)
    summary = scaling_probe.summarize_ladder([one, two])
    assert summary["ladder"][1]["correctness"] == "passed"
    assert summary["ladder"][1]["efficiency"] == pytest.approx(0.04 / (2 * 0.024))
    assert scaling_probe.correctness_gate_passed(summary)


def test_gate_records_marginal_agreement_but_stops_only_above_five_sigma(tmp_path: Path) -> None:
    one_log, marginal_log, failed_log = tmp_path / "one.log", tmp_path / "marginal.log", tmp_path / "failed.log"
    _write_log(one_log, devices=1, batch=4096, energy=-2.90, error=0.01, interval=0.04)
    _write_log(marginal_log, devices=2, batch=2048, energy=-2.84, error=0.01, interval=0.024)
    _write_log(failed_log, devices=4, batch=1024, energy=-2.80, error=0.01, interval=0.012)
    one, marginal, failed = tmp_path / "one.json", tmp_path / "marginal.json", tmp_path / "failed.json"
    _result(one, one_log, devices=1, energy=-2.90, error=0.01, interval=0.04)
    _result(marginal, marginal_log, devices=2, energy=-2.84, error=0.01, interval=0.024)
    _result(failed, failed_log, devices=4, energy=-2.80, error=0.01, interval=0.012)
    marginal_summary = scaling_probe.summarize_ladder([one, marginal])
    assert marginal_summary["ladder"][1]["correctness"] == "marginal"
    assert marginal_summary["ladder"][1]["agreement"] == "marginal"
    assert scaling_probe.correctness_gate_passed(marginal_summary)
    failed_summary = scaling_probe.summarize_ladder([one, failed])
    assert failed_summary["ladder"][1]["correctness"] == "failed"
    assert failed_summary["ladder"][1]["agreement"] == "inconsistent"
    assert not scaling_probe.correctness_gate_passed(failed_summary)


def test_gate_cli_creates_a_missing_result_directory(tmp_path: Path) -> None:
    one_log, two_log = tmp_path / "one.log", tmp_path / "two.log"
    _write_log(one_log, devices=1, batch=4096, energy=-2.90, error=0.02, interval=0.04)
    _write_log(two_log, devices=2, batch=2048, energy=-2.88, error=0.02, interval=0.024)
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    _result(one, one_log, devices=1, energy=-2.90, error=0.02, interval=0.04)
    _result(two, two_log, devices=2, energy=-2.88, error=0.02, interval=0.024)
    output = tmp_path / "missing" / "gate.json"
    assert scaling_probe.main(["gate", "--output", str(output), str(one), str(two)]) == 0
    assert json.loads(output.read_text())["ladder"][1]["correctness"] == "passed"


def test_reanalyse_uses_only_the_requested_progress_lines(tmp_path: Path) -> None:
    log = tmp_path / "arm.log"
    log.write_text(
        "\n".join(
            [
                "2026-08-29T00:00:00.000000Z\tStarting QMC with 1 XLA devices",
                "2026-08-29T00:00:00.000001Z\tdevice_batch_size=4096",
                "2026-08-29T00:00:00.000002Z\tINFO Step 0",
                "2026-08-29T00:00:00.010000Z\tSCALING_PROBE Step 0",
                "2026-08-29T00:00:00.010001Z\tINFO Step 1",
                "2026-08-29T00:00:00.020000Z\tSCALING_PROBE Step 1",
                "2026-08-29T00:00:00.020001Z\tINFO Step 2",
                "2026-08-29T00:00:00.030000Z\tSCALING_PROBE Step 2",
                "2026-08-29T00:00:01.000000Z\tenergy=-2.900000 stderr=0.020000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema": scaling_probe.SCHEMA,
                "arm": {"backend": "ferminet", "ansatz": "ferminet", "system": "he", "steps": 2, "gpu_count": 1, "total_batch_size": 4096},
                "command": ["ferminet"],
                "process_exit_code": 0,
                "log": str(log),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reanalysed" / "result.json"
    result = scaling_probe.reanalyse_arm(
        source,
        output=output,
        device_regex=DEVICE,
        batch_regex=BATCH,
        energy_regex=ENERGY,
        step_regex=r"SCALING_PROBE Step (?P<step>\d+)",
    )
    assert result["status"] == "passed"
    assert result["intervals"][0]["seconds_per_step"] == pytest.approx(0.01)
    assert result["reanalysed_from"] == str(source)


def test_run_arm_writes_microsecond_wrapper_log_and_structured_result(tmp_path: Path) -> None:
    command = [
        "python",
        "-c",
        "print('Starting QMC with 1 XLA devices'); print('device_batch_size=4096'); "
        "[print(f'Step {step}') for step in (0, 100, 200)]; "
        "print('energy=-2.900000 stderr=0.020000')",
    ]
    result = scaling_probe.run_arm(
        output=tmp_path / "arm.json",
        log_path=tmp_path / "arm.log",
        backend="ferminet",
        ansatz="ferminet",
        system="he",
        steps=200,
        gpu_count=1,
        total_batch_size=4096,
        command=command,
        device_regex=DEVICE,
        batch_regex=BATCH,
        energy_regex=ENERGY,
        step_regex=r"Step (?P<step>\d+)",
    )
    assert result["status"] == "passed"
    assert json.loads((tmp_path / "arm.json").read_text())["observed_devices"] == 1
    wrapper_line = (tmp_path / "arm.log").read_text().splitlines()[0]
    assert wrapper_line.startswith("202") and "." in wrapper_line.split("\t", 1)[0]


def test_ferminet_runner_uses_a_blocking_training_tail(tmp_path: Path) -> None:
    stats = tmp_path / "train_stats.csv"
    stats.write_text(
        "step,energy\n" + "\n".join(f"{step},{-2.9 + step * 1.e-5}" for step in range(500)) + "\n",
        encoding="utf-8",
    )
    energy, stderr = summarize_training_tail(tmp_path, first_step=400)
    assert energy == pytest.approx(-2.895505)
    assert stderr > 0


def test_deepqmc_runner_uses_total_batch_and_a_blocking_training_tail() -> None:
    assert per_device_batch(4096, 4) == 1024
    with pytest.raises(ValueError, match="does not divide"):
        per_device_batch(4095, 4)
    energy, stderr = training_tail(
        [-2.9 + step * 1.e-5 for step in range(500)], first_tail_step=400
    )
    assert energy == pytest.approx(-2.895505)
    assert stderr > 0
