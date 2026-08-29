"""Run and analyse a single-model, multi-GPU scaling arm.

The probe intentionally launches one command, rather than one independent
process per GPU.  Its input batch is therefore the total scientific batch and
does not change with the requested GPU count.  It timestamps every emitted log
line at microsecond precision, so timing never relies on filesystem mtimes.

The command is backend-owned: callers supply its argument vector after ``--``
and provide regular expressions describing the backend's device, batch,
energy, and step log lines.  This keeps the harness usable for a new backend
without importing that backend on a login node or in the test suite.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence


SCHEMA = "nnqmc-scaling-probe/v1"
LOG_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
WARMUP_CUTS = tuple(range(100, 401, 50))
DEFAULT_DEVICE_REGEX = r"Starting QMC with (?P<devices>\d+) XLA devices"
DEFAULT_STEP_REGEX = r"(?i)\bstep\s*(?P<step>\d+)\b"


class ScalingProbeError(ValueError):
    """Raised when a result cannot support a correctness or timing claim."""


@dataclass(frozen=True)
class Interval:
    """One elapsed interval inferred from two timestamped progress lines."""

    start_step: int
    end_step: int
    seconds: float
    seconds_per_step: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible interval."""

        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "seconds": self.seconds,
            "seconds_per_step": self.seconds_per_step,
        }


def _utc_now() -> str:
    """Return a UTC timestamp with microseconds for the wrapper log."""

    return datetime.now(timezone.utc).strftime(LOG_TIMESTAMP_FORMAT)


def _compile(pattern: str, label: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ScalingProbeError(f"invalid {label} regex: {exc}") from exc


def _require_group(pattern: re.Pattern[str], group: str, label: str) -> None:
    if group not in pattern.groupindex:
        raise ScalingProbeError(f"{label} regex must define named group {group!r}")


def _parse_wrapper_line(line: str) -> tuple[datetime, str] | None:
    """Decode one timestamped wrapper log line, or ignore unrelated lines."""

    timestamp, separator, message = line.rstrip("\n").partition("\t")
    if not separator:
        return None
    try:
        return datetime.strptime(timestamp, LOG_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc), message
    except ValueError:
        return None


def _find_matches(
    lines: Iterable[tuple[datetime, str]], pattern: re.Pattern[str]
) -> list[tuple[datetime, re.Match[str]]]:
    return [(timestamp, match) for timestamp, message in lines if (match := pattern.search(message))]


def _intervals(progress: Sequence[tuple[datetime, re.Match[str]]]) -> list[Interval]:
    """Build consecutive, strictly increasing step intervals."""

    points = [(int(match.group("step")), timestamp) for timestamp, match in progress]
    result: list[Interval] = []
    for (first_step, first_time), (second_step, second_time) in zip(points, points[1:]):
        if second_step <= first_step:
            continue
        seconds = (second_time - first_time).total_seconds()
        if seconds <= 0:
            continue
        result.append(
            Interval(
                start_step=first_step,
                end_step=second_step,
                seconds=seconds,
                seconds_per_step=seconds / (second_step - first_step),
            )
        )
    return result


def _fit_seconds_per_step(progress: Sequence[tuple[datetime, re.Match[str]]], cut: int) -> float | None:
    """Fit elapsed seconds against step after ``cut`` with an intercept."""

    points = [(int(match.group("step")), timestamp.timestamp()) for timestamp, match in progress]
    points = [(step, stamp) for step, stamp in points if step >= cut]
    if len(points) < 2:
        return None
    mean_step = statistics.fmean(step for step, _ in points)
    mean_time = statistics.fmean(stamp for _, stamp in points)
    denominator = sum((step - mean_step) ** 2 for step, _ in points)
    if denominator == 0:
        return None
    return sum((step - mean_step) * (stamp - mean_time) for step, stamp in points) / denominator


def _rate_summary(intervals: Sequence[Interval], cut: int) -> dict[str, float | int | None]:
    rates = [item.seconds_per_step for item in intervals if item.start_step >= cut]
    if not rates:
        return {"interval_count": 0, "min_seconds_per_step": None, "max_seconds_per_step": None, "median_seconds_per_step": None}
    return {
        "interval_count": len(rates),
        "min_seconds_per_step": min(rates),
        "max_seconds_per_step": max(rates),
        "median_seconds_per_step": statistics.median(rates),
    }


def analyse_log(
    log_path: str | Path,
    *,
    expected_devices: int,
    device_regex: str = DEFAULT_DEVICE_REGEX,
    batch_regex: str,
    energy_regex: str,
    step_regex: str = DEFAULT_STEP_REGEX,
) -> dict[str, Any]:
    """Extract device, batch, energy, and timing evidence from one wrapper log.

    ``batch_regex`` needs a named ``batch`` group.  ``energy_regex`` needs
    named ``energy`` and ``error`` groups, and the last matching line is used.
    ``step_regex`` needs named ``step``.  These requirements make all reported
    scientific quantities evidence from the backend's own output.
    """

    if expected_devices <= 0:
        raise ScalingProbeError("expected device count must be positive")
    patterns = {
        "device": _compile(device_regex, "device"),
        "batch": _compile(batch_regex, "batch"),
        "energy": _compile(energy_regex, "energy"),
        "step": _compile(step_regex, "step"),
    }
    _require_group(patterns["device"], "devices", "device")
    _require_group(patterns["batch"], "batch", "batch")
    _require_group(patterns["energy"], "energy", "energy")
    _require_group(patterns["energy"], "error", "energy")
    _require_group(patterns["step"], "step", "step")
    path = Path(log_path)
    try:
        parsed = [item for line in path.read_text(encoding="utf-8").splitlines() if (item := _parse_wrapper_line(line))]
    except OSError as exc:
        raise ScalingProbeError(f"cannot read wrapper log {path}: {exc}") from exc

    device_matches = _find_matches(parsed, patterns["device"])
    observed_devices = int(device_matches[-1][1].group("devices")) if device_matches else None
    batch_matches = _find_matches(parsed, patterns["batch"])
    observed_batch = int(batch_matches[-1][1].group("batch")) if batch_matches else None
    energy_matches = _find_matches(parsed, patterns["energy"])
    progress = _find_matches(parsed, patterns["step"])
    intervals = _intervals(progress)
    errors: list[str] = []
    if observed_devices != expected_devices:
        errors.append(f"process reported {observed_devices!r} devices, expected {expected_devices}")
    if observed_batch is None:
        errors.append("per-device batch was not observed in the backend log")
    if not energy_matches:
        errors.append("energy and statistical error were not observed in the backend log")
    if len(progress) < 2 or not intervals:
        errors.append("fewer than two increasing timestamped step observations")

    energy: dict[str, float | str] | None = None
    if energy_matches:
        timestamp, match = energy_matches[-1]
        energy = {
            "hartree": float(match.group("energy")),
            "stderr": float(match.group("error")),
            "observed_at": timestamp.strftime(LOG_TIMESTAMP_FORMAT),
        }
        if energy["stderr"] < 0:
            errors.append("energy statistical error is negative")

    warmup: dict[str, Any] = {}
    for cut in WARMUP_CUTS:
        warmup[str(cut)] = {
            "fitted_seconds_per_step": _fit_seconds_per_step(progress, cut),
            **_rate_summary(intervals, cut),
        }
    return {
        "log": str(path),
        "observed_devices": observed_devices,
        "observed_per_device_batch": observed_batch,
        "energy": energy,
        "intervals": [item.to_dict() for item in intervals],
        "warmup_cuts": warmup,
        "status": "passed" if not errors else "failed",
        "errors": errors,
    }


def run_arm(
    *,
    output: str | Path,
    log_path: str | Path,
    backend: str,
    ansatz: str,
    system: str,
    steps: int,
    gpu_count: int,
    total_batch_size: int,
    command: Sequence[str],
    device_regex: str,
    batch_regex: str,
    energy_regex: str,
    step_regex: str,
    visible_devices: str | None = None,
) -> dict[str, Any]:
    """Run one arm, timestamp its log, and write a structured result.

    ``visible_devices`` is recorded as a launch condition only.  Passing
    correctness requires a matching process-reported count in the log.
    """

    if not backend or not ansatz or not system:
        raise ScalingProbeError("backend, ansatz, and system must be non-empty")
    if steps <= 0 or gpu_count <= 0 or total_batch_size <= 0:
        raise ScalingProbeError("steps, GPU count, and total batch size must be positive")
    if total_batch_size % gpu_count:
        raise ScalingProbeError("total batch size must divide evenly across the requested GPU arm")
    if not command:
        raise ScalingProbeError("arm command is required")

    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # A backend writes progress to a pipe here, so retain each progress line as
    # it arrives instead of accepting a process-exit burst of buffered output.
    environment["PYTHONUNBUFFERED"] = "1"
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    process_exit_code: int | None = None
    with log.open("x", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()}\tSCALING_PROBE command={json.dumps(list(command))}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(f"{_utc_now()}\t{line.rstrip()}\n")
            handle.flush()
        process_exit_code = process.wait()
        handle.write(f"{_utc_now()}\tSCALING_PROBE exit_code={process_exit_code}\n")

    analysis = analyse_log(
        log,
        expected_devices=gpu_count,
        device_regex=device_regex,
        batch_regex=batch_regex,
        energy_regex=energy_regex,
        step_regex=step_regex,
    )
    if process_exit_code is not None and process_exit_code != 0:
        analysis["errors"].append(f"backend command exited {process_exit_code}")
        analysis["status"] = "failed"
    result = {
        "schema": SCHEMA,
        "arm": {
            "backend": backend,
            "ansatz": ansatz,
            "system": system,
            "steps": steps,
            "gpu_count": gpu_count,
            "total_batch_size": total_batch_size,
            "visible_devices": visible_devices,
        },
        "command": list(command),
        "process_exit_code": process_exit_code,
        **analysis,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def reanalyse_arm(
    source_result: str | Path,
    *,
    output: str | Path,
    device_regex: str,
    batch_regex: str,
    energy_regex: str,
    step_regex: str,
) -> dict[str, Any]:
    """Re-extract a completed arm from its immutable timestamped wrapper log.

    Backends may emit more than one line that resembles a progress record.
    This function preserves the raw arm JSON and log, while producing a new
    result that names the source and uses the caller's exact line contract.
    """

    source_path = Path(source_result)
    source = _load_result(source_path)
    arm = source.get("arm")
    if not isinstance(arm, dict):
        raise ScalingProbeError(f"source result {source_path} has no arm metadata")
    try:
        analysis = analyse_log(
            source["log"],
            expected_devices=int(arm["gpu_count"]),
            device_regex=device_regex,
            batch_regex=batch_regex,
            energy_regex=energy_regex,
            step_regex=step_regex,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScalingProbeError(f"source result {source_path} is missing valid arm evidence") from exc
    process_exit_code = source.get("process_exit_code")
    if process_exit_code != 0:
        analysis["errors"].append(f"backend command exited {process_exit_code}")
        analysis["status"] = "failed"
    result = {
        "schema": SCHEMA,
        "arm": arm,
        "command": source.get("command", []),
        "process_exit_code": process_exit_code,
        "reanalysed_from": str(source_path),
        **analysis,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load_result(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScalingProbeError(f"cannot read arm result {path}: {exc}") from exc
    if payload.get("schema") != SCHEMA:
        raise ScalingProbeError(f"unsupported scaling result schema in {path}")
    return payload


def summarize_ladder(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Compare each arm with its matching 1-GPU energy and timing arm."""

    records = [_load_result(path) for path in paths]
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        arm = record["arm"]
        if arm["gpu_count"] == 1:
            key = (arm["backend"], arm["ansatz"], arm["system"], arm["steps"], arm["total_batch_size"])
            baselines[key] = record
    results: list[dict[str, Any]] = []
    for record in records:
        arm = record["arm"]
        key = (arm["backend"], arm["ansatz"], arm["system"], arm["steps"], arm["total_batch_size"])
        baseline = baselines.get(key)
        comparison: dict[str, Any] = {"result": str(record.get("log", "")), "arm": arm}
        if baseline is None:
            comparison.update({"correctness": "unassessed", "scaling": "unassessed", "reason": "matching 1-GPU arm is absent"})
            results.append(comparison)
            continue
        energy = record.get("energy")
        baseline_energy = baseline.get("energy")
        if record.get("status") != "passed" or baseline.get("status") != "passed" or not energy or not baseline_energy:
            comparison.update({"correctness": "failed", "scaling": "unassessed", "reason": "arm or baseline lacks valid evidence"})
            results.append(comparison)
            continue
        delta = abs(float(energy["hartree"]) - float(baseline_energy["hartree"]))
        combined_error = math.hypot(float(energy["stderr"]), float(baseline_energy["stderr"]))
        correctness = "passed" if delta <= combined_error else "failed"
        comparison.update({"correctness": correctness, "energy_delta_hartree": delta, "combined_stderr_hartree": combined_error})
        if correctness == "failed":
            comparison.update({"scaling": "unassessed", "reason": "correctness gate failed"})
            results.append(comparison)
            continue
        if arm["gpu_count"] == 1:
            comparison.update({"scaling": "baseline", "efficiency": 1.0})
            results.append(comparison)
            continue
        baseline_rate = baseline["warmup_cuts"]["100"]["fitted_seconds_per_step"]
        arm_rate = record["warmup_cuts"]["100"]["fitted_seconds_per_step"]
        if baseline_rate is None or arm_rate is None or arm_rate <= 0:
            comparison.update({"scaling": "unassessed", "reason": "insufficient post-warmup timing observations"})
        else:
            comparison.update({"scaling": "measured", "efficiency": baseline_rate / (arm["gpu_count"] * arm_rate)})
        results.append(comparison)
    return {"schema": SCHEMA, "ladder": results}


def correctness_gate_passed(summary: dict[str, Any]) -> bool:
    """Return whether every supplied ladder result clears scientific correctness.

    A caller supplies a matching 1-GPU reference and a prospective higher-GPU
    rung.  Keeping this check in the reusable probe lets a scheduler script
    stop immediately after a wrong arm, before it can spend an allocation on
    timing a numerically invalid ladder.
    """

    ladder = summary.get("ladder")
    if not isinstance(ladder, list) or not ladder:
        raise ScalingProbeError("correctness gate requires a non-empty ladder summary")
    return all(isinstance(entry, dict) and entry.get("correctness") == "passed" for entry in ladder)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    arm = subparsers.add_parser("run-arm", help="run one process and write its structured arm result")
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--log", type=Path, required=True)
    arm.add_argument("--backend", required=True)
    arm.add_argument("--ansatz", required=True)
    arm.add_argument("--system", required=True)
    arm.add_argument("--steps", type=int, required=True)
    arm.add_argument("--gpus", type=int, required=True)
    arm.add_argument("--total-batch-size", type=int, default=4096)
    arm.add_argument("--visible-devices")
    arm.add_argument("--device-regex", default=DEFAULT_DEVICE_REGEX)
    arm.add_argument("--batch-regex", required=True)
    arm.add_argument("--energy-regex", required=True)
    arm.add_argument("--step-regex", default=DEFAULT_STEP_REGEX)
    arm.add_argument("command", nargs=argparse.REMAINDER)
    summary = subparsers.add_parser("summarize", help="compare structured arms with 1-GPU baselines")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("results", nargs="+", type=Path)
    gate = subparsers.add_parser("gate", help="write a comparison and fail if any arm is scientifically invalid")
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("results", nargs="+", type=Path)
    reanalyse = subparsers.add_parser("reanalyse", help="re-extract a completed arm using a stricter log-line contract")
    reanalyse.add_argument("--source-result", type=Path, required=True)
    reanalyse.add_argument("--output", type=Path, required=True)
    reanalyse.add_argument("--device-regex", default=DEFAULT_DEVICE_REGEX)
    reanalyse.add_argument("--batch-regex", required=True)
    reanalyse.add_argument("--energy-regex", required=True)
    reanalyse.add_argument("--step-regex", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    args = _parser().parse_args(argv)
    try:
        if args.action == "run-arm":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            result = run_arm(
                output=args.output,
                log_path=args.log,
                backend=args.backend,
                ansatz=args.ansatz,
                system=args.system,
                steps=args.steps,
                gpu_count=args.gpus,
                total_batch_size=args.total_batch_size,
                command=command,
                device_regex=args.device_regex,
                batch_regex=args.batch_regex,
                energy_regex=args.energy_regex,
                step_regex=args.step_regex,
                visible_devices=args.visible_devices,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "passed" else 1
        if args.action == "reanalyse":
            result = reanalyse_arm(
                args.source_result,
                output=args.output,
                device_regex=args.device_regex,
                batch_regex=args.batch_regex,
                energy_regex=args.energy_regex,
                step_regex=args.step_regex,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "passed" else 1
        summary = summarize_ladder(args.results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.action == "gate":
            return 0 if correctness_gate_passed(summary) else 1
        return 0
    except ScalingProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
