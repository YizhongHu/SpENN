"""Pair-stability v2/v3 parity helpers.

The parity workflow is intentionally explicit because it submits real scheduler
jobs. It compares existing completed result trees after normalizing path,
study-name, and scheduler-id fields that are expected to differ.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shlex
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
V2_DIR = STUDY_DIR.parent / "pair_stability_v2"
V3_DIR = STUDY_DIR
DEFAULT_ATTEMPT_ID = "parity-v2v3"
DEFAULT_BLIND_SEED = 811
NORMALIZED_FIELDS = {
    "created_at",
    "end_time",
    "host",
    "hostname",
    "launcher_job_id",
    "pid",
    "remaining_min",
    "slurm_job_id",
    "start_time",
    "submitted_at",
    "wall_time_sec",
}
VOLATILE_FIELD_PATTERNS = (
    "start_time",
    "end_time",
    "submitted_at",
    "wall_time_sec",
    "perf/wall_time_sec",
)
GENERIC_METRIC_VALUE_FIELDS = {
    "metric_value",
    "metric_seed_mean",
    "metric_seed_stderr",
    "overall_metric_value",
    "secondary_metric_value",
}
RUNBOOK_WAIT_TRAIN = "# Wait for 01_train jobs from both studies to finish before running validation."
RUNBOOK_WAIT_VALIDATION = "# Wait for 02_validation jobs from both studies to finish before running collection."
RUNBOOK_WAIT_FINAL_TRAIN = "# Wait for 06_final_train jobs from both studies to finish before running final eval."
RUNBOOK_WAIT_FINAL_EVAL = "# Wait for 07_final_eval jobs from both studies to finish before final collection/reporting."
FINAL_COLLECT_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
)


def prepare_v2_config(*, attempt_id: str = DEFAULT_ATTEMPT_ID) -> Path:
    """Write v2-named small configs under the v2 results tree."""

    target = V2_DIR / "results" / "parity_configs" / attempt_id
    target.mkdir(parents=True, exist_ok=True)
    for source_name in ("pair_stability.yaml", "pair_validation.yaml", "smoke.yaml"):
        source = V3_DIR / "configs" / source_name
        text = source.read_text()
        text = text.replace("pair_stability_v3", "pair_stability_v2")
        text = text.replace("experiments/hooke/pair_stability_v3", "experiments/hooke/pair_stability_v2")
        (target / source_name).write_text(text)
    grid = OmegaConf.to_container(OmegaConf.load(V3_DIR / "configs" / "grid.yaml"), resolve=True)
    if not isinstance(grid, dict):
        raise ValueError("v3 grid config did not resolve to a mapping")
    grid["study"] = "pair_stability_v2"
    grid["config"] = str(target / "pair_stability.yaml")
    grid["validation_config"] = str(target / "pair_validation.yaml")
    grid["smoke_config"] = str(target / "smoke.yaml")
    grid["results_root"] = str(V2_DIR / "results")
    grid_path = target / "grid.yaml"
    OmegaConf.save(OmegaConf.create(grid), grid_path)
    return grid_path


def submission_runbook(*, attempt_id: str = DEFAULT_ATTEMPT_ID, blind_seed: int = DEFAULT_BLIND_SEED) -> list[str | list[str]]:
    """Return deterministic v2/v3 parity commands that submit to test partitions."""

    v2_grid = prepare_v2_config(attempt_id=attempt_id)
    commands: list[str | list[str]] = [
        [
            "uv",
            "run",
            "python",
            str(V2_DIR / "plan.py"),
            "--grid",
            str(v2_grid),
            "--results-root",
            str(V2_DIR / "results"),
            "--attempt-id",
            attempt_id,
            "--blind",
            "--blind-seed",
            str(blind_seed),
        ],
        [
            "uv",
            "run",
            "python",
            str(V3_DIR / "plan.py"),
            "--results-root",
            str(V3_DIR / "results"),
            "--attempt-id",
            attempt_id,
            "--blind",
            "--blind-seed",
            str(blind_seed),
        ],
    ]
    for study_dir in (V2_DIR, V3_DIR):
        results_root = str(study_dir / "results")
        commands.append(
            [
                "uv",
                "run",
                "--extra",
                "submitit",
                "python",
                str(study_dir / "train.py"),
                "--results-root",
                results_root,
                "--grid-attempt-id",
                attempt_id,
                "--backend",
                "submitit",
                "--device",
                "cpu",
                "--slurm-cpu-partition",
                "test",
                "--slurm-cpu-timeout-min",
                "30",
                "--slurm-mem-gb",
                "60",
                "--chunk-size",
                "8",
            ]
        )
    commands.append(RUNBOOK_WAIT_TRAIN)
    for study_dir in (V2_DIR, V3_DIR):
        results_root = str(study_dir / "results")
        commands.append(
            [
                "uv",
                "run",
                "--extra",
                "submitit",
                "python",
                str(study_dir / "validate.py"),
                "--results-root",
                results_root,
                "--grid-attempt-id",
                attempt_id,
                "--attempt-id",
                attempt_id,
                "--backend",
                "submitit",
                "--device",
                "cuda",
                "--slurm-partition",
                "gpu_test",
                "--slurm-timeout-min",
                "30",
                "--slurm-mem-gb",
                "60",
                "--chunk-size",
                "8",
            ]
        )
    commands.append(RUNBOOK_WAIT_VALIDATION)
    for study_dir in (V2_DIR, V3_DIR):
        results_root = str(study_dir / "results")
        commands.extend(
            [
                [
                    "uv",
                    "run",
                    "python",
                    str(study_dir / "collect.py"),
                    "--results-root",
                    results_root,
                    "--grid-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                ],
                [
                    "uv",
                    "run",
                    "python",
                    str(study_dir / "select_champions.py"),
                    "--results-root",
                    results_root,
                    "--collection-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                ],
                [
                    "uv",
                    "run",
                    "python",
                    str(study_dir / "final_plan.py"),
                    "--results-root",
                    results_root,
                    "--selection-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                ],
                [
                    "uv",
                    "run",
                    "--extra",
                    "submitit",
                    "python",
                    str(study_dir / "final_train.py"),
                    "--results-root",
                    results_root,
                    "--final-grid-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                    "--backend",
                    "submitit",
                    "--device",
                    "cpu",
                    "--slurm-cpu-partition",
                    "test",
                    "--slurm-cpu-timeout-min",
                    "30",
                    "--slurm-mem-gb",
                    "60",
                    "--chunk-size",
                    "8",
                ],
            ]
        )
    commands.append(RUNBOOK_WAIT_FINAL_TRAIN)
    for study_dir in (V2_DIR, V3_DIR):
        results_root = str(study_dir / "results")
        commands.append(
            [
                "uv",
                "run",
                "--extra",
                "submitit",
                "python",
                str(study_dir / "final_eval.py"),
                "--results-root",
                results_root,
                "--final-grid-attempt-id",
                attempt_id,
                "--final-train-attempt-id",
                attempt_id,
                "--attempt-id",
                attempt_id,
                "--backend",
                "submitit",
                "--device",
                "cuda",
                "--slurm-partition",
                "gpu_test",
                "--slurm-timeout-min",
                "30",
                "--slurm-mem-gb",
                "60",
                "--chunk-size",
                "8",
            ]
        )
    commands.append(RUNBOOK_WAIT_FINAL_EVAL)
    for study_dir in (V2_DIR, V3_DIR):
        results_root = str(study_dir / "results")
        commands.extend(
            [
                [
                    "uv",
                    "run",
                    "python",
                    str(study_dir / "final_collect.py"),
                    "--results-root",
                    results_root,
                    "--final-eval-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                ],
                [
                    "uv",
                    "run",
                    "python",
                    str(study_dir / "final_report.py"),
                    "--results-root",
                    results_root,
                    "--final-collect-attempt-id",
                    attempt_id,
                    "--attempt-id",
                    attempt_id,
                ],
            ]
        )
    return commands


def compare_lineages(
    *,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    v2_attempt_id: str | None = None,
    v3_attempt_id: str | None = None,
) -> list[str]:
    """Return normalized parity differences for the completed v2/v3 lineages.

    ``v2_attempt_id``/``v3_attempt_id`` default to ``attempt_id``. Passing a
    different v2 attempt compares a fresh v3 lineage against a fixed,
    already-completed v2 reference without rerunning v2 (v2 never changes);
    both concrete attempt ids are normalized to one token before comparison.
    """

    v2_attempt = v2_attempt_id or attempt_id
    v3_attempt = v3_attempt_id or attempt_id
    attempt_ids = (v2_attempt, v3_attempt)
    differences: list[str] = []
    for parts in _comparison_artifacts():
        difference = _compare_artifact(parts, v2_attempt=v2_attempt, v3_attempt=v3_attempt)
        if difference is not None:
            differences.append(difference)
    differences.extend(
        _compare_submission_presence(v2_attempt=v2_attempt, v3_attempt=v3_attempt, attempt_ids=attempt_ids)
    )
    differences.extend(_check_v3_stage_plans(v3_attempt))
    return differences


def print_commands(commands: Sequence[str | Sequence[str]]) -> None:
    """Print shell commands for the parity runbook."""

    for command in commands:
        if isinstance(command, str):
            print(command)
        else:
            print(shlex.join([str(part) for part in command]))


def _comparison_artifacts() -> list[tuple[str, ...]]:
    artifacts = [
        ("00_grid", "manifest.json"),
        ("00_grid", "unblind.json"),
        ("03_collect", "summary.csv"),
        ("03_collect", "failures.csv"),
        ("03_collect", "collection_report.json"),
        ("04_select", "champions.csv"),
        ("04_select", "selection_report.json"),
        ("05_final_grid", "source_champions.csv"),
        ("05_final_grid", "final_jobs.csv"),
        ("05_final_grid", "manifest.json"),
        ("08_final_collect", "manifest.yaml"),
        ("09_final_report", "final_report.json"),
        ("09_final_report", "report.md"),
        ("09_final_report", "tables", "energy_components_and_virial_by_winner.csv"),
    ]
    artifacts.extend(("08_final_collect", table) for table in FINAL_COLLECT_TABLES)
    artifacts.extend(("09_final_report", "tables", table) for table in FINAL_COLLECT_TABLES)
    return artifacts


def _compare_artifact(parts: tuple[str, ...], *, v2_attempt: str, v3_attempt: str) -> str | None:
    stage, *rest = parts
    v2_path = V2_DIR / "results" / stage / v2_attempt / Path(*rest)
    v3_path = V3_DIR / "results" / stage / v3_attempt / Path(*rest)
    label = "/".join((stage, *rest))
    attempt_ids = (v2_attempt, v3_attempt)
    if not v2_path.is_file() or not v3_path.is_file():
        return f"missing comparison artifact: {v2_path} / {v3_path}"
    if _normalized_file(v2_path, attempt_ids) != _normalized_file(v3_path, attempt_ids):
        return f"normalized artifact differs: {label}"
    return None


def _compare_submission_presence(
    *, v2_attempt: str, v3_attempt: str, attempt_ids: tuple[str, ...]
) -> list[str]:
    differences = []
    for stage, run_root in (
        ("01_train", "01_train"),
        ("02_validation", "02_validation"),
        ("06_final_train", "06_final_train"),
        ("07_final_eval", "07_final_eval"),
    ):
        v2_submissions = sorted((V2_DIR / "results" / run_root).glob(f"**/{v2_attempt}/submission.json"))
        v3_submissions = sorted((V3_DIR / "results" / run_root).glob(f"**/{v3_attempt}/submission.json"))
        if len(v2_submissions) != len(v3_submissions):
            differences.append(f"{stage}: submission count differs: {len(v2_submissions)} != {len(v3_submissions)}")
            continue
        for left, right in zip(v2_submissions, v3_submissions, strict=True):
            if _normalized_file(left, attempt_ids) != _normalized_file(right, attempt_ids):
                differences.append(f"{stage}: normalized submission differs: {left.name}")
    return differences


def _check_v3_stage_plans(attempt_id: str) -> list[str]:
    differences = []
    for stage in ("01_train", "02_validation"):
        plan_dir = V3_DIR / "results" / stage / "stage_plans" / attempt_id
        for filename in ("stage_manifest.json", "tasks.jsonl", "execution_records.jsonl"):
            if not (plan_dir / filename).is_file():
                differences.append(f"v3 missing toolkit plan artifact: {plan_dir / filename}")
    return differences


def _tokenize_attempt_ids(text: str, attempt_ids: tuple[str, ...]) -> str:
    """Replace concrete attempt ids with one token so v2/v3 ids can differ."""

    for attempt_id in sorted(set(attempt_ids), key=len, reverse=True):
        text = text.replace(attempt_id, "<ATTEMPT_ID>")
    return text


def _normalized_file(path: Path, attempt_ids: tuple[str, ...] = ()) -> Any:
    text = _tokenize_attempt_ids(path.read_text(), attempt_ids)
    if path.suffix == ".csv":
        return _normalize_csv(text)
    if path.suffix == ".jsonl":
        return [_normalize(json.loads(line)) for line in text.splitlines() if line.strip()]
    if path.suffix == ".json":
        return _normalize(json.loads(text))
    return _normalize(text)


def _normalize_csv(text: str) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = []
    for row in reader:
        volatile_metric = _is_volatile_metric(row.get("metric"))
        rows.append(
            {
                key: (
                    "<VOLATILE>"
                    if _is_volatile_field(key) or (volatile_metric and key in GENERIC_METRIC_VALUE_FIELDS)
                    else _normalize(value)
                )
                for key, value in row.items()
            }
        )
    return {"fieldnames": fieldnames, "rows": rows}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        volatile_metrics = {
            key
            for key in ("metric", "overall_metric", "secondary_metric")
            if _is_volatile_metric(value.get(key))
        }
        return {
            key: (
                "<VOLATILE>"
                if _is_volatile_metric_value_field(key, volatile_metrics)
                or _is_volatile_metric_identity_field(key, volatile_metrics)
                else _normalize(item)
            )
            for key, item in sorted(value.items())
            if key not in NORMALIZED_FIELDS and not _is_volatile_field(key)
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        text = value
        replacements = {
            str(V2_DIR / "results"): "<RESULTS_ROOT>",
            str(V3_DIR / "results"): "<RESULTS_ROOT>",
            str(V2_DIR): "<STUDY_DIR>",
            str(V3_DIR): "<STUDY_DIR>",
            "experiments/hooke/pair_stability_v2/results": "<RESULTS_ROOT>",
            "experiments/hooke/pair_stability_v3/results": "<RESULTS_ROOT>",
            "experiments/hooke/pair_stability_v2": "<STUDY_DIR>",
            "experiments/hooke/pair_stability_v3": "<STUDY_DIR>",
            "Pair Stability V2": "Pair Stability <STUDY_VERSION>",
            "Pair Stability V3": "Pair Stability <STUDY_VERSION>",
            "pair_stability_v2": "<STUDY>",
            "pair_stability_v3": "<STUDY>",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"<STUDY_DIR>/results/parity_configs/[^/]+/", "<STUDY_DIR>/configs/", text)
        text = re.sub(r"<RESULTS_ROOT>/parity_configs/[^/]+/", "<STUDY_DIR>/configs/", text)
        text = re.sub(r"parity-v2v3-[0-9]{8}-[0-9]{6}", "parity-v2v3-<ATTEMPT>", text)
        text = re.sub(r"(cpu|cuda):[0-9]+", r"\1:<JOB_ID>", text)
        return text
    return value


def _is_volatile_field(key: str | None) -> bool:
    if key is None:
        return False
    return key in NORMALIZED_FIELDS or any(pattern in key for pattern in VOLATILE_FIELD_PATTERNS)


def _is_volatile_metric(metric: object) -> bool:
    return isinstance(metric, str) and _is_volatile_field(metric)


def _is_volatile_metric_value_field(key: str, volatile_metrics: set[str]) -> bool:
    if key in {"metric_value", "metric_seed_mean", "metric_seed_stderr"}:
        return "metric" in volatile_metrics
    if key == "overall_metric_value":
        return "overall_metric" in volatile_metrics
    if key == "secondary_metric_value":
        return "secondary_metric" in volatile_metrics
    return False


def _is_volatile_metric_identity_field(key: str, volatile_metrics: set[str]) -> bool:
    if key == "overall_champion":
        return "overall_metric" in volatile_metrics
    if key == "secondary_champion":
        return "secondary_metric" in volatile_metrics
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run parity helper commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument(
        "--v2-attempt-id",
        default=None,
        help="Fixed v2 reference attempt for compare (defaults to --attempt-id). "
        "v2 never changes, so a completed v2 lineage can be reused across compares.",
    )
    parser.add_argument(
        "--v3-attempt-id",
        default=None,
        help="v3 attempt for compare (defaults to --attempt-id).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-v2-config")
    subparsers.add_parser("print-runbook")
    subparsers.add_parser("compare")
    args = parser.parse_args(argv)

    if args.command == "prepare-v2-config":
        print(prepare_v2_config(attempt_id=args.attempt_id))
        return 0
    if args.command == "print-runbook":
        print_commands(submission_runbook(attempt_id=args.attempt_id))
        return 0
    differences = compare_lineages(
        attempt_id=args.attempt_id,
        v2_attempt_id=args.v2_attempt_id,
        v3_attempt_id=args.v3_attempt_id,
    )
    if differences:
        for difference in differences:
            print(difference)
        return 1
    v2_attempt = args.v2_attempt_id or args.attempt_id
    v3_attempt = args.v3_attempt_id or args.attempt_id
    if v2_attempt == v3_attempt:
        print(f"pair-stability v2/v3 parity passed for attempt {v2_attempt}")
    else:
        print(f"pair-stability v2/v3 parity passed: v2 {v2_attempt} vs v3 {v3_attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
