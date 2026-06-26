"""Collect Hooke pair validation-scan run outputs into normalized tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from .study_manifest import (
        collect_report_dir,
        phase_run_root,
        phase_study_name,
        phase_study_phase,
        phase_study_version,
        run_kind_for_dir,
    )
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import (
        collect_report_dir,
        phase_run_root,
        phase_study_name,
        phase_study_phase,
        phase_study_version,
        run_kind_for_dir,
    )

REQUIRED_COLUMNS = (
    "run_dir",
    "status",
    "status/current_event",
    "status/exception_type",
    "status/exception_message",
    "study_name",
    "study_version",
    "study_phase",
    "config_id",
    "runtime.seed",
    "optimizer_params.lr",
    "model_params.channels",
    "model_params.layers",
    "model_params.gate_activation",
    "system.n_particles",
    "system.n_electrons",
    "system.spin.n_up",
    "system.spin.n_down",
    "validation/energy",
    "validation/energy_stderr",
    "validation/energy_variance",
    "validation/local_energy_finite_fraction",
    "validation/sampler/acceptance_rate",
    "validation/sampler/n_walkers",
    "validation/sampler/burn_in",
    "validation/sampler/n_steps",
    "validation/sampler/proposal_scale",
    "validation/sampler/seed",
    "validation/sampler/n_electrons",
    "validation/sampler/radius_mean",
    "validation/sampler/radius_q99",
    "validation/sampler/radius_max",
    "validation/sampler/electron_distance_q01",
    "validation/sampler/electron_distance_min",
    "validation/sampler/position_rms",
    "checks/data_integrity/passed",
    "checks/gradient/passed",
    "checks/equivariance/full_model/passed",
    "runtime/wall_time_sec",
    "git/sha",
    "wandb/run_id",
    "checkpoint/latest_path",
)

CONFIG_FIELDS = (
    "runtime.seed",
    "optimizer_params.lr",
    "model_params.channels",
    "model_params.layers",
    "model_params.gate_activation",
    "system.n_particles",
    "system.n_electrons",
    "system.spin.n_up",
    "system.spin.n_down",
)

REQUIRED_VALIDATION_METRIC = "validation/energy"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector CLI."""

    args = _parse_args(argv)
    collect_runs(
        manifest_path=args.manifest,
        run_root=args.run_root,
        output_dir=args.output_dir,
        run_dirs=args.run_dirs,
        allow_other_studies=args.allow_other_studies,
        include_smoke=args.include_smoke,
        phase=args.phase,
    )
    return 0


def collect_runs(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    run_dirs: Sequence[str | Path] | None = None,
    allow_other_studies: bool = False,
    include_smoke: bool = False,
    phase: str = "validation_train",
) -> list[dict[str, Any]]:
    """Collect validation run directories and write ``runs.csv``/``runs.jsonl``.

    Parameters
    ----------
    manifest_path
        Study manifest declaring the authoritative study name.
    run_root
        Root searched for run directories when ``run_dirs`` is not supplied.
    output_dir
        Directory receiving normalized tables.
    run_dirs
        Optional explicit run directories.
    allow_other_studies
        If ``False``, include only runs whose resolved config has
        ``study.name`` equal to the manifest phase's study name.
    include_smoke
        Include run directories whose first staged run-id component is
        ``smoke``. Defaults to full runs only.
    phase
        Manifest phase used for the default study-name filter.
    """

    manifest = _load_yaml(manifest_path)
    study_name = phase_study_name(manifest, phase)
    study_version = phase_study_version(manifest, phase)
    study_phase = phase_study_phase(manifest, phase)
    root = Path(run_root) if run_root is not None else Path(phase_run_root(manifest, phase))
    candidates = [Path(path) for path in run_dirs] if run_dirs else _discover_run_dirs(root)
    if not include_smoke:
        candidates = [path for path in candidates if run_kind_for_dir(root, path) != "smoke"]

    rows: list[dict[str, Any]] = []
    for run_dir in candidates:
        row = collect_run_dir(run_dir)
        if not allow_other_studies and (
            row.get("study_name") != study_name
            or row.get("study_version") != study_version
            or (row.get("study_phase") not in (None, "", study_phase))
        ):
            continue
        rows.append(row)

    rows.sort(key=lambda item: str(item.get("run_dir", "")))
    output = Path(output_dir) if output_dir is not None else Path(collect_report_dir(manifest))
    _write_outputs(rows, output)
    return rows


def collect_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """Return one normalized row for a local run directory."""

    run_path = Path(run_dir)
    cfg = _load_yaml_if_present(run_path / "resolved_config.yaml")
    metrics = _read_metrics(run_path)
    metadata = _load_json_if_present(run_path / "metadata.json")
    status_artifact = _load_json_if_present(run_path / "status.json")
    run_start = _load_json_if_present(run_path / "run_start.json")

    row: dict[str, Any] = {column: None for column in REQUIRED_COLUMNS}
    row["run_dir"] = str(run_path)
    row["study_name"] = _select(cfg, "study.name")
    row["study_version"] = _select(cfg, "study.version")
    row["study_phase"] = _select(cfg, "study.phase")
    row["config_id"] = _select(cfg, "study.config_id")

    for field in CONFIG_FIELDS:
        row[field] = _select(cfg, field)
    if row["system.n_electrons"] is None:
        row["system.n_electrons"] = row["system.n_particles"]

    row.update(metrics)
    row["status"] = _classify_status(run_path, metrics, metadata, status_artifact)
    row.update(_status_debug_fields(status_artifact, metadata))
    row["git/sha"] = _select(run_start, "git.sha") or metadata.get("git_commit")
    row["wandb/run_id"] = _select(metadata, "wandb.run_id") or _select(metadata, "wandb_run_id")
    latest_path = run_path / "checkpoints" / "latest.json"
    if latest_path.exists():
        row["checkpoint/latest_path"] = str(latest_path)

    if not row["config_id"]:
        row["config_id"] = _default_config_id(row)
    return row


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--phase", default="validation_train")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", nargs="*", type=Path)
    parser.add_argument("--allow-other-studies", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args(argv)


def _discover_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    run_dirs = []
    for path in run_root.rglob("resolved_config.yaml"):
        if "checkpoints" in path.relative_to(run_root).parts:
            continue
        run_dirs.append(path.parent)
    return sorted(run_dirs)


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                namespace = str(row.get("namespace") or "").strip("/")
                key = str(row.get("key") or "").strip("/")
                if namespace and key:
                    metrics[f"{namespace}/{key}"] = parse_scalar(row.get("value"))
    if jsonl_path.is_file() and jsonl_path.stat().st_size > 0:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                namespace = str(record.get("namespace") or "").strip("/")
                values = record.get("metrics") or {}
                if not namespace or not isinstance(values, Mapping):
                    continue
                for key, value in values.items():
                    metrics[f"{namespace}/{key}"] = value
    return metrics


def parse_scalar(value: Any) -> Any:
    """Parse a CSV scalar while preserving empty values as ``None``."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _classify_status(
    run_dir: Path,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status_artifact: Mapping[str, Any],
) -> str:
    status_values = {
        str(status_artifact.get("status", "")).lower(),
        str(metadata.get("status", "")).lower(),
    }
    if (run_dir / "error.json").exists() or status_values.intersection({"failed", "error", "exception"}):
        return "failed"
    if not _has_metric_file(run_dir):
        return "missing_metrics"
    if metrics.get(REQUIRED_VALIDATION_METRIC) is None:
        return "missing_validation"
    if "completed" in status_values:
        return "completed"
    return "incomplete"


def _status_debug_fields(status_artifact: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return failure/debug fields copied from durable status artifacts."""

    return {
        "status/current_event": status_artifact.get("current_event"),
        "status/exception_type": status_artifact.get("exception_type") or metadata.get("exception_type"),
        "status/exception_message": status_artifact.get("exception_message") or metadata.get("exception_message"),
    }


def _has_metric_file(run_dir: Path) -> bool:
    return any((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in ("metrics.csv", "metrics.jsonl"))


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted({key for row in rows for key in row if key not in REQUIRED_COLUMNS})
    columns = [*REQUIRED_COLUMNS, *extra_columns]
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    with (output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_yaml(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    return data if isinstance(data, dict) else {}


def _load_yaml_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_yaml(path)


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _default_config_id(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "optimizer_params.lr",
        "model_params.channels",
        "model_params.layers",
        "model_params.gate_activation",
    ):
        value = row.get(key)
        if value is not None:
            parts.append(f"{key.split('.')[-1]}{_slug(value)}")
    return "config_" + "_".join(parts) if parts else ""


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


__all__ = [
    "REQUIRED_COLUMNS",
    "collect_run_dir",
    "collect_runs",
    "main",
    "parse_scalar",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
