"""Manifest and job helpers for the Hooke pair validation study."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

DEFAULT_MANIFEST = Path(__file__).with_name("manifest.yaml")
TRAIN_PHASES = {"validation_train", "final_train"}
EVAL_PHASES = {"final_eval"}
COLLECT_STAGE = "02_collect"
SELECT_STAGE = "03_select"
FINAL_TRAIN_STAGE = "04_final_train"
FINAL_EVAL_STAGE = "05_final_eval"
SMOKE_PHASE_TARGETS = {
    "smoke_train": "validation_train",
    "smoke_eval": "final_eval",
}
SEED_KEY = "runtime.seed"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML as resolved plain Python data."""

    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write YAML from structured data."""

    OmegaConf.save(config=OmegaConf.create(dict(data)), f=path, resolve=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be a mapping: {path}")
            rows.append(row)
    return rows


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write JSONL rows with finite-safe scalars."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def phase_for_launch(phase: str, *, target_phase: str | None = None) -> tuple[str, bool]:
    """Return manifest phase name and whether the launch is a smoke."""

    if phase in SMOKE_PHASE_TARGETS:
        return target_phase or SMOKE_PHASE_TARGETS[phase], True
    return phase, False


def phase_config(manifest: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    """Return a manifest phase block."""

    phases = manifest.get("phases")
    if not isinstance(phases, Mapping) or phase not in phases:
        raise ValueError(f"manifest phase {phase!r} is not defined")
    block = phases[phase]
    if not isinstance(block, Mapping):
        raise ValueError(f"manifest phase {phase!r} must be a mapping")
    return block


def phase_base_config(manifest: Mapping[str, Any], phase: str) -> str:
    """Return the concrete base config path for a phase."""

    block = phase_config(manifest, phase)
    key = str(block.get("base_config") or "")
    configs = manifest.get("configs") if isinstance(manifest.get("configs"), Mapping) else {}
    path = configs.get(key) if key in configs else key
    if not path:
        raise ValueError(f"phase {phase!r} does not declare base_config")
    return str(path)


def phase_run_root(manifest: Mapping[str, Any], phase: str) -> str:
    """Return phase run root."""

    block = phase_config(manifest, phase)
    value = block.get("run_root")
    if value:
        return str(value)
    raise ValueError(f"phase {phase!r} must declare run_root in the manifest")


def phase_slurm_log_dir(manifest: Mapping[str, Any], phase: str) -> str:
    """Return phase Slurm log directory."""

    block = phase_config(manifest, phase)
    value = block.get("slurm_log_dir")
    if value:
        return str(value)
    raise ValueError(f"phase {phase!r} must declare slurm_log_dir in the manifest")


def report_root(manifest: Mapping[str, Any]) -> str:
    """Return manifest report root."""

    value = _select(manifest, "paths.report_root")
    if not value:
        raise ValueError("manifest paths.report_root is required")
    return str(value)


def collect_report_dir(manifest: Mapping[str, Any]) -> str:
    """Return the validation-collection artifact directory."""

    return str(Path(report_root(manifest)) / COLLECT_STAGE)


def select_report_dir(manifest: Mapping[str, Any]) -> str:
    """Return the validation-selection artifact directory."""

    return str(Path(report_root(manifest)) / SELECT_STAGE)


def final_train_report_dir(manifest: Mapping[str, Any]) -> str:
    """Return the final-train artifact directory."""

    return str(Path(report_root(manifest)) / FINAL_TRAIN_STAGE)


def final_train_plan_dir(manifest: Mapping[str, Any]) -> str:
    """Return the final-train plan directory."""

    return str(Path(final_train_report_dir(manifest)) / "plans")


def final_eval_report_dir(manifest: Mapping[str, Any]) -> str:
    """Return the final-eval artifact directory."""

    return str(Path(report_root(manifest)) / FINAL_EVAL_STAGE)


def final_eval_plan_dir(manifest: Mapping[str, Any]) -> str:
    """Return the final-eval plan directory."""

    return str(Path(final_eval_report_dir(manifest)) / "plans")


def phase_study_name(manifest: Mapping[str, Any], phase: str) -> str:
    """Return phase-local study name."""

    block = phase_config(manifest, phase)
    value = _fixed_override(block, "study.name") or block.get("study_name") or _select(manifest, "study.name")
    if not value:
        raise ValueError(f"phase {phase!r} needs overrides.fixed.study.name or study.name")
    return str(value)


def phase_study_version(manifest: Mapping[str, Any], phase: str) -> str:
    """Return phase-local study version."""

    block = phase_config(manifest, phase)
    value = _fixed_override(block, "study.version") or _select(manifest, "study.version")
    if not value:
        raise ValueError(f"phase {phase!r} needs overrides.fixed.study.version or study.version")
    return str(value)


def phase_study_phase(manifest: Mapping[str, Any], phase: str) -> str:
    """Return recorded study phase label."""

    block = phase_config(manifest, phase)
    return str(_fixed_override(block, "study.phase") or phase)


def phase_provenance_overrides(manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Return explicit study provenance overrides for a phase."""

    return {
        "study.name": phase_study_name(manifest, phase),
        "study.version": phase_study_version(manifest, phase),
        "study.phase": phase_study_phase(manifest, phase),
    }


def phase_sweep(manifest: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    """Return phase sweep axes."""

    overrides = phase_config(manifest, phase).get("overrides")
    if not isinstance(overrides, Mapping):
        return {}
    sweep = overrides.get("sweep")
    return sweep if isinstance(sweep, Mapping) else {}


def expected_validation_seeds(manifest: Mapping[str, Any]) -> list[str]:
    """Return expected validation replicate seeds as strings."""

    source_phase = str(_select(manifest, "selection.source_phase") or "validation_train")
    seeds = phase_sweep(manifest, source_phase).get(SEED_KEY, [])
    if not seeds and isinstance(manifest.get("grid"), Mapping):
        seeds = manifest["grid"].get(str(manifest.get("seed_key") or SEED_KEY), [])
    return [_key_text(seed) for seed in seeds]


def profile_device(manifest: Mapping[str, Any], profile: str) -> str:
    """Return runtime device for a profile."""

    block = _profile_block(manifest, profile)
    return str(block.get("device") or ("cuda" if profile == "gpu" else "cpu"))


def profile_environment(manifest: Mapping[str, Any], profile: str) -> str:
    """Return UV environment path for a profile."""

    block = _profile_block(manifest, profile)
    return str(block.get("uv_environment") or (".venv-gpu" if profile == "gpu" else ".venv"))


def profile_extras(manifest: Mapping[str, Any], profile: str) -> list[str]:
    """Return UV extras for a profile."""

    block = _profile_block(manifest, profile)
    extras = block.get("uv_extras")
    if isinstance(extras, Sequence) and not isinstance(extras, (str, bytes)):
        return [str(item) for item in extras]
    return ["cu126" if profile == "gpu" else "cpu", "submitit"]


def profile_slurm(manifest: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    """Return Slurm resource settings for a manifest profile."""

    slurm = _profile_block(manifest, profile).get("slurm")
    return slurm if isinstance(slurm, Mapping) else {}


def cartesian_phase_jobs(
    manifest: Mapping[str, Any],
    phase: str,
    *,
    selected: Mapping[str, Any] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Return concrete job rows for a cartesian train phase."""

    block = phase_config(manifest, phase)
    if str(block.get("mode")) != "cartesian":
        raise ValueError(f"phase {phase!r} is not cartesian")
    axes = [(str(key), _as_sequence(value, key=str(key))) for key, value in phase_sweep(manifest, phase).items()]
    if not axes:
        raise ValueError(f"phase {phase!r} needs overrides.sweep axes")

    jobs: list[dict[str, Any]] = []
    reversed_axes = list(reversed(axes))
    for reversed_values in itertools.product(*(values for _, values in reversed_axes)):
        values = list(reversed(reversed_values))
        sweep_values = {key: value for (key, _), value in zip(axes, values, strict=True)}
        jobs.append(_cartesian_job(manifest, phase, sweep_values, selected=selected, smoke=False))

    if smoke:
        jobs = _select_smoke_jobs(jobs, block)
        jobs = [_apply_smoke_overlay(job, block) for job in jobs]
    return jobs


def command_for_job(
    job: Mapping[str, Any],
    *,
    device: str,
    python: str = "python",
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return the direct SpENN command for one job row."""

    overrides = [str(item) for item in job.get("overrides", [])]
    if not any(item.startswith("runtime.device=") for item in overrides):
        overrides.append(f"runtime.device={device}")
    run_py = "run.py"
    base_config = Path(str(job["base_config"]))
    if repo_root is not None:
        root = Path(repo_root)
        run_py = str(root / run_py)
        if not base_config.is_absolute():
            base_config = root / base_config
    return [python, "-u", run_py, "--config", str(base_config), *overrides]


def slurm_options(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    profile: str,
    job_count: int,
) -> dict[str, Any]:
    """Return sbatch options for a phase/profile launch."""

    block = phase_config(manifest, phase)
    resources = profile_slurm(manifest, profile)
    job_name = str(block.get("job_name") or f"hooke-pv-{phase.replace('_', '-')}")
    options: dict[str, Any] = {
        "job_name": job_name,
        "log_dir": phase_slurm_log_dir(manifest, phase),
    }
    for key in ("partition", "gres", "cpus_per_task", "mem_gb", "timeout_min"):
        value = resources.get(key)
        if value is not None:
            options[key] = value
    array_parallelism = resources.get("array_parallelism")
    if array_parallelism is None:
        array_parallelism = job_count
    options["array_parallelism"] = int(array_parallelism)
    return options


def job_index_sweep(job_count: int) -> str:
    """Return a comma-separated sweep over job indexes."""

    if job_count <= 0:
        raise ValueError("job count must be positive")
    return ",".join(str(index) for index in range(job_count))


def selected_config_id(selected: Mapping[str, Any] | None) -> str | None:
    """Return selected config id from a selector artifact."""

    if not selected:
        return None
    value = _select(selected, "selected.config_id") or _select(selected, "selection.selected_config_id")
    return None if value in (None, "") else str(value)


def selected_hyperparameters(selected: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return selected hyperparameters as dotted-key mapping."""

    if not selected:
        return {}
    values = _select(selected, "selected.hyperparameters")
    if isinstance(values, Mapping):
        return {str(key): value for key, value in values.items()}
    result: dict[str, Any] = {}
    for key in (
        "optimizer_params.lr",
        "model_params.channels",
        "model_params.layers",
        "model_params.gate_activation",
    ):
        value = _select(selected, f"selected.{key}")
        if value is not None:
            result[key] = value
    return result


def eval_job_run_id(train_seed: Any, eval_seed: Any) -> str:
    """Return a stable final-eval run id."""

    return f"train_seed={_key_text(train_seed)}_eval_seed={_key_text(eval_seed)}"


def stage_run_id(kind: str, config_id: str, suffix: str) -> str:
    """Return a run id whose first folder names smoke/full status."""

    return f"{kind}/{config_id}/{suffix}"


def run_kind_for_dir(run_root: str | Path, run_dir: str | Path) -> str:
    """Return ``smoke``/``full`` from a staged run directory path.

    Older or explicit test directories that do not start with a run-kind folder
    are treated as full runs.
    """

    root = Path(run_root)
    path = Path(run_dir)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return "full"
    if parts and parts[0] in {"smoke", "full"}:
        return parts[0]
    return "full"


def _cartesian_job(
    manifest: Mapping[str, Any],
    phase: str,
    sweep_values: Mapping[str, Any],
    *,
    selected: Mapping[str, Any] | None,
    smoke: bool,
) -> dict[str, Any]:
    block = phase_config(manifest, phase)
    overrides = _phase_fixed_overrides(manifest, phase)
    overrides.update(_phase_selection_overrides(block, selected))
    overrides.update(dict(sweep_values))

    config_id = selected_config_id(selected)
    if not config_id:
        config_id = config_id_from_values(overrides)
    overrides.setdefault("study.config_id", config_id)

    run_root = phase_run_root(manifest, phase)
    overrides["run.layout"] = "flat"
    run_id = _train_run_id(phase, overrides, config_id=config_id, smoke=smoke)
    overrides["run.root"] = run_root
    overrides["run.run_id"] = run_id

    return {
        "phase": "smoke_train" if smoke else phase,
        "target_phase": phase,
        "base_config": phase_base_config(manifest, phase),
        "run_root": run_root,
        "run_id": run_id,
        "run_dir": f"{run_root}/{run_id}",
        "config_id": config_id,
        "overrides": dotlist(overrides),
    }


def _phase_fixed_overrides(manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    block = phase_config(manifest, phase)
    overrides = block.get("overrides")
    fixed = overrides.get("fixed") if isinstance(overrides, Mapping) else {}
    values = {str(key): value for key, value in fixed.items()} if isinstance(fixed, Mapping) else {}
    values.update({key: value for key, value in phase_provenance_overrides(manifest, phase).items() if key not in values})
    return values


def _phase_selection_overrides(block: Mapping[str, Any], selected: Mapping[str, Any] | None) -> dict[str, Any]:
    overrides = block.get("overrides")
    if not isinstance(overrides, Mapping):
        return {}
    mapping = overrides.get("from_selection")
    if not isinstance(mapping, Mapping):
        return {}
    values: dict[str, Any] = {}
    dotted_hparams = selected_hyperparameters(selected)
    for target_key, source_path in mapping.items():
        value = _select(selected or {}, str(source_path))
        if value is None and str(target_key) in dotted_hparams:
            value = dotted_hparams[str(target_key)]
        if value is None:
            raise ValueError(f"selected_config.yaml does not provide {source_path!r}")
        values[str(target_key)] = value
    return values


def _select_smoke_jobs(jobs: Sequence[dict[str, Any]], block: Mapping[str, Any]) -> list[dict[str, Any]]:
    smoke = block.get("smoke") if isinstance(block.get("smoke"), Mapping) else {}
    mode = str(smoke.get("select") or "first")
    if mode != "first":
        raise ValueError(f"unsupported smoke.select={mode!r}")
    return [dict(jobs[0])] if jobs else []


def _apply_smoke_overlay(job: dict[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    smoke = block.get("smoke") if isinstance(block.get("smoke"), Mapping) else {}
    overlay = smoke.get("overlay") if isinstance(smoke.get("overlay"), Mapping) else {}
    overrides = dotlist_dict(job["overrides"])
    overrides.update({str(key): value for key, value in overlay.items()})
    target_phase = str(job.get("target_phase") or job.get("phase"))
    config_id = str(overrides.get("study.config_id") or job.get("config_id") or "config")
    run_id = _train_run_id(target_phase, overrides, config_id=config_id, smoke=True)
    overrides["run.run_id"] = run_id
    overrides["run.layout"] = "flat"
    job = dict(job)
    job["phase"] = "smoke_train" if block.get("orchestrator") == "train" else "smoke_eval"
    job["run_id"] = run_id
    job["run_dir"] = f"{job['run_root']}/{run_id}"
    job["overrides"] = dotlist(overrides)
    return job


def dotlist(values: Mapping[str, Any]) -> list[str]:
    """Convert a mapping into OmegaConf dotlist overrides."""

    return [f"{key}={_dotlist_value(value)}" for key, value in values.items()]


def dotlist_dict(overrides: Sequence[str]) -> dict[str, Any]:
    """Parse simple ``key=value`` dotlist entries into a mapping."""

    result: dict[str, Any] = {}
    for item in overrides:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise ValueError(f"override lacks '=': {item!r}")
        result[key] = _parse_scalar(value)
    return result


def config_id_from_values(values: Mapping[str, Any]) -> str:
    """Return deterministic non-seed config id."""

    parts = []
    for key in (
        "optimizer_params.lr",
        "model_params.channels",
        "model_params.layers",
        "model_params.gate_activation",
    ):
        if key in values:
            parts.append(f"{key.split('.')[-1]}{_slug(values[key])}")
    return "config_" + "_".join(parts) if parts else "config"


def _train_run_id(phase: str, overrides: Mapping[str, Any], *, config_id: str, smoke: bool) -> str:
    seed = _key_text(overrides.get(SEED_KEY))
    kind = "smoke" if smoke else "full"
    return stage_run_id(kind, config_id, f"seed={seed}")


def _profile_block(manifest: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, Mapping) or profile not in profiles:
        raise ValueError(f"manifest profile {profile!r} is not defined")
    block = profiles[profile]
    if not isinstance(block, Mapping):
        raise ValueError(f"manifest profile {profile!r} must be a mapping")
    return block


def _as_sequence(value: Any, *, key: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"manifest sweep axis {key!r} must be a sequence")
    if not value:
        raise ValueError(f"manifest sweep axis {key!r} must be non-empty")
    return list(value)


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _fixed_override(block: Mapping[str, Any], key: str) -> Any:
    overrides = block.get("overrides")
    fixed = overrides.get("fixed") if isinstance(overrides, Mapping) else {}
    if isinstance(fixed, Mapping):
        return fixed.get(key)
    return None


def _dotlist_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:g}"
    return str(value)


def _parse_scalar(value: Any) -> Any:
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


def _key_text(value: Any) -> str:
    parsed = _parse_scalar(value)
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return "" if parsed is None else str(parsed)


def _slug(value: Any) -> str:
    text = _dotlist_value(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "DEFAULT_MANIFEST",
    "EVAL_PHASES",
    "SEED_KEY",
    "SMOKE_PHASE_TARGETS",
    "TRAIN_PHASES",
    "cartesian_phase_jobs",
    "collect_report_dir",
    "command_for_job",
    "config_id_from_values",
    "dotlist",
    "dotlist_dict",
    "eval_job_run_id",
    "expected_validation_seeds",
    "final_eval_plan_dir",
    "final_eval_report_dir",
    "final_train_plan_dir",
    "final_train_report_dir",
    "job_index_sweep",
    "load_jsonl",
    "load_yaml",
    "phase_base_config",
    "phase_config",
    "phase_for_launch",
    "phase_provenance_overrides",
    "phase_run_root",
    "phase_slurm_log_dir",
    "phase_study_name",
    "phase_study_phase",
    "phase_study_version",
    "phase_sweep",
    "profile_device",
    "profile_environment",
    "profile_extras",
    "profile_slurm",
    "report_root",
    "run_kind_for_dir",
    "save_yaml",
    "select_report_dir",
    "selected_config_id",
    "selected_hyperparameters",
    "slurm_options",
    "stage_run_id",
    "write_jsonl",
]
