"""Shared data-handler helpers for Hooke experiment wrappers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from spenn.training.artifacts import run_time_stamp

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HookeScriptSpec:
    """Describe a generated-config handoff to an executable script.

    Parameters
    ----------
    entrypoint : pathlib.Path
        Script that executes the actual experiment stack.
    generated_subdir : str
        Subdirectory under ``outputs/generated_configs`` for generated YAML.
    run_id_prefix : str
        Prefix used when a run id is not supplied by the caller.
    """

    entrypoint: Path
    generated_subdir: str
    run_id_prefix: str


def run_generated_config(
    cfg_or_path: DictConfig | Path | str,
    spec: HookeScriptSpec,
    *,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    forwarded_overrides: list[str] | None = None,
    apply_forwarded_overrides: bool = True,
) -> dict[str, object]:
    """Generate a config, execute the script, and return its summary.

    Parameters
    ----------
    cfg_or_path : omegaconf.DictConfig, pathlib.Path, or str
        Template config object or YAML path.
    spec : HookeScriptSpec
        Script handoff specification.
    run_id : str or None, optional
        Run identifier to apply as the top-level ``run_id`` override.
    output_root : str, pathlib.Path, or None, optional
        Output root to apply as the top-level ``output_root`` override.
    forwarded_overrides : list of str or None, optional
        Dotlist overrides to merge and pass through for artifact recording.
    apply_forwarded_overrides : bool, optional
        Whether to apply dotlist overrides before writing the generated config.
        Set this to ``False`` only when the caller has already merged those
        overrides into ``cfg_or_path`` and wants them recorded without applying
        them a second time.

    Returns
    -------
    dict
        Summary emitted by the executable script.
    """

    forwarded = list(forwarded_overrides or [])
    cfg = _load_config(cfg_or_path)
    dotlist = [override for override in forwarded if "=" in override] if apply_forwarded_overrides else []
    generated_overrides = _generated_overrides(cfg, spec, run_id=run_id, output_root=output_root, dotlist=dotlist)
    if generated_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(generated_overrides))
    OmegaConf.resolve(cfg)

    generated = _write_generated_config(cfg, spec)
    recorded_overrides = _recorded_overrides(forwarded, generated_overrides)
    command = [sys.executable, str(spec.entrypoint), "--config", str(generated), *recorded_overrides]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Hooke executable script failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    summary = OmegaConf.to_container(OmegaConf.create(completed.stdout), resolve=True)
    if not isinstance(summary, dict):
        raise TypeError(f"Hooke executable summary must be a mapping, got {type(summary).__name__}")
    return dict(summary)


def resolve_config_path(path: Path, config_dir: Path) -> Path:
    """Resolve a Hooke config path or short template name.

    Parameters
    ----------
    path : pathlib.Path
        User-supplied config path or basename.
    config_dir : pathlib.Path
        Directory containing Hooke config templates.

    Returns
    -------
    pathlib.Path
        Existing config path when found, otherwise the original path so callers
        get the normal file-not-found error from OmegaConf.
    """

    if path.exists():
        return path
    candidates = [config_dir / path]
    if path.suffix == "":
        candidates.append(config_dir / f"{path}.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def configured_run_id(cfg: DictConfig) -> str | None:
    """Return a non-empty run id configured on either supported key.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Config that may contain top-level ``run_id`` or nested ``run.id``.

    Returns
    -------
    str or None
        Configured run id, or ``None`` when both fields are absent or blank.
    """

    for key in ("run_id", "run.id"):
        value = OmegaConf.select(cfg, key, default=None)
        if value is not None and str(value) not in {"", "None", "null"}:
            return str(value)
    return None


def _generated_overrides(
    cfg: DictConfig,
    spec: HookeScriptSpec,
    *,
    run_id: str | None,
    output_root: str | Path | None,
    dotlist: list[str],
) -> list[str]:
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist)) if dotlist else cfg
    selected_run_time = OmegaConf.select(merged, "run.time", default=None)
    if selected_run_time is None:
        selected_run_time = run_time_stamp()
    selected_run_id = run_id
    if selected_run_id is None:
        selected_run_id = configured_run_id(merged)
    if selected_run_id is None:
        selected_run_id = _new_run_id(spec.run_id_prefix, str(selected_run_time))
    generated = [*dotlist, f"run.time={selected_run_time}", f"run_id={selected_run_id}"]
    if output_root is not None:
        generated.append(f"output_root={output_root}")
    return generated


def _recorded_overrides(forwarded: list[str], generated_overrides: list[str]) -> list[str]:
    recorded = list(forwarded)
    for override in generated_overrides:
        if override not in recorded:
            recorded.append(override)
    return recorded


def _load_config(cfg_or_path: DictConfig | Path | str) -> DictConfig:
    if isinstance(cfg_or_path, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg_or_path, resolve=False))
    return OmegaConf.load(Path(cfg_or_path))


def _new_run_id(prefix: str, run_time: str | None = None) -> str:
    selected_time = run_time_stamp() if run_time is None else str(run_time)
    return f"{prefix}_{selected_time}_{uuid4().hex[:8]}"


def _write_generated_config(cfg: DictConfig, spec: HookeScriptSpec) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    run_id = str(OmegaConf.select(cfg, "run_id"))
    selected_root = OmegaConf.select(cfg, "output_root", default=OmegaConf.select(cfg, "run.output_root", default="outputs"))
    output_root = Path(str(selected_root))
    root = output_root if output_root.is_absolute() else ROOT / output_root
    path = root / "generated_configs" / date / spec.generated_subdir / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    return path


__all__ = ["HookeScriptSpec", "configured_run_id", "resolve_config_path", "run_generated_config"]
