"""Generic run artifact helpers for configured SpENN executions."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUN_DIRS = ("checkpoints", "traces", "diagnostics", "figures")


class ArtifactManager:
    """Own the standard artifact layout for one run.

    Parameters
    ----------
    root : pathlib.Path or str
        Output root. Relative paths are interpreted relative to the
        repository root.
    experiment : str
        Experiment family name.
    sector : str
        Experiment sector or suite name.
    run_id : str
        Unique run identifier.
    """

    def __init__(self, root: Path | str, experiment: str, sector: str, run_id: str) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_absolute() else ROOT / root_path
        self.experiment = str(experiment)
        self.sector = str(sector)
        self.run_id = str(run_id)

    @property
    def run_dir(self) -> Path:
        """Return the run directory path."""

        return self.root / self.experiment / self.sector / self.run_id

    def make_dirs(self) -> None:
        """Create the run directory and standard child directories."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_RUN_DIRS:
            self.path(name).mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        """Return a path under this run directory."""

        return self.run_dir.joinpath(*parts)


@dataclass
class RunMetadata:
    """Execution metadata captured for one configured run."""

    run_id: str
    run_name: str
    timestamp: str
    git_commit: str
    git_branch: str
    dirty_worktree: bool
    command: str | None
    config_path: str | None
    resolved_config_path: str
    run_dir: str
    device: str
    dtype: str
    status: str = "initialized"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        data = asdict(self)
        extra = data.pop("extra")
        data.update(extra)
        return data


@dataclass
class RunResult:
    """Result returned by a configured runner."""

    status: str
    run_dir: Path | None = None
    error: str | None = None


@dataclass
class RunContext:
    """Runtime context shared by runners, callbacks, and loggers."""

    cfg: DictConfig
    source_cfg: DictConfig
    artifact_manager: ArtifactManager
    metadata: RunMetadata
    callbacks: list[Any] = field(default_factory=list)
    loggers: list[Any] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        """Return the active run directory."""

        return self.artifact_manager.run_dir

    def path(self, *parts: str) -> Path:
        """Return a path under the active run directory."""

        return self.artifact_manager.path(*parts)

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        namespace: str = "run",
        event: str | None = None,
    ) -> None:
        """Emit one metric record to every configured logger."""

        from spenn.logging import LogRecord

        record = LogRecord(step=step, namespace=namespace, metrics=dict(metrics), event=event)
        for logger in self.loggers:
            logger.log(record)


def generate_run_id(run_name: str) -> str:
    """Return a timestamped run identifier."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(run_name)
    return f"{timestamp}_{slug}_{uuid4().hex[:6]}"


def build_run_metadata(
    cfg: DictConfig,
    *,
    command: str | None,
    config_path: str | None,
) -> RunMetadata:
    """Build metadata for a resolved run config."""

    git = collect_git_metadata()
    return RunMetadata(
        run_id=str(OmegaConf.select(cfg, "run.run_id")),
        run_name=str(OmegaConf.select(cfg, "experiment.run_name", default=OmegaConf.select(cfg, "experiment.name"))),
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=str(git["git_commit"]),
        git_branch=str(git["git_branch"]),
        dirty_worktree=bool(git["dirty_worktree"]),
        command=command,
        config_path=config_path,
        resolved_config_path=str(Path(str(OmegaConf.select(cfg, "run.dir"))) / "resolved_config.yaml"),
        run_dir=str(OmegaConf.select(cfg, "run.dir")),
        device=str(OmegaConf.select(cfg, "runtime.device", default="cpu")),
        dtype=str(OmegaConf.select(cfg, "runtime.dtype", default="float64")),
    )


def collect_git_metadata() -> dict[str, Any]:
    """Collect git commit, branch, and dirty-state metadata."""

    status = _run_git(["git", "status", "--short", "--untracked-files=all"])
    return {
        "git_commit": _run_git(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_git(["git", "branch", "--show-current"]),
        "dirty_worktree": bool(status.strip()),
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "run"


def _run_git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


__all__ = [
    "ArtifactManager",
    "REQUIRED_RUN_DIRS",
    "RunContext",
    "RunMetadata",
    "RunResult",
    "build_run_metadata",
    "collect_git_metadata",
    "generate_run_id",
    "write_json",
]
