"""Run artifact helpers for reproducible experiments."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def make_run_id(prefix: str = "run") -> str:
    """Return a timestamped run identifier.

    Parameters
    ----------
    prefix : str, optional
        Human-readable run id prefix.

    Returns
    -------
    str
        Run identifier suitable for artifact directory names.
    """

    return f"{prefix}_{datetime.now().strftime('%H%M%S')}_{uuid4().hex[:8]}"


def make_output_dir(output_root: Path, *, run_name: str, run_id: str, include_plots: bool = True) -> Path:
    """Create the standard run artifact directory tree.

    Parameters
    ----------
    output_root : pathlib.Path
        Root output directory, relative to the repository root unless absolute.
    run_name : str
        Run family name, for example ``"hooke_singlet_spenn"``.
    run_id : str
        Reproducible run identifier.
    include_plots : bool, optional
        Whether to create the ``plots`` CSV-data directory.

    Returns
    -------
    pathlib.Path
        Created output directory.
    """

    date = datetime.now().strftime("%Y-%m-%d")
    path = output_root if output_root.is_absolute() else ROOT / output_root
    path = path / date / run_name / run_id
    children = [".hydra", "checkpoints", "metrics", "artifacts"]
    if include_plots:
        children.append("plots")
    for child in children:
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def write_config_artifacts(output_dir: Path, cfg: DictConfig, overrides: list[str]) -> None:
    """Write resolved configuration artifacts.

    Parameters
    ----------
    output_dir : pathlib.Path
        Run output directory.
    cfg : omegaconf.DictConfig
        Resolved run configuration.
    overrides : list of str
        Command-line overrides to record.

    Returns
    -------
    None
        Config artifacts are written under ``output_dir/.hydra``.
    """

    OmegaConf.save(cfg, output_dir / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create(overrides), output_dir / ".hydra" / "overrides.yaml")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to a CSV artifact.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    rows : list of dict
        CSV rows. The first row defines the header.

    Returns
    -------
    None
        The CSV file is written when rows are present.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON artifact.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    payload : dict
        JSON-serializable payload.

    Returns
    -------
    None
        The JSON file is written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def git_metadata() -> dict[str, str]:
    """Collect git commit and dirty-state metadata.

    Returns
    -------
    dict of str to str
        Git commit hash and short dirty-state text.
    """

    return {
        "git_commit": _run_git(["git", "rev-parse", "HEAD"]),
        "dirty_git_state": _run_git(["git", "status", "--short", "--untracked-files=all"]),
    }


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pad rows so every CSV row has the same keys.

    Parameters
    ----------
    rows : list of dict
        Heterogeneous metric rows.

    Returns
    -------
    list of dict
        Rows with a stable union header.
    """

    if not rows:
        return rows
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _run_git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()
