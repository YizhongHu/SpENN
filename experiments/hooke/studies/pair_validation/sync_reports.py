"""Mirror pair-validation reports while keeping only compact run artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from .study_manifest import DEFAULT_MANIFEST, load_yaml, report_root
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import DEFAULT_MANIFEST, load_yaml, report_root

CHECKPOINT_DIRNAME = "checkpoints"
LATEST_JSON = "latest.json"
SLURM_LOG_DIRNAMES = {"slurm", "slurm_logs"}


@dataclass(frozen=True)
class SyncPlan:
    """Concrete copy plan for one reports sync."""

    files: tuple[Path, ...]
    skipped_checkpoint_files: int
    skipped_slurm_log_files: int
    warnings: tuple[str, ...] = ()


@dataclass
class SyncSummary:
    """Counts from one reports sync."""

    source: Path
    destination: Path
    dry_run: bool
    scanned_files: int
    copied_files: int
    copied_bytes: int
    skipped_checkpoint_files: int
    skipped_slurm_log_files: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""

        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "dry_run": self.dry_run,
            "scanned_files": self.scanned_files,
            "copied_files": self.copied_files,
            "copied_bytes": self.copied_bytes,
            "skipped_checkpoint_files": self.skipped_checkpoint_files,
            "skipped_slurm_log_files": self.skipped_slurm_log_files,
            "warnings": list(self.warnings),
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reports-sync CLI."""

    args = _parse_args(argv)
    source = resolve_source(args.manifest, args.source)
    summary = sync_reports(
        source=source,
        destination=args.destination,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    _print_summary(summary)
    return 0


def resolve_source(manifest_path: str | Path, source: str | Path | None = None) -> Path:
    """Resolve the report source from an explicit path or the manifest."""

    if source is not None:
        return Path(source).resolve()
    manifest = load_yaml(manifest_path)
    raw = Path(report_root(manifest))
    if raw.is_absolute():
        return raw
    cwd_candidate = raw.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = _find_repo_root(Path(manifest_path).resolve().parent)
    return (repo_root / raw).resolve()


def sync_reports(
    *,
    source: str | Path,
    destination: str | Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> SyncSummary:
    """Mirror report files into ``destination``.

    The destination is replaced as a mirror. Slurm log directories are skipped,
    and checkpoint directories keep only ``latest.json`` plus the step directory
    referenced by that pointer.
    """

    source_root = Path(source).resolve()
    destination_root = Path(destination).resolve()
    _validate_roots(source_root, destination_root)

    plan = build_sync_plan(source_root)
    copied_bytes = sum(path.lstat().st_size for path in plan.files)
    if verbose:
        for path in plan.files:
            print(path.relative_to(source_root))
    if not dry_run:
        _replace_destination(destination_root)
        for path in plan.files:
            _copy_file(path, destination_root / path.relative_to(source_root))
    return SyncSummary(
        source=source_root,
        destination=destination_root,
        dry_run=dry_run,
        scanned_files=(
            len(plan.files)
            + plan.skipped_checkpoint_files
            + plan.skipped_slurm_log_files
        ),
        copied_files=len(plan.files),
        copied_bytes=copied_bytes,
        skipped_checkpoint_files=plan.skipped_checkpoint_files,
        skipped_slurm_log_files=plan.skipped_slurm_log_files,
        warnings=list(plan.warnings),
    )


def build_sync_plan(source: str | Path) -> SyncPlan:
    """Return files to copy from ``source``."""

    source_root = Path(source).resolve()
    latest_dirs, warnings = _latest_checkpoint_dirs(source_root)
    files: list[Path] = []
    skipped_checkpoint = 0
    skipped_slurm = 0
    for path in sorted(source_root.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        reason = _skip_reason(path, source_root, latest_dirs)
        if reason == "slurm_logs":
            skipped_slurm += 1
            continue
        if reason == "checkpoint":
            skipped_checkpoint += 1
            continue
        files.append(path)
    return SyncPlan(
        files=tuple(files),
        skipped_checkpoint_files=skipped_checkpoint,
        skipped_slurm_log_files=skipped_slurm,
        warnings=tuple(warnings),
    )


def _latest_checkpoint_dirs(source_root: Path) -> tuple[dict[Path, Path | None], list[str]]:
    latest_dirs: dict[Path, Path | None] = {}
    warnings: list[str] = []
    for latest_json in sorted(source_root.rglob(f"{CHECKPOINT_DIRNAME}/{LATEST_JSON}")):
        checkpoint_root = latest_json.parent.resolve()
        try:
            data = json.loads(latest_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"{latest_json}: invalid JSON: {exc}")
            latest_dirs[checkpoint_root] = None
            continue
        checkpoint_name = data.get("checkpoint_dir") if isinstance(data, dict) else None
        if not checkpoint_name:
            warnings.append(f"{latest_json}: missing checkpoint_dir")
            latest_dirs[checkpoint_root] = None
            continue
        latest_dir = (checkpoint_root / str(checkpoint_name)).resolve()
        if not latest_dir.is_dir():
            warnings.append(f"{latest_json}: latest checkpoint directory missing: {latest_dir}")
            latest_dirs[checkpoint_root] = None
            continue
        latest_dirs[checkpoint_root] = latest_dir
    return latest_dirs, warnings


def _skip_reason(path: Path, source_root: Path, latest_dirs: dict[Path, Path | None]) -> str | None:
    rel_parts = path.relative_to(source_root).parts
    if any(part in SLURM_LOG_DIRNAMES for part in rel_parts):
        return "slurm_logs"
    if CHECKPOINT_DIRNAME not in rel_parts:
        return None

    checkpoint_index = rel_parts.index(CHECKPOINT_DIRNAME)
    checkpoint_root = source_root.joinpath(*rel_parts[: checkpoint_index + 1]).resolve()
    if path.name == LATEST_JSON and path.parent.resolve() == checkpoint_root:
        return None
    latest_dir = latest_dirs.get(checkpoint_root)
    if latest_dir is not None and _is_relative_to(path.resolve(), latest_dir):
        return None
    return "checkpoint"


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _replace_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"destination must be a directory or absent: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def _validate_roots(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"source report directory does not exist: {source}")
    if source == destination:
        raise ValueError("source and destination must be different directories")
    if _is_relative_to(destination, source):
        raise ValueError("destination must not be inside the source report directory")
    if _is_relative_to(source, destination):
        raise ValueError("destination must not be an ancestor of the source report directory")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _print_summary(summary: SyncSummary) -> None:
    print(f"source: {summary.source}")
    print(f"destination: {summary.destination}")
    print(f"dry_run: {str(summary.dry_run).lower()}")
    print(f"copied_files: {summary.copied_files}")
    print(f"copied_bytes: {summary.copied_bytes}")
    print(f"skipped_checkpoint_files: {summary.skipped_checkpoint_files}")
    print(f"skipped_slurm_log_files: {summary.skipped_slurm_log_files}")
    for warning in summary.warnings:
        print(f"warning: {warning}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        help="Destination directory to replace with the compact report mirror.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Study manifest used to find reports.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Override the source reports directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the copy summary without writing the destination.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each copied relative path.")
    return parser.parse_args(argv)


__all__ = [
    "SyncPlan",
    "SyncSummary",
    "build_sync_plan",
    "main",
    "resolve_source",
    "sync_reports",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
