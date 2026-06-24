"""Plan final pair-stability replicate jobs from selected champions.

Consumes a durable ``04_select`` attempt and writes a durable ``05_final_grid``
attempt. The final grid is the source of truth for final replicate indices and
the independent final train/eval seed policy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from utils.io import write_json
from utils.layout import (
    STAGE_FINAL_GRID,
    STAGE_SELECT,
    final_grid_attempt_dir,
    latest_attempt_id,
    smoke_attempt_id,
    stage_dir,
    write_latest,
)
from utils.time import STUDY_TIMEZONE, new_attempt_id

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_TRAIN_CONFIG = STUDY_DIR / "configs" / "pair_stability.yaml"
DEFAULT_EVAL_CONFIG = STUDY_DIR / "configs" / "pair_validation.yaml"
DEFAULT_REPLICATES = 3
SMOKE_CHAMPION_LIMIT = 2
SMOKE_REPLICATES = 1


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _resolve_selection_attempt(results_root: Path, selection_attempt_id: str | None, *, smoke: bool) -> str:
    if selection_attempt_id is not None:
        return selection_attempt_id
    select_stage = stage_dir(results_root, STAGE_SELECT)
    attempt_id = latest_attempt_id(select_stage, smoke=smoke)
    if attempt_id is None:
        raise FileNotFoundError(f"no selection attempts under {select_stage}")
    return attempt_id


def read_champions(selection_dir: Path) -> list[dict[str, str]]:
    """Read selected champions from ``04_select/{attempt_id}/champions.csv``."""

    champions_path = selection_dir / "champions.csv"
    if not champions_path.is_file():
        raise FileNotFoundError(f"selection attempt has no champions.csv: {champions_path}")
    with champions_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _final_run_id(champion: dict[str, str], *, replicate_index: int) -> str:
    config_id = champion.get("config_id") or "unknown-config"
    winner_kind = champion.get("winner_kind") or "winner"
    return f"{config_id}_winner-{winner_kind}_rep-{int(replicate_index)}"


def _seed_policy(replicate_index: int) -> dict[str, int]:
    return {
        "final_train_sampler_seed": 101 + int(replicate_index),
        "final_train_model_seed": 1001 + int(replicate_index),
        "final_eval_seed": 10001 + int(replicate_index),
    }


def build_final_jobs(
    champions: Sequence[dict[str, str]],
    *,
    source_selection_attempt_id: str,
    source_selection_attempt_dir: str | Path,
    replicates: int,
    champion_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Expand selected champion rows into final replicate job rows."""

    selected = [
        (index, champion)
        for index, champion in enumerate(champions)
        if str(champion.get("config_id", "")).strip()
    ]
    if champion_limit is not None:
        selected = selected[: int(champion_limit)]
    jobs: list[dict[str, Any]] = []
    for champion_index, champion in selected:
        source_champion_id = f"champion-{champion_index:04d}"
        for replicate_index in range(int(replicates)):
            seeds = _seed_policy(replicate_index)
            jobs.append(
                {
                    "source_selection_attempt_id": source_selection_attempt_id,
                    "source_selection_attempt_dir": str(source_selection_attempt_dir),
                    "source_champion_id": source_champion_id,
                    "source_champion_row_index": champion_index,
                    "source_scan_run_id": champion.get("config_id", ""),
                    "source_scan_run_ids": champion.get("run_ids", ""),
                    "final_run_id": _final_run_id(champion, replicate_index=replicate_index),
                    "replicate_index": replicate_index,
                    "winner_kind": champion.get("winner_kind", ""),
                    "architecture": champion.get("architecture", ""),
                    "normalization": champion.get("normalization", ""),
                    "basis_envelope": champion.get("architecture", ""),
                    "lr": champion.get("lr", ""),
                    "channels": champion.get("channels", ""),
                    "metric": champion.get("metric", ""),
                    "metric_value": champion.get("metric_value", ""),
                    **seeds,
                    "source_champion": dict(champion),
                }
            )
    return jobs


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_final_grid_attempt(
    *,
    results_root: str | Path,
    attempt_id: str,
    created_at: str,
    source_selection_attempt_id: str,
    source_selection_attempt_dir: str | Path,
    train_config: str | Path,
    eval_config: str | Path,
    replicates: int,
    smoke: bool,
    champions: Sequence[dict[str, str]],
    jobs: Sequence[dict[str, Any]],
) -> Path:
    """Write ``05_final_grid`` artifacts and return the attempt directory."""

    results_root = Path(results_root)
    attempt = final_grid_attempt_dir(results_root, attempt_id)
    (attempt / "jobs").mkdir(parents=True, exist_ok=True)

    source_selection_dir = Path(source_selection_attempt_dir)
    write_json(
        attempt / "source_selection_attempt.json",
        {
            "selection_attempt_id": source_selection_attempt_id,
            "selection_attempt_dir": str(source_selection_dir),
            "champions_path": str(source_selection_dir / "champions.csv"),
        },
    )
    champions_text = (source_selection_dir / "champions.csv").read_text()
    (attempt / "source_champions.csv").write_text(champions_text)

    columns = [
        "source_selection_attempt_id",
        "source_champion_id",
        "source_champion_row_index",
        "source_scan_run_id",
        "source_scan_run_ids",
        "final_run_id",
        "replicate_index",
        "winner_kind",
        "architecture",
        "normalization",
        "basis_envelope",
        "lr",
        "channels",
        "metric",
        "metric_value",
        "final_train_sampler_seed",
        "final_train_model_seed",
        "final_eval_seed",
    ]
    _write_csv(attempt / "final_jobs.csv", jobs, columns)
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['final_run_id']}.json", job)

    manifest = {
        "study": "pair_stability",
        "stage": STAGE_FINAL_GRID,
        "attempt_id": attempt_id,
        "created_at": created_at,
        "results_root": str(results_root),
        "source_selection_attempt_id": source_selection_attempt_id,
        "source_selection_attempt_dir": str(source_selection_dir),
        "train_config": str(train_config),
        "eval_config": str(eval_config),
        "replicates": int(replicates),
        "smoke": bool(smoke),
        "n_source_champions": len(champions),
        "n_jobs": len(jobs),
        "seed_policy": {
            "final_train_sampler_seed": "101 + replicate_index",
            "final_train_model_seed": "1001 + replicate_index",
            "final_eval_seed": "10001 + replicate_index",
        },
    }
    write_json(attempt / "manifest.json", manifest)
    OmegaConf.save(OmegaConf.create(manifest), attempt / "manifest.yaml")
    write_latest(stage_dir(results_root, STAGE_FINAL_GRID), attempt_id, smoke=smoke)
    return attempt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-grid planning arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--selection-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--replicates", type=positive_int, default=DEFAULT_REPLICATES)
    parser.add_argument("--limit-champions", type=positive_int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Plan first 1-2 champions with one replicate each.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create a ``05_final_grid`` attempt from selected champions."""

    args = parse_args(argv)
    results_root = Path(args.results_root)
    selection_attempt_id = _resolve_selection_attempt(results_root, args.selection_attempt_id, smoke=args.smoke)
    selection_dir = stage_dir(results_root, STAGE_SELECT) / selection_attempt_id
    champions = read_champions(selection_dir)

    replicates = SMOKE_REPLICATES if args.smoke else args.replicates
    champion_limit = SMOKE_CHAMPION_LIMIT if args.smoke else args.limit_champions
    attempt_id = args.attempt_id or new_attempt_id()
    if args.smoke:
        attempt_id = smoke_attempt_id(attempt_id)
    created_at = datetime.now(STUDY_TIMEZONE).isoformat(timespec="seconds")

    jobs = build_final_jobs(
        champions,
        source_selection_attempt_id=selection_attempt_id,
        source_selection_attempt_dir=selection_dir,
        replicates=replicates,
        champion_limit=champion_limit,
    )
    attempt = write_final_grid_attempt(
        results_root=results_root,
        attempt_id=attempt_id,
        created_at=created_at,
        source_selection_attempt_id=selection_attempt_id,
        source_selection_attempt_dir=selection_dir,
        train_config=args.train_config,
        eval_config=args.eval_config,
        replicates=replicates,
        smoke=args.smoke,
        champions=champions,
        jobs=jobs,
    )
    print(f"[pair_stability] wrote 05_final_grid attempt {attempt_id} with {len(jobs)} jobs -> {attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
