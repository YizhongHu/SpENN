"""Generate and run Hooke multibody SpENN configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hooke.runner import HookeScriptSpec, run_generated_config  # noqa: E402
from spenn.training.artifacts import make_output_dir, make_run_id, run_time_stamp, write_json  # noqa: E402
from spenn.training.artifacts import write_config_artifacts  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "spenn.yaml"
DEFAULT_RUN_ID_PREFIX = "hooke_multibody_spenn"
TRAIN_ENTRYPOINT = ROOT / "train.py"


def main() -> None:
    """Run a Hooke multibody SpENN config through the generic train script.

    Returns
    -------
    None
        The run summary is printed to standard output in YAML format.
    """

    args = _parse_args()
    cfg = load_config(_resolve_template_path(args.config))
    if args.scan_spins:
        summary = run_spin_scan(
            cfg,
            forwarded_overrides=args.forwarded,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    else:
        summary = run(
            cfg,
            forwarded_overrides=args.forwarded,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    print(OmegaConf.to_yaml(summary))


def load_config(path: Path) -> DictConfig:
    """Load a config template, resolving optional ``inherits``.

    Parameters
    ----------
    path : pathlib.Path
        Config path or short name resolved against ``configs``.

    Returns
    -------
    omegaconf.DictConfig
        Loaded and merged config template.
    """

    cfg = OmegaConf.load(path)
    parent = OmegaConf.select(cfg, "inherits", default=None)
    if parent is None:
        return cfg
    parent_path = _resolve_template_path(Path(str(parent)))
    return OmegaConf.merge(load_config(parent_path), cfg)


def run(
    cfg: DictConfig,
    *,
    forwarded_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Generate a config, run SpENN training, and summarize the result.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hooke multibody SpENN config template.
    forwarded_overrides : list of str or None, optional
        Dotlist overrides recorded and forwarded to ``train.py``.
    run_id : str or None, optional
        Run id override.
    output_root : str, pathlib.Path, or None, optional
        Output root override.

    Returns
    -------
    dict
        Hooke multibody summary.
    """

    generic = run_generated_config(
        cfg,
        HookeScriptSpec(
            entrypoint=TRAIN_ENTRYPOINT,
            generated_subdir="hooke_multibody_spenn",
            run_id_prefix=DEFAULT_RUN_ID_PREFIX,
        ),
        run_id=run_id,
        output_root=output_root,
        forwarded_overrides=forwarded_overrides,
    )
    return _summary_from_artifact(generic["output_dir"])


def run_spin_scan(
    cfg: DictConfig,
    *,
    forwarded_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Run fixed-spin-sector configs and report the lowest-energy candidate.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Benchmark config with ``scan.spin_partitions``.
    forwarded_overrides : list of str or None, optional
        Dotlist overrides recorded and forwarded to each child run.
    run_id : str or None, optional
        Scan id. Child run ids append ``up{n_up}_down{n_down}``.
    output_root : str, pathlib.Path, or None, optional
        Output root override.

    Returns
    -------
    dict
        Spin-scan summary.
    """

    run_time = run_time_stamp()
    scan_id = run_id or make_run_id("hooke_multibody_spin_scan", run_time=run_time)
    output_root_path = Path(str(output_root or OmegaConf.select(cfg, "output_root", default="outputs")))
    scan_cfg = OmegaConf.merge(
        cfg,
        {
            "run": {"time": run_time},
            "run_id": scan_id,
            "output_root": str(output_root_path),
        },
    )
    partitions = OmegaConf.select(cfg, "scan.spin_partitions", default=None)
    if not partitions:
        raise ValueError("scan.spin_partitions must contain at least one [n_up, n_down] pair")
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for partition in partitions:
        n_up, n_down = int(partition[0]), int(partition[1])
        child_run_id = f"{scan_id}_up{n_up}_down{n_down}"
        child_cfg = OmegaConf.merge(
            scan_cfg,
            {
                "n_electrons": n_up + n_down,
                "n_up": n_up,
                "n_down": n_down,
                "run_id": child_run_id,
                "run": {"time": run_time},
            },
        )
        summary = run(
            child_cfg,
            forwarded_overrides=forwarded_overrides,
            run_id=child_run_id,
            output_root=output_root,
        )
        summaries.append(summary)
        rows.append(
            {
                "run_id": summary["run_id"],
                "output_dir": summary["output_dir"],
                "run_time": summary["run_time"],
                "n_up": n_up,
                "n_down": n_down,
                "energy_mean": summary["energy_mean"],
                "local_energy_variance": summary["local_energy_variance"],
                "acceptance_rate": summary["acceptance_rate"],
            }
        )
    best = min(rows, key=lambda row: float(row["energy_mean"]))
    output_dir = make_output_dir(
        output_root_path,
        run_name="hooke_multibody_spin_scan",
        run_id=scan_id,
        include_plots=False,
    )
    scan_artifact_cfg = OmegaConf.create(OmegaConf.to_container(scan_cfg, resolve=False))
    OmegaConf.resolve(scan_artifact_cfg)
    write_config_artifacts(output_dir, scan_artifact_cfg, forwarded_overrides or [])
    _write_csv(output_dir / "metrics" / "spin_scan_summary.csv", rows)
    payload = {
        "entrypoint": "experiments/hooke_multibody/run_spenn.py",
        "status": "ok",
        "mode": "spin_scan",
        "run_id": scan_id,
        "run_time": run_time,
        "output_dir": str(output_dir),
        "config": OmegaConf.to_container(scan_artifact_cfg, resolve=True),
        "best_run": best,
        "runs": summaries,
    }
    write_json(output_dir / "artifacts" / "summary.json", payload)
    return payload


def _summary_from_artifact(output_dir: str | Path) -> dict[str, object]:
    with (Path(output_dir) / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    cfg = artifact["config"]
    metrics = artifact["metrics"]
    acceptance = float(metrics["sampler/acceptance_rate"])
    validation = cfg.get("validation", {})
    acceptance_min = float(validation.get("acceptance_min", 0.0))
    acceptance_max = float(validation.get("acceptance_max", 1.0))
    energy = float(metrics["spenn/energy/mean"])
    return {
        "entrypoint": "experiments/hooke_multibody/run_spenn.py",
        "status": "ok",
        "can_reach_goal": math.isfinite(energy) and acceptance_min <= acceptance <= acceptance_max,
        "run_id": artifact["run_id"],
        "run_time": artifact["run_time"],
        "output_dir": artifact["output_dir"],
        "n_electrons": cfg["system"]["n_electrons"],
        "n_up": cfg["system"]["n_up"],
        "n_down": cfg["system"]["n_down"],
        "omega": cfg["system"]["harmonic_omega"],
        "reference_available": False,
        "energy_mean": energy,
        "local_energy_variance": metrics["spenn/local_energy/variance"],
        "acceptance_rate": acceptance,
        "mean_pair_distance": metrics["sampler/mean_pair_distance"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_template_path(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    if path.exists() and path.is_file():
        return path
    candidates = [CONFIG_DIR / path]
    if path.suffix == "":
        candidates.append(CONFIG_DIR / f"{path}.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config template path or name.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scan-spins", action="store_true", help="Run fixed spin sectors from scan.spin_partitions.")
    parser.add_argument("forwarded", nargs="*", help="Dotlist overrides recorded and forwarded to train.py.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
