"""Write Hooke multibody reference and baseline artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hooke.runner import configured_run_id, resolve_config_path  # noqa: E402
from experiments.hooke_multibody.reference import (  # noqa: E402
    gaussian_hartree_reference,
    gaussian_pair_distance_density_rows,
    gaussian_radial_density_rows,
)
from spenn.training.artifacts import (  # noqa: E402
    git_metadata,
    make_output_dir,
    make_run_id,
    run_time_stamp,
    write_config_artifacts,
    write_csv,
    write_json,
)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "reference.yaml"


def main() -> None:
    """Write a reference or baseline run and print its summary.

    Returns
    -------
    None
        The summary is printed to standard output in YAML format.
    """

    args = _parse_args()
    cfg = OmegaConf.load(resolve_config_path(args.config, CONFIG_DIR))
    dotlist = [override for override in args.forwarded if "=" in override]
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    summary = run(cfg, forwarded_overrides=args.forwarded, run_id=args.run_id, output_root=args.output_root)
    print(OmegaConf.to_yaml(summary))


def run(
    cfg: DictConfig,
    *,
    forwarded_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Write reference or baseline artifacts.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Reference metadata config.
    forwarded_overrides : list of str or None, optional
        CLI overrides to record.
    run_id : str or None, optional
        Run id override.
    output_root : str, pathlib.Path, or None, optional
        Output root override.

    Returns
    -------
    dict
        Reference placeholder summary.
    """

    run_time = str(OmegaConf.select(cfg, "run.time", default=run_time_stamp()))
    selected_run_id = run_id or configured_run_id(cfg)
    if selected_run_id is None:
        selected_run_id = make_run_id("hooke_multibody_reference", run_time=run_time)
    selected_root = Path(str(output_root or OmegaConf.select(cfg, "output_root", default="outputs")))
    cfg = OmegaConf.merge(
        cfg,
        {"run": {"time": run_time}, "run_id": selected_run_id, "output_root": str(selected_root)},
    )
    OmegaConf.resolve(cfg)
    output_dir = make_output_dir(
        selected_root,
        run_name="hooke_multibody_reference",
        run_id=selected_run_id,
        include_plots=False,
    )
    write_config_artifacts(output_dir, cfg, forwarded_overrides or [])
    row, tables = _reference_outputs(cfg)
    write_csv(output_dir / "data" / "reference_observables.csv", [row])
    for name, rows in tables.items():
        write_csv(output_dir / "data" / name, rows)
    summary = {
        "entrypoint": "experiments/hooke_multibody/run_reference.py",
        "status": "ok",
        "run_id": selected_run_id,
        "run_time": run_time,
        "output_dir": str(output_dir),
        "git": git_metadata(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "reference_available": row["reference_available"],
        "method": row["method"],
        "baseline_available": row.get("baseline_available", False),
        "baseline_method": row.get("baseline_method", ""),
        "baseline_energy": row.get("baseline_energy", ""),
    }
    write_json(output_dir / "artifacts" / "summary.json", summary)
    return summary


def _reference_outputs(cfg: DictConfig) -> tuple[dict[str, object], dict[str, list[dict[str, float]]]]:
    method = str(OmegaConf.select(cfg, "reference.method", default="none"))
    n_electrons = int(OmegaConf.select(cfg, "system.n_electrons", default=3))
    n_up = int(OmegaConf.select(cfg, "system.n_up", default=2))
    n_down = int(OmegaConf.select(cfg, "system.n_down", default=1))
    spatial_dim = int(OmegaConf.select(cfg, "system.spatial_dim", default=3))
    harmonic_omega = float(OmegaConf.select(cfg, "system.harmonic_omega", default=0.5))
    row: dict[str, object] = {
        "reference_available": bool(OmegaConf.select(cfg, "reference.available", default=False)),
        "method": method,
        "n_electrons": n_electrons,
        "n_up": n_up,
        "n_down": n_down,
        "spatial_dim": spatial_dim,
        "harmonic_omega": harmonic_omega,
        "baseline_available": False,
        "baseline_method": "",
        "baseline_high_accuracy": False,
        "baseline_energy": "",
    }
    if method == "none":
        return row, {}
    if method != "gaussian_hartree_variational":
        raise ValueError(f"Unsupported Hooke multibody reference method: {method!r}")
    alpha = OmegaConf.select(cfg, "reference.gaussian.alpha", default=None)
    baseline = gaussian_hartree_reference(
        n_electrons=n_electrons,
        n_up=n_up,
        n_down=n_down,
        harmonic_omega=harmonic_omega,
        spatial_dim=spatial_dim,
        alpha=None if alpha is None else float(alpha),
    )
    row.update(baseline.as_row())
    bins = int(OmegaConf.select(cfg, "reference.gaussian.bins", default=64))
    r_max = float(OmegaConf.select(cfg, "reference.gaussian.r_max", default=6.0))
    return row, {
        "reference_radial_density.csv": gaussian_radial_density_rows(alpha=baseline.alpha, bins=bins, r_max=r_max),
        "reference_pair_distance_density.csv": gaussian_pair_distance_density_rows(
            alpha=baseline.alpha,
            bins=bins,
            r_max=r_max,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Reference config path or name.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("forwarded", nargs="*", help="Dotlist overrides recorded in artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
