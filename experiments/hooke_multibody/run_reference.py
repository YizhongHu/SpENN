"""Write Hooke multibody reference metadata artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hooke.runner import resolve_config_path  # noqa: E402
from spenn.training.artifacts import (  # noqa: E402
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
    """Write a reference-placeholder run and print its summary.

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
    """Write metadata-only reference artifacts.

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
    selected_run_id = run_id or str(OmegaConf.select(cfg, "run_id", default=""))
    if not selected_run_id or selected_run_id == "None":
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
    row = {
        "reference_available": bool(OmegaConf.select(cfg, "reference.available", default=False)),
        "method": str(OmegaConf.select(cfg, "reference.method", default="none")),
        "n_electrons": int(OmegaConf.select(cfg, "system.n_electrons", default=3)),
        "n_up": int(OmegaConf.select(cfg, "system.n_up", default=2)),
        "n_down": int(OmegaConf.select(cfg, "system.n_down", default=1)),
        "harmonic_omega": float(OmegaConf.select(cfg, "system.harmonic_omega", default=0.5)),
    }
    write_csv(output_dir / "data" / "reference_observables.csv", [row])
    summary = {
        "entrypoint": "experiments/hooke_multibody/run_reference.py",
        "status": "ok",
        "run_id": selected_run_id,
        "run_time": run_time,
        "output_dir": str(output_dir),
        "reference_available": row["reference_available"],
        "method": row["method"],
    }
    write_json(output_dir / "artifacts" / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Reference config path or name.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("forwarded", nargs="*", help="Dotlist overrides recorded in artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
