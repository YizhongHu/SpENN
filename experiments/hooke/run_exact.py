"""Generate and run exact Hooke benchmark configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hooke.runner import HookeScriptSpec, resolve_config_path, run_generated_config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "debug_singlet.yaml"
DEFAULT_RUN_ID_PREFIX = "hooke_exact"
TRAIN_ENTRYPOINT = ROOT / "scripts" / "train.py"


def main() -> None:
    """Run an exact Hooke benchmark through the generic train script.

    Returns
    -------
    None
        The Hooke summary is printed to standard output in YAML format.
    """

    args = _parse_args()
    forwarded = list(args.forwarded)
    if args.sector is not None:
        forwarded.append(f"sector={args.sector}")
    summary = run(
        OmegaConf.load(resolve_config_path(args.config, CONFIG_DIR)),
        forwarded_overrides=forwarded,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    print(OmegaConf.to_yaml(summary))


def run(
    cfg: DictConfig,
    *,
    forwarded_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Generate a config, run exact diagnostics, and summarize the result.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Exact Hooke config template from ``experiments/hooke/configs``.
    forwarded_overrides : list of str or None, optional
        Dotlist overrides recorded and forwarded to ``scripts/train.py``.
    run_id : str or None, optional
        Run id override. Stored as the top-level ``run_id`` config value.
    output_root : str, pathlib.Path, or None, optional
        Output root override. Stored as the top-level ``output_root`` value.

    Returns
    -------
    dict
        Hooke exact summary.
    """

    generic = run_generated_config(
        cfg,
        HookeScriptSpec(
            entrypoint=TRAIN_ENTRYPOINT,
            generated_subdir="hooke_exact",
            run_id_prefix=DEFAULT_RUN_ID_PREFIX,
        ),
        run_id=run_id,
        output_root=output_root,
        forwarded_overrides=forwarded_overrides,
    )
    return _summary_from_artifact(generic["output_dir"])


def _summary_from_artifact(output_dir: str | Path) -> dict[str, object]:
    with (Path(output_dir) / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    cfg = artifact["config"]
    metrics = artifact["metrics"]
    tolerance = float(cfg.get("validation", {}).get("sample_energy_tolerance", float("inf")))
    preflight_tolerance = float(cfg.get("validation", {}).get("preflight_energy_tolerance", float("inf")))
    preflight_error = float(metrics.get("preflight/max_abs_error", 0.0))
    energy_abs_error = float(metrics["energy/abs_error"])
    reached = bool(energy_abs_error < tolerance and preflight_error < preflight_tolerance)
    return {
        "entrypoint": "experiments/hooke/run_exact.py",
        "status": "ok",
        "can_reach_goal": reached,
        "run_id": artifact["run_id"],
        "output_dir": artifact["output_dir"],
        "sector": cfg["sector"],
        "energy_mean": metrics["energy/mean"],
        "energy_exact": metrics["energy/exact"],
        "energy_abs_error": energy_abs_error,
        "acceptance_rate": metrics["sampler/acceptance_rate"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Hooke config template path or name.")
    parser.add_argument("--sector", choices=("singlet", "triplet"), default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("forwarded", nargs="*", help="Dotlist overrides recorded and forwarded to scripts/train.py.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
