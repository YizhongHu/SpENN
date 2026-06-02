"""Generate and run SpENN Hooke benchmark configs."""

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
DEFAULT_CONFIG = CONFIG_DIR / "triplet_spenn.yaml"
DEFAULT_RUN_ID_PREFIX = "hooke_spenn"
TRAIN_ENTRYPOINT = ROOT / "train.py"


def main() -> None:
    """Run a SpENN Hooke benchmark through the generic train script.

    Returns
    -------
    None
        The Hooke summary is printed to standard output in YAML format.
    """

    args = _parse_args()
    summary = run(
        OmegaConf.load(resolve_config_path(args.config, CONFIG_DIR)),
        forwarded_overrides=args.forwarded,
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
    """Generate a config, run SpENN training, and summarize the result.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        SpENN Hooke config template from ``experiments/hooke/configs``.
    forwarded_overrides : list of str or None, optional
        Dotlist overrides recorded and forwarded to ``train.py``.
    run_id : str or None, optional
        Run id override. Stored as the top-level ``run_id`` config value.
    output_root : str, pathlib.Path, or None, optional
        Output root override. Stored as the top-level ``output_root`` value.

    Returns
    -------
    dict
        Hooke SpENN summary.
    """

    generic = run_generated_config(
        cfg,
        HookeScriptSpec(
            entrypoint=TRAIN_ENTRYPOINT,
            generated_subdir="hooke_spenn",
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
    validation = cfg.get("validation", {})
    reached = bool(
        metrics["comparison/energy_abs_error"]
        <= float(validation.get("energy_tolerance", validation.get("comparison_energy_abs_error_tolerance", 10.0)))
        and metrics["comparison/radial_logabs_rmse"]
        <= float(validation.get("radial_logabs_rmse_tolerance", validation.get("comparison_logabs_mae_tolerance", 10.0)))
        and abs(metrics["comparison/cusp_slope_error"]) <= float(validation.get("cusp_slope_tolerance", 10.0))
        and metrics["comparison/sign_alignment_accuracy"] >= float(validation.get("sign_alignment_min", 0.0))
        and metrics["spenn/local_energy/variance"] <= float(validation.get("local_energy_variance_tolerance", float("inf")))
        and _exchange_reached(metrics, validation)
    )
    return {
        "entrypoint": "experiments/hooke/run_spenn.py",
        "status": "ok",
        "can_reach_goal": reached,
        "run_id": artifact["run_id"],
        "output_dir": artifact["output_dir"],
        "sector": cfg["sector"],
        "energy_mean": metrics["spenn/energy/mean"],
        "energy_exact": metrics["exact/energy"],
        "energy_abs_error": metrics["comparison/energy_abs_error"],
        "radial_logabs_rmse": metrics["comparison/radial_logabs_rmse"],
        "cusp_slope_error": metrics["comparison/cusp_slope_error"],
        "sign_alignment_accuracy": metrics["comparison/sign_alignment_accuracy"],
        "acceptance_rate": metrics["sampler/acceptance_rate"],
    }


def _exchange_reached(metrics: dict[str, float], validation: dict[str, object]) -> bool:
    tolerance = float(validation.get("exchange_error_tolerance", 1.0e-8))
    expected = str(validation.get("exchange_mode", "particle_antisymmetric"))
    if expected == "spatial_singlet":
        return (
            float(metrics["comparison/symmetry_error_max"]) <= tolerance
            and float(metrics.get("comparison/sign_match_accuracy", 1.0))
            >= float(validation.get("sign_match_min", 1.0))
        )
    if expected != "particle_antisymmetric":
        raise ValueError(f"Unsupported validation.exchange_mode: {expected!r}")
    return (
        float(metrics["comparison/antisymmetry_error_max"]) <= tolerance
        and float(metrics["comparison/sign_flip_accuracy"]) >= float(validation.get("sign_flip_min", 1.0))
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Hooke config template path or name.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("forwarded", nargs="*", help="Dotlist overrides recorded and forwarded to train.py.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
