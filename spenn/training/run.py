"""Generic train/evaluate execution from resolved configs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from spenn.data.batch import ElectronBatch, Walkers
from spenn.diagnostics.base import DiagnosticContext, DiagnosticResult
from spenn.diagnostics.wavefunction import pair_distance
from spenn.training.artifacts import (
    git_metadata,
    make_output_dir,
    make_run_id,
    normalize_rows,
    write_config_artifacts,
    write_csv,
    write_json,
)


def run_config(cfg: DictConfig, *, forwarded_overrides: list[str] | None = None) -> dict[str, object]:
    """Run a train or evaluate job from a Hydra-style config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Run configuration.
    forwarded_overrides : list of str or None, optional
        CLI dotlist overrides to record in artifacts.

    Returns
    -------
    dict
        JSON-serializable run summary.
    """

    cfg = _prepare_config(cfg)
    dtype = _resolve_dtype(str(cfg.get("dtype", "float64")))
    torch.manual_seed(int(cfg.get("seed", 0)))
    device = _resolve_device(str(cfg.get("device", "cpu")))
    mode = str(OmegaConf.select(cfg, "run.mode", default="train"))
    run_id = str(OmegaConf.select(cfg, "run.id", default=cfg.get("run_id")))
    output_root = Path(str(OmegaConf.select(cfg, "run.output_root", default=cfg.get("output_root", "outputs"))))
    run_name = str(OmegaConf.select(cfg, "run.name", default=cfg.get("experiment_name", "spenn_run")))
    write_plot_data = bool(OmegaConf.select(cfg, "artifacts.write_plot_data", default=True))
    output_dir = make_output_dir(output_root, run_name=run_name, run_id=run_id, include_plots=write_plot_data)
    write_config_artifacts(output_dir, cfg, forwarded_overrides or [])

    system = instantiate(cfg.system).to(device=device, dtype=dtype)
    model = instantiate(_model_cfg(cfg)).to(device=device, dtype=dtype)
    hamiltonian = instantiate(cfg.hamiltonian, _partial_=True)(system=system)

    train_rows: list[dict[str, object]] = []
    if mode == "train":
        train_rows = _train(cfg, model=model, hamiltonian=hamiltonian, system=system, dtype=dtype, device=device)
    elif mode != "evaluate":
        raise ValueError(f"Unsupported run.mode: {mode!r}")

    production = _production(cfg, model=model, hamiltonian=hamiltonian, system=system, dtype=dtype, device=device)
    final_metrics = dict(production["metrics"])
    tables: dict[str, list[dict[str, object]]] = dict(production["tables"])
    context = DiagnosticContext(
        cfg=cfg,
        model=model,
        hamiltonian=hamiltonian,
        system=system,
        sampler=production["sampler"],
        walkers=production["walkers"],
        local_energy=production["local_energy"],
        pair_distance=production["pair_distance"],
        dtype=dtype,
        device=device,
    )
    for diagnostic in _diagnostics(cfg):
        result = _call_diagnostic(diagnostic, context)
        final_metrics.update(result.metrics)
        tables.update(result.tables)

    write_csv(output_dir / "metrics" / "energy_trace.csv", normalize_rows(production["energy_rows"]))
    write_csv(output_dir / "metrics" / "train_metrics.csv", normalize_rows(train_rows) if train_rows else [final_metrics])
    write_csv(output_dir / "metrics" / "sampler_metrics.csv", [_sampler_metrics(final_metrics)])
    if any(key.startswith("comparison/") for key in final_metrics):
        write_csv(output_dir / "metrics" / "comparison_metrics.csv", [final_metrics])
    if write_plot_data:
        for name, rows in tables.items():
            write_csv(output_dir / "plots" / f"{name}.csv", rows)
    if bool(OmegaConf.select(cfg, "artifacts.write_checkpoint", default=True)):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "exact_energy": getattr(system, "exact_energy", None),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            output_dir / "checkpoints" / "final_model.pt",
        )
    summary = {
        "entrypoint": "scripts/train.py",
        "status": "ok",
        "mode": mode,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "git": git_metadata(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": final_metrics,
    }
    write_json(output_dir / "artifacts" / "summary.json", summary)
    return {
        "entrypoint": "scripts/train.py",
        "status": "ok",
        "mode": mode,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "final_metrics": final_metrics,
    }


def _prepare_config(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    run_id = OmegaConf.select(cfg, "run_id", default=None)
    if run_id is None:
        run_id = OmegaConf.select(cfg, "run.id", default=None)
    if run_id is None:
        run_id = make_run_id(str(OmegaConf.select(cfg, "run.id_prefix", default="run")))
    output_root = OmegaConf.select(cfg, "output_root", default=None)
    if output_root is None:
        output_root = OmegaConf.select(cfg, "run.output_root", default=None)
    if output_root is None:
        output_root = "outputs"
    cfg = OmegaConf.merge(cfg, {"run_id": str(run_id), "output_root": str(output_root)})
    OmegaConf.resolve(cfg)
    return cfg


def _train(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    hamiltonian: Any,
    system: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, object]]:
    sampler = _instantiate_sampler(cfg, dtype=dtype)
    optimizer_cfg = OmegaConf.create(OmegaConf.to_container(cfg.optimizer, resolve=True))
    grad_clip = OmegaConf.select(cfg, "trainer.grad_clip", default=OmegaConf.select(cfg, "optimizer.grad_clip", default=None))
    optimizer_cfg.pop("grad_clip", None)
    optimizer = instantiate(optimizer_cfg, _partial_=True)(params=model.parameters())
    trainer = instantiate(cfg.trainer, _partial_=True)(
        model=model,
        sampler=sampler,
        hamiltonian=hamiltonian,
        loss=instantiate(cfg.loss),
        optimizer=optimizer,
        system=system,
        grad_clip=None if grad_clip is None else float(grad_clip),
        device=device,
    )
    rows = []
    exact_energy = getattr(system, "exact_energy", None)
    for step, metrics in enumerate(trainer.fit()):
        row = {"step": step, "training/vmc_step": step}
        row.update(_to_scalar_dict(metrics))
        if "energy" in row:
            row["spenn/energy/mean"] = row["energy"]
        if "variance" in row:
            row["spenn/local_energy/variance"] = row["variance"]
        if "objective" in row:
            row["training/objective"] = row["objective"]
        if "acceptance_rate" in row:
            row["sampler/acceptance_rate"] = row["acceptance_rate"]
        if "grad_norm" in row:
            row["grad/norm"] = row["grad_norm"]
        if exact_energy is not None and "energy" in row:
            row["exact/energy"] = float(exact_energy)
            row["comparison/energy_error"] = float(row["energy"]) - float(exact_energy)
            row["comparison/energy_abs_error"] = abs(float(row["energy"]) - float(exact_energy))
        rows.append(row)
    return rows


def _production(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    hamiltonian: Any,
    system: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    sampler = _instantiate_sampler(cfg, dtype=dtype)
    walkers = sampler.initialize(system=system, n_walkers=int(cfg.sampler.n_walkers), device=device)
    walkers = sampler.sample(model, walkers, int(cfg.sampler.warmup_steps))
    blocks = int(OmegaConf.select(cfg, "sampler.production_blocks", default=1))
    steps_per_block = int(OmegaConf.select(cfg, "sampler.steps_per_block", default=getattr(sampler, "steps_per_iter", 1)))
    energy_rows: list[dict[str, object]] = []
    all_local_energy: list[torch.Tensor] = []
    all_r12: list[torch.Tensor] = []
    for block in range(blocks):
        walkers = sampler.sample(model, walkers, steps_per_block)
        local_energy = _local_energy(model, hamiltonian, walkers).detach()
        r12 = pair_distance(walkers.positions).detach()
        all_local_energy.append(local_energy.cpu())
        all_r12.append(r12.cpu())
        energy_rows.append(_energy_row(cfg, block, local_energy, exact_energy=getattr(system, "exact_energy", None), sampler=sampler, r12=r12))
    local_energy = torch.cat(all_local_energy)
    r12 = torch.cat(all_r12)
    return {
        "sampler": sampler,
        "walkers": walkers,
        "local_energy": local_energy,
        "pair_distance": r12,
        "energy_rows": energy_rows,
        "metrics": _final_metrics(cfg, local_energy, r12, exact_energy=getattr(system, "exact_energy", None), sampler=sampler),
        "tables": {},
    }


def _final_metrics(
    cfg: DictConfig,
    local_energy: torch.Tensor,
    r12: torch.Tensor,
    *,
    exact_energy: float | None,
    sampler: Any,
) -> dict[str, float]:
    mean = local_energy.mean()
    std = local_energy.std(unbiased=False)
    variance = local_energy.var(unbiased=False)
    prefix = str(OmegaConf.select(cfg, "run.energy_prefix", default="spenn" if str(OmegaConf.select(cfg, "run.mode", default="train")) == "train" else "energy"))
    metrics = {
        "sampler/acceptance_rate": float(getattr(sampler, "acceptance_rate", 0.0)),
        "sampler/proposal_scale": float(getattr(getattr(sampler, "move", None), "step_size", float("nan"))),
        "sampler/mean_r12": float(r12.mean().item()),
        "sampler/std_r12": float(r12.std(unbiased=False).item()),
        "sampler/equilibration_steps": float(OmegaConf.select(cfg, "sampler.warmup_steps", default=0)),
    }
    if prefix == "spenn":
        metrics.update(
            {
                "spenn/energy/mean": float(mean.item()),
                "spenn/energy/std": float(std.item()),
                "spenn/energy/sem": float((std / math.sqrt(local_energy.numel())).item()),
                "spenn/local_energy/variance": float(variance.item()),
            }
        )
        if exact_energy is not None:
            metrics["exact/energy"] = float(exact_energy)
            metrics["comparison/energy_error"] = float((mean - float(exact_energy)).item())
            metrics["comparison/energy_abs_error"] = float((mean - float(exact_energy)).abs().item())
    else:
        metrics.update(
            {
                "energy/mean": float(mean.item()),
                "energy/std": float(std.item()),
                "energy/sem": float((std / math.sqrt(local_energy.numel())).item()),
                "local_energy/variance": float(variance.item()),
            }
        )
        if exact_energy is not None:
            metrics["energy/exact"] = float(exact_energy)
            metrics["energy/error"] = float((mean - float(exact_energy)).item())
            metrics["energy/abs_error"] = float((mean - float(exact_energy)).abs().item())
    return metrics


def _energy_row(
    cfg: DictConfig,
    block: int,
    local_energy: torch.Tensor,
    *,
    exact_energy: float | None,
    sampler: Any,
    r12: torch.Tensor,
) -> dict[str, object]:
    metrics = _final_metrics(cfg, local_energy, r12, exact_energy=exact_energy, sampler=sampler)
    return {"step": block, **metrics}


def _local_energy(model: torch.nn.Module, hamiltonian: Any, walkers: Walkers) -> torch.Tensor:
    batch = ElectronBatch(positions=walkers.positions, system=walkers.aux.get("system"), spins=walkers.spins)
    local_energy = hamiltonian.local_energy(model, batch)
    if local_energy.shape != (walkers.batch_size,):
        raise ValueError(f"local energy must have shape [{walkers.batch_size}], got {tuple(local_energy.shape)}")
    return local_energy


def _instantiate_sampler(cfg: DictConfig, *, dtype: torch.dtype) -> Any:
    sampler = instantiate(cfg.sampler)
    if hasattr(sampler, "dtype"):
        sampler.dtype = dtype
    return sampler


def _diagnostics(cfg: DictConfig) -> list[Any]:
    diagnostics_cfg = OmegaConf.select(cfg, "diagnostics", default=None)
    if diagnostics_cfg is None:
        return []
    diagnostics = []
    if isinstance(diagnostics_cfg, ListConfig):
        iterable = diagnostics_cfg
    else:
        iterable = diagnostics_cfg.values()
    for item in iterable:
        if isinstance(item, DictConfig) and "_target_" in item:
            diagnostics.append(instantiate(item))
    return diagnostics


def _call_diagnostic(diagnostic: Any, context: DiagnosticContext) -> DiagnosticResult:
    result = diagnostic(context)
    if isinstance(result, DiagnosticResult):
        return result
    if isinstance(result, dict):
        return DiagnosticResult(metrics={key: float(value) for key, value in result.items()})
    raise TypeError(f"Diagnostic must return DiagnosticResult or dict, got {type(result).__name__}")


def _model_cfg(cfg: DictConfig) -> DictConfig:
    path = OmegaConf.select(cfg, "run.model_path", default="model")
    selected = OmegaConf.select(cfg, str(path), default=None)
    if selected is None:
        raise KeyError(f"Missing model config at {path!r}")
    return selected


def _sampler_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if key.startswith("sampler/")}


def _to_scalar_dict(metrics: dict[str, Any]) -> dict[str, object]:
    converted = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            converted[key] = float(value.item())
        else:
            converted[key] = value
    return converted


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().replace("torch.", "")
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name!r}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)
