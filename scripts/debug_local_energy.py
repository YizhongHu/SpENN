"""Phase 1 local-energy diagnostic entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.data.batch import ElectronBatch


def load_config(argv: list[str] | None = None):
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="config", overrides=argv or sys.argv[1:])


def main() -> None:
    cfg = load_config()
    dtype = getattr(torch, str(cfg.get("dtype", "float64")))
    torch.manual_seed(int(cfg.get("seed", 0)))
    device_name = str(cfg.get("device", "cpu"))
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    system = instantiate(cfg.system).to(device=device, dtype=dtype)
    model = instantiate(cfg.model).to(device=device, dtype=dtype)
    hamiltonian = instantiate(cfg.hamiltonian, _partial_=True)(system=system)
    sampler = instantiate(cfg.sampler)
    loss_fn = instantiate(cfg.loss)
    walkers = sampler.initialize(system=system, device=device)
    batch = walkers.to(device=device, dtype=dtype)
    loss, metrics = loss_fn(
        model,
        hamiltonian,
        ElectronBatch(positions=batch.positions, system=batch.aux.get("system")),
    )
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "debug_local_energy",
                "status": "ok",
                "loss": float(loss.item()),
                **{k: float(v.item()) for k, v in metrics.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
