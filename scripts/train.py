"""Phase 1 VMC training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _to_scalar_dict(metrics: dict) -> dict:
    converted = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            converted[key] = float(value.item())
        else:
            converted[key] = value
    return converted


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
    loss = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, _partial_=True)(params=model.parameters())
    trainer = instantiate(cfg.trainer, _partial_=True)(
        model=model,
        sampler=sampler,
        hamiltonian=hamiltonian,
        loss=loss,
        optimizer=optimizer,
        system=system,
        device=device,
    )
    history = trainer.fit()
    print(OmegaConf.to_yaml({"entrypoint": "train", "status": "ok", "final_metrics": _to_scalar_dict(history[-1]) if history else {}}))


if __name__ == "__main__":
    main()
