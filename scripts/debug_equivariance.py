"""Phase 1 equivariance diagnostic entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.data_structures.batch import ElectronBatch


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
    encoder = instantiate(cfg.model.encoder).to(device=device, dtype=dtype)
    sampler = instantiate(cfg.sampler)
    walkers = sampler.initialize(system=system, device=device)
    batch = ElectronBatch(positions=walkers.positions, spins=walkers.spins, system=walkers.aux.get("system"))
    flipped = ElectronBatch(
        positions=batch.positions.flip(dims=(1,)),
        spins=None if batch.spins is None else batch.spins.flip(dims=(1,)),
        system=batch.system,
    )
    original = encoder(batch)
    permuted = encoder(flipped)
    original_s = original.get(2, (2))
    original_a = original.get(2, (1, 1))
    permuted_s = permuted.get(2, (2)).flip(dims=(1, 2))
    permuted_a = permuted.get(2, (1, 1)).flip(dims=(1, 2))
    ok = (
        torch.allclose(original_s, permuted_s)
        and torch.allclose(original_a, permuted_a)
        and torch.allclose(original_s, original_s.transpose(1, 2))
        and torch.allclose(original_a, -original_a.transpose(1, 2))
    )
    print(OmegaConf.to_yaml({"entrypoint": "debug_equivariance", "status": "ok" if ok else "failed"}))


if __name__ == "__main__":
    main()
