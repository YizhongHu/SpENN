#!/usr/bin/env python3
"""C2 precision witness: upstream fp32 eigendecomposition on a synthetic spectrum.

Upstream compute_a_inv/compute_g_inv unconditionally cast factors to float32
before torch.linalg.eigh, regardless of factor_dtype/inv_dtype float64
(kfac/layers/eigen.py at pin 5987766a). This arm measures whether that matters
at scientific magnitude, not merely as a dtype mismatch: an ill-conditioned
A factor with spectrum logspace(0, -8) (controlled condition number 1e8),
gradients aligned with small-eigenvalue
directions, and damping swept over 1e-2, 1e-3, 1e-4 relative to ||A|| = 1.

Reported: relative error of the preconditioned gradient produced through the
upstream path versus an all-float64 reference of the SAME eigen-decomposition
formula. The reference reproduces upstream's algorithm exactly (eigh, clamp,
outer-product damping) in float64, so every discrepancy is attributable to the
float32 downcast alone, not to algorithmic differences.

Round-1 corrections applied per fixing-side disposition:
- Eigenvalue lists are reported in ASCENDING order on BOTH sides and paired
  elementwise, with a per-pair relative error (the round-1 receipt paired a
  descending fp64 list against an ascending upstream list).
- An INDEPENDENT float64 control is added: for G = I the preconditioned
  gradient equals grad @ (A + damping I)^-1, computed by torch.linalg.solve.
  The reference-vs-solve error is now MEASURED, not asserted, so the
  reference path's own error floor is on record.
- The spectrum is SYNTHETIC: logspace(0, -8) is a stand-in for ill-conditioned
  curvature, chosen for a controlled condition number of 1e8; it is not a
  measured TPEN Fisher/QGT spectrum.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools"))
    from kfac_compatibility_probe import check_upstream, git

    review_tip = git(repo, "rev-parse", "HEAD")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("review checkout has tracked changes")
    upstream = check_upstream(Path(os.environ["P2_KFAC_SOURCE"]).resolve())
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    torch.manual_seed(0)
    torch.set_num_threads(1)

    n = 32
    module = torch.nn.Linear(n, 1, bias=False).double()
    helper = LinearModuleHelper(module)

    # Symmetric A with spectrum logspace(0, -8): condition number 1e8.
    generator = torch.Generator().manual_seed(1234)
    raw = torch.randn(n, n, dtype=torch.float64, generator=generator)
    q_basis, _ = torch.linalg.qr(raw)
    eigenvalues = torch.logspace(0, -8, n, dtype=torch.float64)
    a_factor = q_basis @ torch.diag(eigenvalues) @ q_basis.T
    a_factor = (a_factor + a_factor.T) / 2
    g_factor = torch.eye(1, dtype=torch.float64)

    # Gradient aligned with the three smallest-eigenvalue directions — the
    # directions a VMC preconditioner exists to amplify.
    grad = (q_basis[:, -1] + q_basis[:, -2] + q_basis[:, -3]).reshape(1, n)

    receipt = {
        "witness": "p2-c2-fp32-eigh-precision",
        "revision": "round-2: ascending eigenvalue pairing, measured float64 solve control",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "review_tip": review_tip,
        "upstream": upstream,
        "python": sys.executable,
        "torch": torch.__version__,
        "n": n,
        "condition_number": 1e8,
        "spectrum": "SYNTHETIC logspace(0,-8); not a measured TPEN Fisher/QGT spectrum",
        "sweeps": {},
    }

    for damping in (1e-2, 1e-3, 1e-4):
        layer = KFACEigenLayer(
            helper, tdc=TorchDistributedCommunicator(),
            inv_dtype=torch.float64, factor_dtype=torch.float64,
        )
        layer.a_factor = a_factor.clone()
        layer.g_factor = g_factor.clone()
        with torch.no_grad():
            module.weight.grad = grad.clone()
        layer.compute_a_inv(damping=damping)   # upstream: eigh in float32
        layer.compute_g_inv(damping=damping)
        layer.preconditioned_grad(damping=damping)
        upstream_update = layer.grad.clone()

        # All-float64 reference of the identical algorithm.
        da_ref, qa_ref = torch.linalg.eigh(a_factor)
        da_ref = torch.clamp(da_ref, min=0.0)
        dg_ref, qg_ref = torch.linalg.eigh(g_factor)
        dg_ref = torch.clamp(dg_ref, min=0.0)
        v1 = qg_ref.T @ grad @ qa_ref
        v2 = v1 / (torch.outer(dg_ref, da_ref) + damping)
        reference_update = qg_ref @ v2 @ qa_ref.T

        # Independent float64 control: with G = I the preconditioned gradient
        # is grad @ (A + damping I)^-1, by direct solve — no eigh involved.
        identity = torch.eye(n, dtype=torch.float64)
        solve_update = torch.linalg.solve(
            (a_factor + damping * identity).T, grad.T,
        ).T

        absolute = (upstream_update - reference_update).norm().item()
        relative = absolute / reference_update.norm().item()
        ref_vs_solve = ((reference_update - solve_update).norm()
                        / solve_update.norm()).item()
        upstream_vs_solve = ((upstream_update - solve_update).norm()
                             / solve_update.norm()).item()

        # Ascending on BOTH sides, paired elementwise (round-1 correction).
        fp64_ascending = torch.sort(eigenvalues)[0][:3]
        upstream_ascending = torch.sort(layer.da)[0][:3]
        eigenvalue_pairs = [
            {
                "fp64": fp64_value.item(),
                "upstream_fp32_path": upstream_value.item(),
                "relative_error": abs(upstream_value.item() - fp64_value.item())
                / fp64_value.item(),
            }
            for fp64_value, upstream_value in zip(fp64_ascending,
                                                  upstream_ascending)
        ]

        receipt["sweeps"][f"damping={damping}"] = {
            "relative_error_of_preconditioned_grad": relative,
            "absolute_error": absolute,
            "reference_update_norm": reference_update.norm().item(),
            "upstream_update_norm": upstream_update.norm().item(),
            "reference_vs_float64_solve_relative_error": ref_vs_solve,
            "upstream_vs_float64_solve_relative_error": upstream_vs_solve,
            "smallest_three_eigenvalue_pairs_ascending": eigenvalue_pairs,
        }

    # Preserve the reviewer's pre-declared diagnostic threshold. This does not
    # select TPEN's C2 oracle tolerance or establish its observed conditioning.
    receipt["fp32_eps"] = 1.1920929e-07
    receipt["materiality_threshold_relative_error"] = 1e-6
    receipt["threshold_scope"] = "synthetic diagnostic, not an operator-frozen TPEN tolerance"
    receipt["scientifically_material"] = any(
        sweep["relative_error_of_preconditioned_grad"] > 1e-6
        for sweep in receipt["sweeps"].values()
    )

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
