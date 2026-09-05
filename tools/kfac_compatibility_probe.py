#!/usr/bin/env python3
"""Diagnose the pinned KFAC candidate; this probe cannot admit an adapter.

Run in a cluster allocation with an already provisioned TPEN/KFAC environment.
No installation, training integration, private upstream access, or DDP launch is
performed. JSON goes to stdout. Exit 1 means a completed negative probe; exit 2
means missing provenance/environment or inconclusive evidence. Neither is a pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

UPSTREAM_SHA = "5987766a43739de7eb950f564da54559f2504579"
REPO = Path(__file__).resolve().parents[1]
UNESTABLISHED = "UNESTABLISHED-pending-reviewer-cluster-evidence"


def criterion_verdicts() -> dict:
    """Keep full compatibility distinct from this probe's narrower diagnostics.

    Even successful single-process controls cannot establish the required
    whole-TPEN extension and multi-rank contracts. Blocker metadata distinguishes
    operator decisions and qualification work from reviewer execution; this
    diagnostic never converts pending evidence into a pass.
    """
    obligations = {
        "C1": "Complete named TPEN Kronecker/exact-scalar coverage and trace extraction",
        "C2": "Operator-frozen VMC convention and all-family dense score/factor oracles",
        "C3": "Sum-plus-count factors for unequal/empty shards and global-empty rejection under real multi-rank execution",
        "C4": "Every required TPEN registration, factor and state path qualified through public APIs",
        "C5": "Exact complete-state TPEN restart across factor and inverse refresh under real multi-rank execution",
    }
    blockers = {
        "C1": ["pre_adapter_qualification", "reviewer_cluster_evidence"],
        "C2": ["operator_decision", "pre_adapter_qualification", "reviewer_cluster_evidence"],
        "C3": ["pre_adapter_qualification", "reviewer_cluster_evidence"],
        "C4": ["pre_adapter_qualification", "reviewer_cluster_evidence"],
        "C5": ["pre_adapter_qualification", "reviewer_cluster_evidence"],
    }
    return {key: {"state": UNESTABLISHED, "missing_evidence": obligation,
                  "blocking_parties": blockers[key]}
            for key, obligation in obligations.items()}


def git(path: Path, *args: str) -> str:
    """Read Git provenance without changing the checkout."""
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def check_upstream(source: Path) -> dict:
    """Require the inspected pin and byte-identical imported Python sources."""
    if git(source, "rev-parse", "HEAD") != UPSTREAM_SHA:
        raise ValueError("upstream checkout is not the P2 pin")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("upstream tracked source is dirty")
    import kfac

    imported = Path(kfac.__file__).resolve().parent
    expected_files = {
        Path(name).relative_to("kfac")
        for name in git(source, "ls-files", "kfac").splitlines()
        if name.endswith(".py")
    }
    if {p.relative_to(imported) for p in imported.rglob("*.py")} != expected_files:
        raise ValueError("imported KFAC Python file inventory differs from the pin")
    digest = hashlib.sha256()
    for relative in sorted(expected_files):
        expected = (source / "kfac" / relative).read_bytes()
        actual = (imported / relative).read_bytes()
        if actual != expected:
            raise ValueError(f"imported KFAC source differs: {relative}")
        digest.update(str(relative).encode() + b"\0" + actual)
    return {"sha": UPSTREAM_SHA, "version": kfac.__version__,
            "python_source_sha256": digest.hexdigest(), "imported_from": str(imported)}


def representative_model(seeded: bool):
    """Build a whole TPEN parameter census with both mixing families active."""
    import torch
    from tpen.data.atomic_configuration import AtomicConfiguration
    from tpen.data.paths import (
        LinearPathMetadata, NormalizedChannels, NormalizedOrders,
        PathMetadata, compose_path_layout,
    )
    from tpen.nn import (
        CompositeMixing, CurvatureElectronNucleusCuspLaw,
        ElectronElectronCusp, ElectronNucleusCusp, Embedding, EquivariantMixing,
        GaussianConfinement, LinearEquivariantMixing,
        TPENLayer, TPENWaveFunction, TorchInitializer,
    )
    from tpen.nn.readout.pfaffian import PfaffianReadout
    from tpen.nn.update import ChannelMappedUpdater

    linear = LinearPathMetadata.generate(max_order=2)
    tensor = PathMetadata.generate(max_order=2, max_virtual_order=2, output_embedding="canonical")
    layout = compose_path_layout(
        linear=linear, tensor_product=tensor,
        input_orders=NormalizedOrders((1, 2)), output_orders=NormalizedOrders((1, 2)),
        input_channels=NormalizedChannels(((1, 2), (2, 2))),
        output_channels=NormalizedChannels(((1, 2), (2, 2))),
    )
    from tpen.nn import PathAggregation

    mixing = CompositeMixing(layout=layout, producers=(
        LinearEquivariantMixing(max_order=2, channels=2, metadata=linear),
        EquivariantMixing(max_order=2, channels=2, paths=tensor),
    ))
    atoms = AtomicConfiguration(positions=torch.zeros(1, 3), charges=torch.tensor([2.0]))
    return TPENWaveFunction(
        embedding=Embedding(max_order=2, spatial_dim=3, out_channels=2,
                            hidden_channels=2, num_hidden_layers=1,
                            initializer=TorchInitializer(seed=2) if seeded else None),
        layers=[TPENLayer(
            mixing=mixing,
            path_aggregation=PathAggregation(max_order=2, channels=2, layout=layout),
            update=ChannelMappedUpdater(max_order=2, channels=2), layout=layout,
        )],
        readout=PfaffianReadout(channels=2, trainable=True),
        envelope=GaussianConfinement(coefficient=0.25, trainable=True),
        factors=[ElectronElectronCusp(trainable_range=True),
                 ElectronNucleusCusp(atoms, CurvatureElectronNucleusCuspLaw())],
    ).double()


def coverage_probe() -> dict:
    """Emit semantic candidate blocks and observed stock-registration omissions.

    Candidate assignments describe obligations, not implemented factor extractors.
    A scalar block below denotes one exact 1-by-1 score block per indexed scalar,
    never a fallback for an otherwise unsupported tensor contraction.
    """
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.register import register_modules

    rows = []
    for seeded in (False, True):
        model = representative_model(seeded)
        registered = register_modules(model, KFACEigenLayer, [], tdc=TorchDistributedCommunicator())
        owners = {}
        for module, (name, _) in registered.items():
            for parameter in module.parameters():
                owners.setdefault(id(parameter), []).append(name)
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("embedding."):
                family, kind = "embedding_seeded" if seeded else "embedding_linear", "kronecker"
            elif ".mixing.producers.0." in name:
                family, kind = "composite_unary", "kronecker"
            elif ".mixing.producers.1." in name:
                family, kind = "composite_tensor", "kronecker"
            elif ".path_aggregation." in name:
                family, kind = "aggregation", "kronecker"
            elif ".update.channel_maps." in name:
                family, kind = "updater", "kronecker"
            elif name.startswith(("readout.", "envelope.", "factors.")):
                family = ("readout" if name.startswith("readout.") else
                          "envelope" if name.startswith("envelope.") else
                          "electron_electron_jastrow" if name.startswith("factors.0.") else
                          "electron_nucleus_cusp")
                kind = "exact_scalar"
            else:
                family, kind = "unknown", "unsupported"
            assigned = owners.get(id(parameter), [])
            # Aliases need a declared shared-parameter policy before admission.
            supported = len(assigned) == 1 and len(aliases[id(parameter)]) == 1 and kind != "unsupported"
            rows.append({
                "model": "seeded" if seeded else "ordinary", "name": name,
                "family": family, "shape": list(parameter.shape), "numel": parameter.numel(),
                "candidate_kind": kind,
                "candidate_blocks": ([f"{name}[{i}]" for i in range(parameter.numel())]
                                     if kind == "exact_scalar" else
                                     [name.rsplit(".", 1)[0]] if family.startswith("embedding_") else [name]),
                "stock_owners": assigned, "aliases": aliases[id(parameter)],
                "stock_covered": supported,
            })
    return {"registry": rows, "unsupported": [r["model"] + ":" + r["name"] for r in rows if not r["stock_covered"]]}


def factor_probe() -> dict:
    """Compare public helpers with sum/count oracles and expose two conventions."""
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper
    from tpen.nn import SeededLinear, TorchInitializer

    module = SeededLinear(1, 1, bias=False, initializer=TorchInitializer(seed=3)).double()
    helper = LinearModuleHelper(module)  # Public helper works even though auto-registration skips it.
    cases = {}
    for counts in ((4,), (2, 2), (1, 3), (1, 0, 3)):
        values = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float64)
        sensitivities = torch.tensor([[2.0], [1.0], [4.0], [3.0]], dtype=torch.float64)
        shards = list(zip(values.split(counts), sensitivities.split(counts)))
        matrices = {}
        for key, index, method in (("A", 0, helper.get_a_factor), ("G", 1, helper.get_g_factor)):
            local = [method(shard[index]) for shard in shards]
            sums = [shard[index].T @ shard[index] for shard in shards]
            oracle = sum(sums) / sum(counts)
            concatenated = method(torch.cat([shard[index] for shard in shards]))
            torch.testing.assert_close(concatenated, oracle, rtol=0, atol=1e-14)
            stock_rank_mean = torch.stack(local).mean(0)
            matrices[key] = {"sums": [s.tolist() for s in sums], "count": sum(counts),
                             "oracle": oracle.tolist(), "stock_rank_mean": stock_rank_mean.tolist(),
                             "matches": torch.allclose(stock_rank_mean, oracle, rtol=0, atol=1e-14)}
        cases[str(counts)] = matrices

    # Reproduce the same count loss through the real public microbatch API.
    layer = KFACEigenLayer(helper, tdc=TorchDistributedCommunicator(), inv_dtype=torch.float64)
    for shard in values.split((1, 3)):
        layer.save_layer_input([shard])
    layer.update_a_factor(alpha=0.5)
    expected_ema = 0.5 * torch.eye(1, dtype=torch.float64) + 0.5 * (values.T @ values / 4)

    a = torch.tensor([1.0, 3.0], dtype=torch.float64)
    g = torch.tensor([2.0, 4.0], dtype=torch.float64)
    score = a * g
    return {
        "shard_arithmetic_only_no_ddp": cases,
        "seeded_public_helper_factor_control": "passed",
        "microbatch_A": layer.a_factor.tolist(), "count_correct_EMA_A": expected_ema.tolist(),
        "microbatch_matches": torch.equal(layer.a_factor, expected_ema),
        "convention_counterexample": {
            "raw_score_second_moment": score.square().mean().item(),
            "uncentered_kronecker": (a.square().mean() * g.square().mean()).item(),
            "centered_score_covariance": ((score - score.mean()).square().mean()).item(),
            "independently_centered_factors": (a.var(unbiased=False) * g.var(unbiased=False)).item(),
        },
    }


def restart_probe() -> dict:
    """Compare public state reloads before/after an inverse refresh at step 4."""
    import torch
    from kfac.preconditioner import KFACPreconditioner

    def build():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False)).double()
        with torch.no_grad():
            model[0].weight.fill_(1.0)
        preconditioner = KFACPreconditioner(
            model, factor_update_steps=1, inv_update_steps=4, factor_decay=0.5,
            damping=0.1, lr=0.01, kl_clip=1e30, inv_dtype=torch.float64,
            factor_dtype=torch.float64, compute_eigenvalue_outer_product=False,
            allreduce_bucket_cap_mb=0, update_factors_in_hook=False,
        )
        return model, preconditioner, torch.optim.SGD(model.parameters(), lr=0.01)

    def step(parts, index):
        model, preconditioner, optimizer = parts
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[float(2 * index + 1)]], dtype=torch.float64)).sum().backward()
        preconditioner.step()
        optimizer.step()

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        return left == right

    results = {}
    for split in (1, 2, 3, 4, 5):
        live = build()
        for index in range(split):
            step(live, index)
        snapshot = [copy.deepcopy(owner.state_dict()) for owner in live]
        resumed = build()
        for owner, state in zip(resumed, snapshot):
            owner.load_state_dict(state)
        updates = []
        for index in range(split, split + 2):
            step(live, index)
            step(resumed, index)
            updates.append({
                "step": index, "model_equal": equal(live[0].state_dict(), resumed[0].state_dict()),
                "serialized_method_equal": equal(live[1].state_dict(), resumed[1].state_dict()),
                "optimizer_equal": equal(live[2].state_dict(), resumed[2].state_dict()),
                "weight_difference": (live[0][0].weight - resumed[0][0].weight).abs().max().item(),
            })
        results[str(split)] = {"completed_steps_at_checkpoint": split,
                               "state_keys": sorted(snapshot[1]),
                               "layer_state_keys": sorted(snapshot[1]["layers"]["0"]),
                               "next_updates": updates}
    return results


def main() -> int:
    """Write a diagnostic receipt without interpreting missing evidence as a pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kfac-source", required=True, type=Path,
                        help="clean external checkout of the exact inspected upstream pin")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    receipt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "verdict": "not_admitted", "upstream_sha": UPSTREAM_SHA,
               "criterion_verdicts": criterion_verdicts(),
               "not_exercised": [
                   "Complete TPEN custom block/factor integration under a frozen VMC convention",
                   "Real multi-rank unequal/empty-shard factor transport and global-empty rejection",
                   "Complete-state whole-TPEN restart across factor and inverse refresh",
                   "DG0 and DP1-DP3 process-group/failure/state integration",
               ],
               "tpen_sha": git(REPO, "rev-parse", "HEAD")}
    try:
        receipt["upstream"] = check_upstream(args.kfac_source.resolve())
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(1)
        receipt["torch"] = torch.__version__
        receipt["coverage"] = coverage_probe()
        receipt["factors"] = factor_probe()
        receipt["restart"] = restart_probe()
        failures = []
        if receipt["coverage"]["unsupported"]:
            failures.append("C1: incomplete stock whole-TPEN coverage")
        if not receipt["factors"]["microbatch_matches"]:
            failures.append("C3: normalized microbatch means discard represented counts")
        if any(not update["model_equal"] for result in receipt["restart"].values() for update in result["next_updates"]):
            failures.append("C5: stock reload changes next parameter updates")
        receipt["observed_stock_failures"] = failures
        receipt["probe_status"] = "negative_witnesses_observed" if failures else "inconclusive"
        code = 1 if failures else 2
    except Exception as error:
        # Preserve partial evidence. A broken environment or probe is not a
        # compatibility failure and cannot count as a successful negative run.
        receipt["probe_status"] = "blocked"
        receipt["error"] = f"{type(error).__name__}: {error}"
        code = 2
    try:
        output = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as error:
        # Nonfinite diagnostic output is an incomplete probe, never exit 1.
        output = json.dumps({"verdict": "not_admitted", "probe_status": "blocked",
                             "criterion_verdicts": criterion_verdicts(),
                             "error": str(error)})
        code = 2
    print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
