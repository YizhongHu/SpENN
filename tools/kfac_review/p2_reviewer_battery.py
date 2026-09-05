#!/usr/bin/env python3
"""P2 durable-reviewer battery for the KFAC compatibility gate (PR #471).

First reviewed tip: f5659e053b607d13108736a19e43e4951cdcf1c7
Upstream pin:       5987766a43739de7eb950f564da54559f2504579 (kfac-pytorch 0.4.2)

Two families of arms:
- stock_*  : corroborate or refute the orchestrator's source-derived STOCK
             failure claims (C3 microbatch averaging, C5 reload divergence,
             empty-shard silence).
- ext_*    : adversarial REFUTATION arms. Each tries to achieve, through
             public upstream APIs only, a behavior the gate left
             UNESTABLISHED: exact restart across an inverse-refresh boundary
             (C5), count-correct sum-plus-count factor accumulation with
             global-empty rejection (C3), and manual registration of a module
             that automatic registration skips (C1/C4 seeded family).

Every arm records observed values; interpretation stays with the reviewer.
Exit 0 means the battery itself completed (arms may still record failures);
exit 3 means the battery could not run.

Run with the checkout venv python directly, never `uv run` (a resync would
prune the kfac editable install).

These are single-process diagnostic fixtures, not a TPEN adapter. The counted
extension arm exercises A accumulation only; its G methods are not qualified.
The receipt records the executing checkout tip and verifies upstream sources.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def tensor_list(t):
    return t.tolist() if t is not None else None


def run_arm(receipt, name, fn):
    try:
        receipt["arms"][name] = {"status": "completed", "result": fn()}
    except Exception as error:  # noqa: BLE001 - diagnostic battery
        import traceback

        receipt["arms"][name] = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Shared public-API single-process builder
# ---------------------------------------------------------------------------

def build_single_process_preconditioner(module, *, layer_cls=None, name="0",
                                        factor_update_steps=1,
                                        inv_update_steps=4,
                                        factor_decay=0.5, damping=0.1,
                                        lr=0.01):
    """Assemble BaseKFACPreconditioner from public constructors only.

    Returns (preconditioner, layer) so the caller keeps a public handle on the
    KFAC layer object it constructed itself (no private attribute access).
    """
    import torch
    from kfac.assignment import KAISAAssignment
    from kfac.base_preconditioner import BaseKFACPreconditioner
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    if layer_cls is None:
        layer_cls = KFACEigenLayer
    tdc = TorchDistributedCommunicator()
    helper = LinearModuleHelper(module)
    layer = layer_cls(
        helper,
        tdc=tdc,
        inv_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    assignment = KAISAAssignment(
        {name: {"A": 1.0, "G": 1.0}},
        local_rank=0,
        world_size=1,
        grad_worker_fraction=1.0,
        group_func=lambda ranks: None,
    )
    preconditioner = BaseKFACPreconditioner(
        {module: (name, layer)},
        assignment=assignment,
        tdc=tdc,
        factor_update_steps=factor_update_steps,
        inv_update_steps=inv_update_steps,
        factor_decay=factor_decay,
        damping=damping,
        kl_clip=1e30,
        lr=lr,
        update_factors_in_hook=False,
    )
    return preconditioner, layer


# ---------------------------------------------------------------------------
# Arm 1 — stock C3: microbatch averaging discards represented counts
# ---------------------------------------------------------------------------

def arm_stock_c3_microbatch():
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    module = torch.nn.Linear(1, 1, bias=False).double()
    helper = LinearModuleHelper(module)
    layer = KFACEigenLayer(
        helper, tdc=TorchDistributedCommunicator(), inv_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    values = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float64)
    for shard in values.split((1, 3)):
        layer.save_layer_input([shard])
    layer.update_a_factor(alpha=0.5)
    stock = layer.a_factor
    # Count-correct EMA over the same 4 rows from identity init:
    oracle = 0.5 * torch.eye(1, dtype=torch.float64) + 0.5 * (values.T @ values / 4)
    return {
        "stock_a_ema": tensor_list(stock),          # expected 23/3 per gate doc
        "count_correct_a_ema": tensor_list(oracle),  # 11
        "expected_stock_23_over_3": abs(stock.item() - 23.0 / 3.0) < 1e-12,
        "stock_matches_count_correct": torch.equal(stock, oracle),
    }


# ---------------------------------------------------------------------------
# Arm 2 — stock C3 addendum: empty shard is accepted silently as a zero factor
# ---------------------------------------------------------------------------

def arm_stock_c3_empty_shard_silent():
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    module = torch.nn.Linear(1, 1, bias=False).double()
    helper = LinearModuleHelper(module)

    empty = torch.empty((0, 1), dtype=torch.float64)
    empty_factor_error = None
    try:
        empty_factor = helper.get_a_factor(empty)
        empty_factor_value = tensor_list(empty_factor)
    except Exception as error:  # noqa: BLE001
        empty_factor_error = f"{type(error).__name__}: {error}"
        empty_factor_value = None

    # Feed an all-empty batch sequence through the public microbatch API.
    layer = KFACEigenLayer(
        helper, tdc=TorchDistributedCommunicator(), inv_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    all_empty_error = None
    try:
        layer.save_layer_input([empty])
        layer.update_a_factor(alpha=0.5)
        all_empty_a = tensor_list(layer.a_factor)
    except Exception as error:  # noqa: BLE001
        all_empty_error = f"{type(error).__name__}: {error}"
        all_empty_a = None

    # Mixed case: one empty and one non-empty microbatch bias the mean.
    layer2 = KFACEigenLayer(
        helper, tdc=TorchDistributedCommunicator(), inv_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    mixed_error = None
    try:
        values = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float64)
        layer2.save_layer_input([empty])
        layer2.save_layer_input([values])
        layer2.update_a_factor(alpha=0.5)
        mixed_a = tensor_list(layer2.a_factor)
    except Exception as error:  # noqa: BLE001
        mixed_error = f"{type(error).__name__}: {error}"
        mixed_a = None

    return {
        "get_a_factor_on_empty_rows": empty_factor_value,
        "get_a_factor_on_empty_rows_error": empty_factor_error,
        "all_empty_update_a_ema": all_empty_a,
        "all_empty_update_error": all_empty_error,
        "mixed_empty_plus_full_a_ema": mixed_a,       # count-correct would be 11
        "mixed_empty_plus_full_error": mixed_error,
        "count_correct_reference": 11.0,
    }


# ---------------------------------------------------------------------------
# Arm 3 — stock C5: default reload changes next updates (splits 1..5)
# ---------------------------------------------------------------------------

def _restart_build():
    import torch
    from kfac.preconditioner import KFACPreconditioner

    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    preconditioner = KFACPreconditioner(
        model, factor_update_steps=1, inv_update_steps=4, factor_decay=0.5,
        damping=0.1, lr=0.01, kl_clip=1e30, inv_dtype=torch.float64,
        factor_dtype=torch.float64, compute_eigenvalue_outer_product=False,
        allreduce_bucket_cap_mb=0, update_factors_in_hook=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, preconditioner, optimizer


def _restart_step(parts, index):
    import torch

    model, preconditioner, optimizer = parts
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[float(2 * index + 1)]], dtype=torch.float64)).sum().backward()
    preconditioner.step()
    optimizer.step()


def arm_stock_c5_restart():
    import torch

    results = {}
    for split in (1, 2, 3, 4, 5):
        live = _restart_build()
        for index in range(split):
            _restart_step(live, index)
        snapshot = [copy.deepcopy(owner.state_dict()) for owner in live]
        resumed = _restart_build()
        for owner, state in zip(resumed, snapshot):
            owner.load_state_dict(state)
        updates = []
        for index in range(split, split + 2):
            _restart_step(live, index)
            _restart_step(resumed, index)
            updates.append({
                "step": index,
                "weights_equal": torch.equal(live[0][0].weight, resumed[0][0].weight),
                "weight_abs_difference": (live[0][0].weight - resumed[0][0].weight).abs().max().item(),
            })
        results[str(split)] = updates
    return results


# ---------------------------------------------------------------------------
# Arm 4 — REFUTATION, C5: exact restart through public APIs only
# ---------------------------------------------------------------------------

def _codec_build():
    import torch

    module = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        module.weight.fill_(1.0)
    preconditioner, layer = build_single_process_preconditioner(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    return module, preconditioner, layer, optimizer


def _codec_step(module, preconditioner, optimizer, index):
    import torch

    optimizer.zero_grad(set_to_none=True)
    module(torch.tensor([[float(2 * index + 1)]], dtype=torch.float64)).sum().backward()
    preconditioner.step()
    optimizer.step()


def arm_ext_c5_public_codec():
    """Save factors AND eigen caches via public properties; restore with
    load_state_dict(compute_inverses=False) plus the public cache setters."""
    import torch

    cache_keys = ("qa", "qg", "da", "dg", "dgda")
    results = {}
    for split in (1, 2, 3, 4, 5):
        module, precond, layer, opt = _codec_build()
        for index in range(split):
            _codec_step(module, precond, opt, index)

        checkpoint = {
            "model": copy.deepcopy(module.state_dict()),
            "kfac": copy.deepcopy(precond.state_dict(include_factors=True)),
            "caches": {
                key: (getattr(layer, key).clone()
                      if getattr(layer, key) is not None else None)
                for key in cache_keys
            },
            "optimizer": copy.deepcopy(opt.state_dict()),
        }

        module_r, precond_r, layer_r, opt_r = _codec_build()
        module_r.load_state_dict(checkpoint["model"])
        opt_r.load_state_dict(checkpoint["optimizer"])
        precond_r.load_state_dict(copy.deepcopy(checkpoint["kfac"]),
                                  compute_inverses=False)
        for key in cache_keys:
            saved = checkpoint["caches"][key]
            setattr(layer_r, key, saved.clone() if saved is not None else None)

        updates = []
        for index in range(split, split + 2):
            _codec_step(module, precond, opt, index)
            _codec_step(module_r, precond_r, opt_r, index)
            updates.append({
                "step": index,
                "weights_equal": torch.equal(module.weight, module_r.weight),
                "weight_abs_difference": (module.weight - module_r.weight).abs().max().item(),
            })
        results[str(split)] = updates
    results["exact_at_all_splits"] = all(
        update["weights_equal"]
        for split, updates in results.items() if split.isdigit()
        for update in updates
    )
    return results


# ---------------------------------------------------------------------------
# Arm 5 — REFUTATION, C3: count-correct accumulation via public subclassing
# ---------------------------------------------------------------------------

def arm_ext_c3_count_correct_subclass():
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    class CountedKFACEigenLayer(KFACEigenLayer):
        """Accumulate unnormalized sums plus represented row counts; refuse a
        global-empty update. Public-surface extension: subclass-owned
        accumulators only, and factor state written solely through the public
        a_factor/g_factor properties and public ModuleHelper methods. The
        parent's underscore-prefixed batch buffers are never read or written
        (round-1 disposition: touching inherited _a_batch/_a_count was not a
        public-only witness)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._counted_a_sum = None   # subclass-owned, not inherited state
            self._counted_a_rows = 0
            self._counted_a_seen = False
            self._counted_g_sum = None
            self._counted_g_rows = 0
            self._counted_g_seen = False

        def save_layer_input(self, input_):
            x = input_[0].to(self.factor_dtype)  # public attribute
            self._counted_a_seen = True
            rows = x.reshape(-1, x.shape[-1]).shape[0]
            if rows == 0:
                return  # an empty shard contributes nothing, not a zero mean
            a_sum = self.module.get_a_factor(x) * rows  # public helper method
            self._counted_a_sum = (
                a_sum if self._counted_a_sum is None
                else self._counted_a_sum + a_sum
            )
            self._counted_a_rows += rows

        def save_layer_grad_output(self, grad_output):
            g = grad_output[0].to(self.factor_dtype)
            if self.grad_scaler is not None:  # public attribute
                g = g / self.grad_scaler()
            self._counted_g_seen = True
            rows = g.reshape(-1, g.shape[-1]).shape[0]
            if rows == 0:
                return
            g_sum = self.module.get_g_factor(g) * rows
            self._counted_g_sum = (
                g_sum if self._counted_g_sum is None
                else self._counted_g_sum + g_sum
            )
            self._counted_g_rows += rows

        def update_a_factor(self, alpha=0.95):
            import torch as _torch

            if self._counted_a_seen and self._counted_a_rows == 0:
                raise RuntimeError("global-empty A update refused (fail-closed)")
            self._counted_a_seen = False
            if self._counted_a_sum is None:
                return
            a_new = self._counted_a_sum / self._counted_a_rows
            self._counted_a_sum = None
            self._counted_a_rows = 0
            if self.a_factor is None:  # public property getter
                self.a_factor = _torch.diag(
                    a_new.new(a_new.shape[0]).fill_(1),
                )  # public property setter, upstream identity init convention
            self.a_factor = (alpha * self.a_factor) + ((1 - alpha) * a_new)

        def update_g_factor(self, alpha=0.95):
            import torch as _torch

            if self._counted_g_seen and self._counted_g_rows == 0:
                raise RuntimeError("global-empty G update refused (fail-closed)")
            self._counted_g_seen = False
            if self._counted_g_sum is None:
                return
            g_new = self._counted_g_sum / self._counted_g_rows
            self._counted_g_sum = None
            self._counted_g_rows = 0
            if self.g_factor is None:
                self.g_factor = _torch.diag(
                    g_new.new(g_new.shape[0]).fill_(1),
                )
            self.g_factor = (alpha * self.g_factor) + ((1 - alpha) * g_new)

    module = torch.nn.Linear(1, 1, bias=False).double()
    helper = LinearModuleHelper(module)
    values = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float64)
    oracle = 0.5 * torch.eye(1, dtype=torch.float64) + 0.5 * (values.T @ values / 4)

    outcomes = {}
    for counts in ((4,), (2, 2), (1, 3), (1, 0, 3)):
        layer = CountedKFACEigenLayer(
            helper, tdc=TorchDistributedCommunicator(),
            inv_dtype=torch.float64, factor_dtype=torch.float64,
        )
        for shard in values.split(counts):
            layer.save_layer_input([shard])
        layer.update_a_factor(alpha=0.5)
        outcomes[str(counts)] = {
            "a_ema": tensor_list(layer.a_factor),
            "matches_concatenated_oracle": torch.equal(layer.a_factor, oracle),
        }

    # Global-empty rejection, which stock accepts silently as a zero factor.
    layer = CountedKFACEigenLayer(
        helper, tdc=TorchDistributedCommunicator(),
        inv_dtype=torch.float64, factor_dtype=torch.float64,
    )
    layer.save_layer_input([torch.empty((0, 1), dtype=torch.float64)])
    try:
        layer.update_a_factor(alpha=0.5)
        global_empty = {"rejected": False, "a_ema": tensor_list(layer.a_factor)}
    except RuntimeError as error:
        global_empty = {"rejected": True, "error": str(error)}

    outcomes["global_empty_rejection"] = global_empty
    outcomes["all_unequal_and_empty_match_oracle"] = all(
        outcomes[key]["matches_concatenated_oracle"]
        for key in ("(4,)", "(2, 2)", "(1, 3)", "(1, 0, 3)")
    )
    return outcomes


# ---------------------------------------------------------------------------
# Arm 6 — REFUTATION, C1/C4 seeded family: manual public registration
# ---------------------------------------------------------------------------

def arm_ext_c1_manual_registration_seeded():
    """SeededLinear is skipped by register_modules (not an nn.Linear subclass),
    but LinearModuleHelper plus the public BaseKFACPreconditioner layers dict
    accepts it. Witness: identical trajectories to an nn.Linear twin."""
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.register import register_modules
    from tpen.nn import SeededLinear, TorchInitializer

    seeded = SeededLinear(1, 1, bias=False,
                          initializer=TorchInitializer(seed=3)).double()
    stock_registry = register_modules(
        seeded, KFACEigenLayer, [], tdc=TorchDistributedCommunicator(),
    )

    control = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        control.weight.copy_(seeded.weight)

    precond_s, _ = build_single_process_preconditioner(seeded)
    precond_c, _ = build_single_process_preconditioner(control)
    opt_s = torch.optim.SGD(seeded.parameters(), lr=0.01)
    opt_c = torch.optim.SGD(control.parameters(), lr=0.01)

    trajectory = []
    for index in range(6):
        x = torch.tensor([[float(2 * index + 1)]], dtype=torch.float64)
        for module, precond, opt in ((seeded, precond_s, opt_s),
                                     (control, precond_c, opt_c)):
            opt.zero_grad(set_to_none=True)
            module(x).sum().backward()
            precond.step()
            opt.step()
        trajectory.append({
            "step": index,
            "weights_equal": torch.equal(seeded.weight, control.weight),
        })

    return {
        "stock_register_modules_found": len(stock_registry),  # expected 0
        "manual_public_registration_trajectory": trajectory,
        "identical_to_linear_control": all(t["weights_equal"] for t in trajectory),
    }


# ---------------------------------------------------------------------------
# Arm 7 — finding: eigendecomposition always downcasts to float32
# ---------------------------------------------------------------------------

def arm_fp32_eigh_downcast():
    import torch
    from kfac.distributed import TorchDistributedCommunicator
    from kfac.layers.eigen import KFACEigenLayer
    from kfac.layers.modules import LinearModuleHelper

    module = torch.nn.Linear(1, 1, bias=False).double()
    helper = LinearModuleHelper(module)
    layer = KFACEigenLayer(
        helper, tdc=TorchDistributedCommunicator(), inv_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    # 1 + 1e-9 is representable in float64 but rounds to 1.0 in float32.
    layer.a_factor = torch.tensor([[1.0 + 1e-9]], dtype=torch.float64)
    layer.compute_a_inv(damping=0.0)
    eigenvalue = layer.da.item()
    return {
        "a_factor": 1.0 + 1e-9,
        "computed_eigenvalue": eigenvalue,
        "float64_information_lost": eigenvalue == 1.0,
        "da_dtype": str(layer.da.dtype),
    }


# ---------------------------------------------------------------------------
# Arm 8 — run the orchestrator's probe verbatim
# ---------------------------------------------------------------------------

def arm_run_orchestrator_probe():
    kfac_source = os.environ["P2_KFAC_SOURCE"]
    repo = os.environ["P2_TPEN_CHECKOUT"]
    probe = Path(repo) / "tools" / "kfac_compatibility_probe.py"
    completed = subprocess.run(
        [sys.executable, str(probe), "--kfac-source", kfac_source],
        capture_output=True, text=True, timeout=1800,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"unparseable_stdout": completed.stdout[-4000:]}
    return {
        "exit_code": completed.returncode,
        "probe_status": payload.get("probe_status"),
        "observed_stock_failures": payload.get("observed_stock_failures"),
        "criterion_verdicts": payload.get("criterion_verdicts"),
        "coverage_unsupported_count": len(
            payload.get("coverage", {}).get("unsupported", []) or [],
        ),
        "stderr_tail": completed.stderr[-2000:],
        "full_receipt_path": os.environ.get("P2_PROBE_RECEIPT_COPY", ""),
        "full_stdout": payload,
    }


def main() -> int:
    receipt = {
        "battery": "p2-reviewer-battery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_pin": "5987766a43739de7eb950f564da54559f2504579",
        "arms": {},
    }
    try:
        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "tools"))
        from kfac_compatibility_probe import check_upstream, git

        receipt["review_tip"] = git(repo, "rev-parse", "HEAD")
        if git(repo, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("review checkout has tracked changes")
        receipt["upstream"] = check_upstream(Path(os.environ["P2_KFAC_SOURCE"]).resolve())
        # The verbatim probe must run from this same checkout.
        os.environ["P2_TPEN_CHECKOUT"] = str(repo)
        import torch

        torch.manual_seed(0)
        torch.set_num_threads(1)
        receipt["python"] = sys.executable
        receipt["torch"] = torch.__version__
        import kfac

        receipt["kfac_version"] = kfac.__version__
        receipt["kfac_imported_from"] = kfac.__file__
    except Exception as error:  # noqa: BLE001
        receipt["environment_error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(receipt, indent=2))
        return 3

    run_arm(receipt, "stock_c3_microbatch", arm_stock_c3_microbatch)
    run_arm(receipt, "stock_c3_empty_shard_silent", arm_stock_c3_empty_shard_silent)
    run_arm(receipt, "stock_c5_restart", arm_stock_c5_restart)
    run_arm(receipt, "ext_c5_public_codec", arm_ext_c5_public_codec)
    run_arm(receipt, "ext_c3_count_correct_subclass", arm_ext_c3_count_correct_subclass)
    run_arm(receipt, "ext_c1_manual_registration_seeded", arm_ext_c1_manual_registration_seeded)
    run_arm(receipt, "fp32_eigh_downcast", arm_fp32_eigh_downcast)
    run_arm(receipt, "orchestrator_probe_verbatim", arm_run_orchestrator_probe)

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
