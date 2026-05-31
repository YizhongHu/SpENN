# Hooke Atom Benchmark

This experiment checks the two-electron Hooke atom benchmarks from
`experiments/experiments.typ`: the opposite-spin singlet at `omega = 1/2`
with exact energy `2.0`, and the same-spin triplet at `omega = 1/4` with
exact energy `1.25`.

The exact runs validate the Hamiltonian, cusp behavior, exchange symmetry, and
Metropolis sampler diagnostics. The SpENN runs train small singlet and triplet
models with VMC energy minimization only; analytic solutions are used afterward
as exact-energy references and radial-shape diagnostics, not as training targets.
The learned configs use tanh encoder MLPs, parity-aware SpechtMP message
activations, linear update heads, and residual feature updates.
The Hooke run scripts are data handlers: they load templates from
`experiments/hooke/configs`, apply dotlist overrides with OmegaConf
interpolation, call the generic `scripts/train.py` train/evaluate stack, then
read the resulting artifacts. Reusable diagnostics write CSV data;
`plot_outputs.py` owns the Hooke-specific figures.

## Reproduce

Run from the repository root:

```bash
uv run python experiments/hooke/run_exact.py --config debug_singlet --run-id codex_debug_singlet
uv run python experiments/hooke/run_exact.py --config debug_triplet --run-id codex_debug_triplet
uv run python experiments/hooke/run_spenn.py --config spenn_singlet_debug --run-id codex_spenn_singlet_vmc_activation
uv run python experiments/hooke/run_spenn.py --config spenn_triplet_debug --run-id codex_spenn_triplet_vmc_activation
uv run python experiments/hooke/plot_outputs.py
```

Run artifacts are written to `outputs/YYYY-MM-DD/hooke_<sector>/<run_id>/`.
Learned comparisons are written to
`outputs/YYYY-MM-DD/hooke_<sector>_spenn/<run_id>/`.
PNG figures are written to:

```text
experiments/hooke/figures/singlet/
experiments/hooke/figures/triplet/
experiments/hooke/figures/singlet_spenn/
experiments/hooke/figures/triplet_spenn/
```
