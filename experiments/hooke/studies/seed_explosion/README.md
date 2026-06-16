# Seed explosion study: final-train seeds 105 / 106 / 107

## Question

In the `hooke_pair_validation` (v2) study, the selected final-train config

```
lr = 3e-3,  channels = 32,  layers = 1,  gate_activation = sigmoid   (singlet, n_particles = 2)
```

is trained at 10 seeds (`runtime.seed = 100..109`) and then evaluated. Seeds
**105, 106, 107** "explode". This study finds *what* explodes and shows that the
three seeds have **different** underlying defects.

All numbers below come from the artifacts already produced on the H200 runs and
synced into `pair_validation/reports/`; the scripts here only read those CSVs
(no model checkpoints are needed). Reproduce with:

```bash
uv run --extra cpu python experiments/hooke/studies/seed_explosion/probe.py            # training is healthy
uv run --extra cpu python experiments/hooke/studies/seed_explosion/probe_eval.py       # eval energy/variance per seed
uv run --extra cpu python experiments/hooke/studies/seed_explosion/analyze_probes.py   # localize + plots
uv run --extra cpu python experiments/hooke/studies/seed_explosion/decompose_explosion.py  # term breakdown
uv run --extra cpu python experiments/hooke/studies/seed_explosion/classify_defects.py # node vs tail
uv run --extra cpu python experiments/hooke/studies/seed_explosion/onset_vs_reach.py   # latent defect -> MC variance
```

## TL;DR

* **Training does not explode.** All 10 seeds converge cleanly to E ≈ 2.0 (the
  exact Hooke pair singlet energy), with finite losses and bounded gradients
  (`probe.py`). The problem is purely in the **evaluation** of the learned
  wavefunction.
* **What explodes is the local energy `E_L`**, via the **kinetic term**, in the
  low-probability tails of the wavefunction. The exact `E_L(R) = 2.0` everywhere;
  the model's `E_L` diverges where its `log|ψ|` has the wrong shape.
* The blow-up inflates the **sampled energy variance** (0.042–0.089 for the three
  seeds vs 0.003–0.02 for the healthy ones) and biases the energy estimate above
  2.0. It does **not** produce NaN/Inf — `local_energy_finite_fraction = 1.0`
  everywhere — so the runs complete "successfully" with bad numbers.
* **The three seeds fail in different coordinates / different ways:**

  | seed | defect | where | worst `E_L` | sampled `E_var` |
  |------|--------|-------|-------------|-----------------|
  | 105  | **spurious node in the centre-of-mass coordinate** (sign flip, &#124;ψ&#124;→0) | COM radius ≈ 2.9–6.9 | +613 / −176 | 0.075 |
  | 106  | **spurious near-nodes in the relative (inter-electron) coordinate** (12 sign flips in the tail) | r12 ≈ 5.7–8 | −119 | 0.042 |
  | 107  | **smooth tail-curvature error, no node** | COM radius ≈ 7.4 | +5.6 | 0.089 |

## Why the kinetic term is the culprit

For this Hamiltonian `E_L = T + V_trap + V_ee`, where the harmonic trap
`V_trap = ½ω²R²` and the Coulomb `V_ee = 1/r12` are fixed analytic functions of
geometry (the probe records them exactly). So **every** departure of `E_L` from
2.0 must live in the kinetic term

```
T = −½ ( ∇²log|ψ| + |∇log|ψ||² ).
```

`decompose_explosion.py` confirms this at each seed's worst point: e.g. seed 105
at COM radius 6.87 has `V_trap = +11.9`, `V_ee = +1.0`, but `T = +600`, giving
`E_L = +613`. Because `T` is a *second derivative* of `log|ψ|`, a small shape
error in the tail is amplified into an enormous local-energy spike. The harmonic
trap grows like `R²`, setting the scale that `T` must cancel; when the tail
curvature is wrong, the cancellation fails catastrophically.

## How a latent tail defect becomes observed MC variance

The deterministic probes scan out to radius 8, far past where the Metropolis
sampler lives (`center_of_mass_rms ≈ 1.25`). `onset_vs_reach.py` shows the
**onset radius** — where `|E_L − 2|` first exceeds 0.5 along the COM sweep —
separates the seeds:

```
healthy (102,103,108,109): COM onset 5.0–6.9   E_var 0.003–0.010
suspect (105,106,107):     COM onset 1.3–1.8   E_var 0.042–0.089
```

For the suspect seeds the explosion begins at *moderate* radius, inside the fat
tail of the sampling distribution, so walkers occasionally land on it. That is
exactly why the sampled local-energy minimum goes negative (−0.38 for 105,
−0.44 for 107) and the energy variance is several times larger.

## The shared root cause and why training never fixed it

The variational energy is dominated by the bulk of `|ψ|²`, where all seeds are
fine. Tail/region defects carry almost no probability mass, so they contribute
negligibly to the training loss — SGD has no gradient signal to remove them.
The high learning rate (3e-3, the largest in the sweep) combined with the
`sigmoid` gate lets the envelope develop spurious structure (kinks and even sign
changes / nodes) in the tails without hurting the training energy. The damage is
only revealed at eval time by (a) the large-sample variance estimator and (b)
the deterministic geometry probes that deliberately walk into the tails.

This is a **config-level fragility realized stochastically per seed**: of the 10
seeds, 102/103/108/109 are clean, 100/101/104 are mildly elevated, and
105/106/107 are the worst — with 105 and 106 crossing the line into genuine
spurious nodes (in orthogonal coordinates), and 107 a severe-but-nodeless tail
error.

## Figures

* `plots/local_energy_sweeps.png` — explosion envelope (worst `|E_L − 2|` over all
  probe directions/offsets) vs pair distance and vs COM radius. Seed 106 spikes
  to ~100 on the pair sweep; seed 105 to ~600 on the COM sweep; 107 mildly on COM.
* `plots/defect_signatures.png` — per-seed signature on its worst slice: `E_L`
  (top) and `log|ψ|` with sign-flip markers (bottom). 105 and 106 show the node
  (red dotted = sign flip); 107 shows a smooth curvature mismatch with no flip.

## Files

| file | purpose |
|------|---------|
| `probe.py` | shows all 10 training runs are healthy (energy → 2.0, finite, bounded grads) |
| `probe_eval.py` | per-seed eval energy, error, variance, local-energy tails, kinetic term |
| `analyze_probes.py` | localizes the departure (cusp/mid/tail bands; COM vs relative) and writes both figures |
| `decompose_explosion.py` | term breakdown at each seed's worst point — proves the kinetic term carries the deviation |
| `classify_defects.py` | sign-flip / `|ψ|→0` test that separates spurious nodes (105, 106) from smooth tail error (107) |
| `onset_vs_reach.py` | bridges the latent probe defect to the observed sampled variance |
