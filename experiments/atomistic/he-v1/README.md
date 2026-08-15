# TPEN He-v1

This development study is the first all-electron TPEN surface: infinite-mass
helium in the nonrelativistic singlet ground state. It is a smoke/contract
study, not a production comparison.

The model uses `NuclearFactorizedEnvelope`: the existing electron-electron
cusp stays in the regular factor, while `NuclearConfinement` supplies the
fixed electron-nucleus Kato factor `-Z_A |r_i-R_A|`. It intentionally has no
Gaussian confinement.

The reference is -2.903724377034119598 Ha, mirrored from the canonical
`he_atom` record in `experiments/baselines/systems.yaml`. Train and eval must
agree exactly on model wiring; eval restores with `strict: true`.

Evaluation records the variational MCMC energy, finite-sample standard error,
and scalar reference comparison together with sampled fermionic
antisymmetry/equivariance contracts. A He-owned positive radial grid now
reports one-sided electron-nucleus cusp slopes, explicit outer-tail slopes,
and a CSV derivative profile while carrying the same nuclear context as the
sampler. Spatial coordinate exchange is checked separately from full label
antisymmetry.

Correlation-aware external tau, ESS, and MCSE statistics remain explicitly
unavailable/absent: no fixed-model trajectory JSONL producer exists yet, so
this study does not estimate them or wire a callback that could imply they do.
