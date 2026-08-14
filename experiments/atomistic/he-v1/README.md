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

Current evaluation records a variational MCMC energy/reference comparison and
the sampled fermionic antisymmetry/equivariance contracts. Nucleus-aware
fixed-grid cusp and tail diagnostics remain a separate implementation need:
Hooke-only grids cannot truthfully supply nuclear context.
