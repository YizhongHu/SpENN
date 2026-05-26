# sampling instructions

Sampling owns proposal moves and walker state. It should treat the model as a
callable wavefunction and preserve walker shape `[batch, n_electrons, spatial_dim]`.
