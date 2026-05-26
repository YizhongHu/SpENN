# physics instructions

Physics code may call the model and use autograd, but it should not inspect
Specht irreps or sampler internals. Return batched local energies.
