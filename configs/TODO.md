# configs TODO

- Keep `configs/config.yaml` as the single active constructor tree for runnable code.
- Add future variants only when the corresponding modules are implemented.
- Keep WandB, DDP, MALA, determinant readouts, `M = 3`, and low-rank virtual-order paths out of the active config until they are backed by current code.
- Make any future `M_virtual = 4` behavior explicit opt-in only.
