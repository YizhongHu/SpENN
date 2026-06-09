# SpENN project specific guidelines

## Environment

- Any environment problems is not worth trouble-shooting by the agent on its own. If it happens, stop and the issue will be resolved interactively.
- This repo uses `uv` to manage python packages. Most commands (including `pytest`) needs to be run with `uv`. Use `uv` to run if possible for reproducibility.
- If it may be necessary to install a new package, stop and inquire instead of proceeding
  with alternatives.
- Do not use `uv run --nosync`. If `uv` environment needs to change, let `uv lock` update for
  reproducibility.

## Conventions
- NumpyDoc is used for documentation
- Use inline comments for comprehensibility

## Tools
- You are strongly encouraged to autonomously spawn subagents to go faster for reading, editing, testing,
running, and debugging tasks.
- You are allowed to autonomously spawn agents for the purposes stated above.
- You are strongly encouraged to autonomously initiate slurm runs for parallizability. Keep slurm logs around
for reproducibility.
- You are allowed to autonomously submit slurm jobs for efficiency.

## Best Practises
- Use existing libraries if possible
- Vectorize with NumPy/PyTorch if possible
- If a config or file or function or class is no longer used, remove it.

## Best Practices

Any reintroduction of `permute_tree`, `validate_tree`, `infer_particle_count`, or equivalent recursive container-probing helpers is a blocker.
These helpers erase representation semantics and are not allowed in SpENN. Particle count, permutation, comparison, and validation must come from explicit typed-object contracts (`.permute(...)`, `.compare(...)`, `.validate(...)`, explicit `n_particles`/`n_electrons` metadata), never from recursively inspecting arbitrary containers.

### Prefer explicit ownership over local convenience

Do not place helper functions wherever they are first needed. Put each helper in the module that owns the relevant concept.

Examples:

```text
Permutation logic       -> spenn/data/permutation.py
Tuple-index logic       -> spenn/data/indices.py
Virtual path logic      -> spenn/reps/paths.py
Partition logic         -> spenn/data/partition.py
Irrep metadata          -> spenn/reps/irreps.py
Young tableaux          -> add a reps-level owner module only when needed
Specht modules          -> spenn/reps/specht.py
Fourier transforms      -> spenn/reps/fourier.py
Trainable modules       -> spenn/nn/
```

Bad:

```python
# spenn/nn/equivariant_mixing.py
def ordered_tuples(...):
    ...
```

Good:

```python
from spenn.data.indices import ordered_tuples
```

### Keep equivariance contracts executable

Every state-like object should implement `.permute(permutation)`. Every equivariant module should subclass `EquivariantMap` and implement `forward_impl`, not `forward`.

Bad:

```python
class MyMap(nn.Module):
    def forward(self, x):
        ...
```

Good:

```python
class MyMap(EquivariantMap):
    def forward_impl(self, x):
        ...
```

`EquivariantMap.forward` owns passive trace recording and delegates to `forward_impl`; it does **not** check equivariance. Runtime equivariance checking is separate: the checkers in `spenn.equivariance.checks` (driven by the `RuntimeEquivariance` callback) plus pytest-only helpers under `tests/`. Do not override `forward` or wrap it with equivariance-check decorators, because that obscures control flow and can cause recursion.

### Separate metadata generation from model execution

Path and irrep metadata should be deterministic and cached. Model code should read metadata; it should not silently regenerate or overwrite metadata during training.

Good:

```python
paths = PathMetadata.load("spenn/cache/paths_canonical.json")
```

Avoid:

```python
# inside training or model forward
paths = generate_virtual_paths(...)
save_paths(paths)
```

Generation and saving should be explicit developer actions.

### Keep path axes explicit until correctness is established

`RealInteraction` should keep a visible path axis:

```text
[batch, channels, paths, indices...]
```

Do not prematurely fold paths into channels. Keeping paths explicit makes debugging, equivariance testing, and path-count checks much easier.

### Implement slow reference versions first

For mathematically delicate operations, prefer a slow, readable reference implementation before vectorizing.

Example:

```python
for path in paths:
    for K in ordered_tuples(n, path.s, distinct=True):
        ...
```

Later vectorized implementations should be tested against the slow reference:

```text
fast(x) == slow(x)
fast(pi x) == pi fast(x)
```

### Do not preserve legacy names in new code

Backwards compatibility is not required for this restructure. Do not use old abstractions in the new path.

Avoid:

```text
SpechtMP
FusionMap
BranchMap
MessageHead
UpdateHead
Convolution
Pooling
FeatureDict
MessageDict
RealTensor
```

Use:

```text
RealFeature
RealInteraction
IrrepInteraction
IrrepFeature
RealUpdate
EquivariantMixing
PathAggregation
Update
SpENNLayer
SpENNWaveFunction
PfaffianReadout
```

### Prefer small PR steps

For this project, correctness is more important than breadth. Prefer small changes with strong tests.

Good PR sequence:

```text
1. Add state dataclasses and permute tests.
2. Add EquivariantMap and runtime-check tests.()
3. Add path metadata and path-count tests.
4. Add slow EquivariantMixing and equivariance tests.
5. Add Fourier/Specht activation.
6. Add readout and wavefunction integration.
```

Avoid large PRs that change state layout, path enumeration, Fourier logic, activation, readout, and experiments at the same time.

## Branches

Coding agents may push only to agent-namespaced branches: Codex to `codex/**`, Claude to `claude/**`.

Agents must not push to branches other than these mentioned above, such as `main` or the `hooke` integration branch,
 merge PRs, or force-push unless the user explicitly asks. Feature branches open PRs against `hooke`.

Agents should respond to PR review comments by adding commits to the existing PR branch.
