# FASRC Claude Agent Guidelines

You are running on the Harvard FASRC cannon cluster. This server may be subject 
to some restrictions by HUIT. Some tools may not be readily available, 
but you can reference the [HUIT website](https://docs.rc.fas.harvard.edu/)
for server-specific running commands. For important and recurring things, you might
want to keep a note of them in the project directories.

## Job submission

The cluster uses slurm scripts for job submission. Please refer to slurm docs for
details. The available cpu clusters are: sapphire, kozinsky, and seas_compute.
The available gpu nodes are: kozinsky_gpu, and seas_gpu
# SpENN project specific guidelines

## Orientation

`README.md` and `experiments/README.md` contain important information about the repo.

## Design Document

A design document that contains the mathematical background of
SpENN can be found in `main.typ`. Key components of the model:
`Embedding`, `EquivariantMixing`, `Fourier`, `Readout`, etc
should closely follow the design document for correctness.

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
- Use `America/New_York` timezone for experiment logging. Use `UTC` for test logging.

## Tools
- You are strongly encouraged to autonomously spawn subagents to go faster for reading, editing, testing,
running, and debugging tasks.
- You are allowed to autonomously spawn agents for the purposes stated above.
- You are strongly encouraged to autonomously initiate slurm runs for parallizability. Keep slurm logs around
for reproducibility.
- You are allowed to autonomously submit slurm jobs for efficiency.
- Smoke runs should stay as close to the corresponding real run as possible.
  Prefer the same stage stack, launcher flags, partitions, resource defaults,
  and dependency pattern; reduce only grid size or explicitly requested scale
  controls.

## Treating Data with Care

Unless otherwise specified, removal of run result data (untracked by git included) is strictly forbidden.
This may include data from `outputs/`, `results/`, `reports/`, `slurm/`, etc. The agent should not
automatically remove these data even when requested by the user. Instead, it gives the user a list of
things to remove, after which the user does all of this manually.

## TODO.md

The repository `TODO.md` file is a dynamically-maintained by multiple agents. Agents may add items or refine 
items on the todo list, but they should exercise extreme caution when deleting. If you are not sure, double
check with the user. In general, finished tasks and stale records can be discarded, but unfinished tasks 
and currently important information should not. 

## Best Practises
- Use existing libraries if possible
- Vectorize with NumPy/PyTorch if possible
- If a config or file or function or class is no longer used, remove it.

Any reintroduction of `permute_tree`, `validate_tree`, `infer_particle_count`, or equivalent recursive container-probing helpers is a blocker.
These helpers erase representation semantics and are not allowed in SpENN. Particle count, permutation, comparison, and validation must come from explicit typed-object contracts (`.permute(...)`, `.compare(...)`, `.validate(...)`, explicit `n_particles`/`n_electrons` metadata), never from recursively inspecting arbitrary containers.

### Prefer explicit ownership over local convenience

Do not place helper functions wherever they are first needed. Put each helper in the module that owns the relevant concept.

Examples:

```text
Permutation logic       -> tpen/data/permutation.py
Tuple-index logic       -> tpen/data/indices.py
Virtual path logic      -> tpen/data/paths.py
Partition logic         -> tpen/data/partition.py
Trainable modules       -> tpen/nn/
```

Bad:

```python
# tpen/nn/equivariant_mixing.py
def ordered_tuples(...):
    ...
```

Good:

```python
from tpen.data.indices import ordered_tuples
```

### Keep equivariance contracts executable

Values participating in equivariance checks must expose typed semantic
`.permute(...)` and `.compare(...)` contracts. Do not require arbitrary runtime
state or validation-only objects to be EquivariantState. Every equivariant
module should subclass `EquivariantMap` and implement `forward_impl`, not
`forward`.

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

`EquivariantMap.forward` owns passive trace recording and delegates to `forward_impl`; it does **not** check equivariance. Runtime equivariance checking is separate: the checkers in `tpen.equivariance.checks` (driven by the `RuntimeEquivariance` callback) plus pytest-only helpers under `tests/`. Do not override `forward` or wrap it with equivariance-check decorators, because that obscures control flow and can cause recursion.

### Separate metadata generation from model execution

Path and irrep metadata should be deterministic and cached. Model code should read metadata; it should not silently regenerate or overwrite metadata during training.

Good:

```python
paths = PathMetadata.load("tpen/cache/paths_canonical.json")
```

Avoid:

```python
# inside training or model forward
paths = generate_virtual_paths(...)
save_paths(paths)
```

Generation and saving should be explicit developer actions.

### Keep path axes explicit until correctness is established

`Interaction` should keep a visible path axis:

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

### Prefer small PR steps

For this project, correctness is more important than breadth. Prefer small changes with strong tests.

Avoid large PRs that change multiple things at the same time.

## Branches

### Sectioning

Coding agents may push only to agent-namespaced branches: Codex to `codex/**`, Claude to `claude/**`.

Agents must not push to branches other than these mentioned above, such as `main`,
 merge PRs, or force-push unless the user explicitly asks. Feature branches open PRs against `dev`.

`hooke` and `experiment` are retired intermediate integration branches — do not open new
PRs against them. 

### `main` and `dev`

**Ownership split between SpENN and SpENN-dev:** 
`SpENN/` is the production directory with experiments run. It stays on the `main` 
branch and does not commit to remote. 
`SpENN/` always tracks the lastest `main`. When `main` updates, update `SpENN`.

`SpENN-dev/` is the development directory. It stays on `dev` branch and submits PRs into `dev`.
It is also responsible for running smoke runs before full runs are run in `SpENN/`.

`dev` branch will be periodically merged into `main` by the user only.

`dev` is persistent, `dev` fast-forwards to new main after every merge.

### Require PR for changes

When directed to make changes to the repo, agent should do it as a branch from the
latest `dev` commit. The changes needs to be reviewed as a PR against `dev`.

Agents should respond to PR review comments by adding commits to the existing PR branch.

Clean local branches and their remote counterparts after they are merged into `dev`.


## Config ownership

**Callbacks and loggers are config-root and owned by the `RunContext`.** They
live at the top level, *not* inside the runner block. A runner config that
declares `callbacks` or `loggers` is rejected by `run_from_config`:

```yaml
runner:
  _target_: tpen.runner.Train
  model: ${model}
  sampler: ${sampler}
  hamiltonian_terms: ${hamiltonian_terms}
  optimizer: ${optimizer}
  trainer: ${trainer}

callbacks: [...]   # config-root, RunContext-owned
loggers: [...]     # config-root, RunContext-owned
```
