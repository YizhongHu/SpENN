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

## Plans and Todos

- Each folder under root contains `TODO.md`. This is a plan for the current directory. Maintain this TODO list to keep information between agents.
- Each folder under root contains `instructions.md`. This states the detailed design of everything in the current directory. Reference this for
  implementation details.

## Tools
- Always encouraged to spawn subagents to go faster for reading, editing, testing, 
running, and debugging if possible to do parallel.
- Always encouraged to initiate slurm runs for parallizability. Keep slurm logs around
for reproducibility.

## Best Practises
- Use existing libraries if possible
- Vectorize with NumPy/PyTorch if possible
- Code that is reused or can potentially be reused should be refactored
- Whenever you use a helper, think whether a potential other implementation would need it,
  and refactor pre-emptively
- Functions that manipulate data should be closer to data rather than 
  being defined only where it is needed
- If a config or file or function or class is no longer used, remove it.