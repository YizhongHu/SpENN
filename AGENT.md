# SpENN project specific guidelines

## Environment

- Any environment problems is not worth trouble-shooting by the agent on its own. If it happens, stop and the issue will be resolved interactively.
- This repo uses `uv` to manage python packages. Most commands (including `pytest`) needs to be run with `uv`.

## Conventions
- NumpyDoc is used for documentation
- Use inline comments for comprehensibility

## Plans and Todos

- Each folder under root contains `TODO.md`. This is a plan for the current directory. Maintain this TODO list to keep information between agents.
- Each folder under root contains `instructions.md`. This states the detailed design of everything in the current directory. Reference this for
  implementation details.