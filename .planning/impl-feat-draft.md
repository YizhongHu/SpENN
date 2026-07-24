---
name: impl-feat
description: Implement an accepted plan or executable contract through
  structured planner, implementer, validator, and critic deliberation. Use when
  production changes should be built, independently tested, and reconciled.
---

# Objective

Implement an accepted direction and validate that the result satisfies its
behavioral contract.

The implementer is the main actor and the only editing role.

The plan guides implementation but does not override evidence.

The user always has final authority.

Stop after implementation, independent validation, and final reconciliation.

# Inputs

Expect the user to provide some combination of

- task
- accepted plan or executable contract
- relevant sources
- constraints
- desired compatibility
- desired implementation philosophy

Inspect supplied sources before editing.

Do not assume unstated requirements.

If information is missing but not critical, make the smallest reasonable
assumption and state it explicitly.

If information is critical, ask.

# Workflow

Use Paseo's native orchestration tools.

Prefer reusing live agents from a preceding RPlan:

- planner remains planner
- implementation representative becomes implementer
- critic remains critic

Always use a separate validator.

If matching agents are unavailable, create them.

Keep agents in the current Paseo workspace.

Do not create worktrees.

Do not create additional workspaces.

Each agent receives

- the same task
- the same sources
- the same constraints

Each also receives its own objective.

The implementer may edit files.

The planner, validator, and critic remain read-only.

---

# Planner

Your objective is to help the implementer preserve the accepted design while
resolving concrete implementation questions.

Consider

- behavioral invariants
- compatibility requirements
- public surfaces
- architectural boundaries
- dependencies
- anti-goals
- unresolved user decisions

Answer targeted questions with the smallest sound constraint.

Do not edit files.

---

# Implementer

Your objective is to produce the smallest maintainable implementation that
satisfies the accepted contract.

Before editing, ask the planner about material ambiguities.

Consider

- existing architecture
- ownership boundaries
- compatibility
- failure behavior
- implementation complexity
- testability
- opportunities to simplify

You may revise implementation details when evidence supports the change.

State material deviations from the accepted plan.

You may edit implementation and test files.

---

# Validator

Your objective is to independently determine whether the implementation
satisfies the accepted contract.

Run the relevant tests after the implementer freezes edits.

Report

- exact commands
- results
- failures
- validation scope
- environment limitations

Do not infer success from the implementer's report.

Do not edit files.

---

# Critic

Your objective is to perform a final attempt to falsify the completed
implementation.

Evaluate

- correctness
- contract coverage
- compatibility
- maintainability
- consistency with supplied design philosophy
- unsupported deviations from the accepted direction

Classify findings as

- blocker
- acceptance gap
- optional improvement

Prefer concrete evidence over hypothetical concerns.

Do not require solving unrelated future problems.

Do not edit files.

---

# Deliberation

Let the implementer and planner discuss targeted questions during
implementation.

Do not require strict adherence to plan details when evidence supports a better
choice.

After the implementer freezes edits, run validator before critic.

Compare findings and identify only material disagreements.

Send concrete blockers and acceptance gaps back to the implementer.

Allow one focused correction round.

Repeat only validation and criticism affected by the correction.

If disagreement remains, summarize it and ask the user when needed.

Do not restart the whole discussion.

---

# Synthesis

The orchestrator is responsible for final reconciliation.

Do not vote.

Do not average opinions.

Exercise judgement.

Prefer the smallest sound implementation.

Reason from

1. reality and explicit user constraints
2. accepted plan and behavioral contract
3. supplied design principles
4. existing architecture and tests
5. implementation preferences

For each material deviation, record the evidence that supports it.

Distinguish

- completed requirements
- accepted deviations
- optional improvements
- unresolved questions

---

# Output

Produce the completed implementation and a concise handoff.

Always include

- changed behavior
- material design choices
- validation commands and results
- critic verdict
- deviations and unresolved decisions

Mention that full agent discussions remain available through Paseo.

Perform integration actions only when the user authorizes them.

# Stop

Stop after presenting the implemented change, validation evidence, and final
evaluation.

Do not

- weaken the accepted contract without user approval
- expand into unrelated work
- perform integration actions without user authorization
- continue autonomously after an unresolved user decision
