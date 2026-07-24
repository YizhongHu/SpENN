---
name: impl-test
description: Implement a test-only executable contract through structured
  planner, test-implementer, and critic deliberation. Use when observable
  behavior should be fixed in tests before production implementation.
---

# Objective

Turn an accepted direction into an independent, executable contract before
production implementation.

The test implementer is the main actor and the only editing role.

The user always has final authority.

Stop after implementing and evaluating the test contract.

Never modify production code.

# Inputs

Expect the user to provide some combination of

- task
- accepted plan or requested behavior
- relevant sources
- constraints
- desired compatibility
- desired testing philosophy

Inspect supplied sources before editing.

Do not assume unstated requirements.

If information is missing but not critical, make the smallest reasonable
assumption and state it explicitly.

If information is critical, ask.

# Workflow

Use Paseo's native orchestration tools.

Prefer reusing live agents from a preceding RPlan:

- planner remains planner
- implementation representative becomes test implementer
- critic remains critic

If matching agents are unavailable, create them.

Keep agents in the current Paseo workspace.

Do not create worktrees.

Do not create additional workspaces.

Each agent receives

- the same task
- the same sources
- the same constraints

Each also receives its own objective.

Do not allow agents to see each other's initial analysis.

The test implementer may edit test files and test-owned fixtures.

The planner and critic remain read-only.

---

# Planner

Your objective is to identify the smallest observable contract that captures
the requested behavior.

Consider

- public behavior
- compatibility requirements
- invalid inputs
- boundary cases
- testable invariants
- anti-goals
- unresolved user decisions

Do not edit files.

---

# Test Implementer

Your objective is to implement a test suite that expresses the accepted
contract independently of the production implementation.

Consider

- independent oracles
- meaningful failure messages
- regression coverage
- test isolation
- collection and execution cost
- expected failures before implementation

Challenge requirements that cannot be observed or tested reliably.

You may edit test files and test-owned fixtures.

Do not modify production code.

---

# Critic

Your objective is to attempt to falsify the test contract.

Evaluate

- whether assertions prove the intended behavior
- whether expected values depend on the implementation under test
- whether important failure or boundary cases are missing
- whether tests can pass for an incorrect implementation
- whether failures could come from unrelated environment assumptions

Prefer concrete evidence over hypothetical concerns.

Keep criticism constructive and actionable.

Do not edit files.

---

# Deliberation

Wait for all initial analyses before comparing them.

Identify only material disagreements.

Ignore stylistic differences.

Let the test implementer ask the planner targeted questions.

Treat the accepted plan as guidance, not a rigid script.

Send concrete test defects from the critic back to the test implementer.

Allow one focused correction round.

If disagreement remains, summarize it and ask the user when needed.

Do not restart the discussion.

---

# Synthesis

The orchestrator is responsible for reconciliation.

Do not vote.

Do not average opinions.

Exercise judgement.

Prefer the smallest sound test contract.

Distinguish

- required behavior
- recommended coverage
- optional improvements
- unresolved questions

Preserve independent expected values and explicit assumptions.

---

# Output

Produce the completed test change and a concise handoff.

Always include

- behavior covered
- test commands and results
- expected failures caused by missing production behavior
- assumptions and unresolved decisions

Mention that full agent discussions remain available through Paseo.

# Stop

Stop after presenting the implemented test contract and evidence.

Do not

- modify production code
- weaken accepted behavior to make tests pass
- expand into unrelated coverage
- perform integration actions without user authorization
