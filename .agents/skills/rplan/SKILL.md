---
name: rplan
description: Refine plans for nontrivial technical work before implementation using multiple Codex subagents coordinated through Paseo. Use when the user wants to design or scope a change, especially for software architecture, research code, workflows, or infrastructure. Produces a concise recommendation for human approval and never performs implementation.
---

# Objective

Develop a realistic implementation plan before any code is written.

This workflow exists to improve planning quality through structured discussion,
not through majority voting or lengthy debate.

The user always has the final authority.

Stop after presenting a planning recommendation.

Never implement code.

---

# Inputs

Expect the user to provide some combination of

- task
- repository
- relevant issues
- design documents
- constraints
- desired philosophy

Inspect the supplied sources before proposing a plan.

Do not assume unstated requirements.

If information is missing but not critical, make the smallest reasonable
assumption and state it explicitly.

If information is critical, ask.

---

# Workflow

Create three subagents.

Do not create worktrees.

Do not create additional workspaces.

The three subagents are

- planner
- implementation representative
- critic

Each agent receives

- the same task
- the same sources
- the same constraints

Each also receives its own objective.

Wait for all three responses before continuing.

Do not allow agents to see each other's responses until their initial analysis
is complete.

---

# Planner

Your objective is to produce the smallest realistic plan that satisfies the
requested goal.

Consider

- design intent
- desired behavior
- function/class surfaces & docstrings
- testing strategy
- dependencies
- implementation sequence
- user decisions
- anti-goals
- known risks

Do not implement code.

---

# Implementation Representative

Your objective is to determine whether the proposed direction is realistically
implementable in the current project.

Consider

- hidden dependencies
- migration cost
- implementation complexity
- testing feasibility
- opportunities to simplify

Challenge unrealistic assumptions.

Do not implement code.

---

# Critic

Your objective is to attempt to falsify the proposal.

Evaluate

- correctness
- adequacy of proposed tests
- consistency with supplied design philosophy, e.g.
  - maintainability
  - readability
  - reproducibility
  - debuggability

Prefer concrete evidence over hypothetical concerns.

Do not reject a proposal merely because it is incomplete.

Do not require solving unrelated future problems.

Keep criticisms constructive and actionable.

Do not implement code.

---

# Deliberation

Compare the three analyses.

Identify only material disagreements.

Ignore stylistic differences.

Send targeted follow-up prompts only to the relevant agents.

Do not restart the discussion.

Allow one follow-up round.

If disagreement remains after one round,

summarize it,

do not continue debating.

---

# Synthesis

The orchestrator is responsible for the final recommendation.

Do not vote.

Do not average opinions.

Exercise judgement.

Prefer the smallest sound plan.

Preserve important disagreement.

Discard resolved discussion.

Distinguish

- requirements
- recommendations
- optional improvements
- unresolved questions

Optimize for human attention.

The user should understand the design and recommendation in a few minutes.

Do not require the user to inspect the full deliberation.

## Decision hierarchy

Reason from the following layers:

1. Reality and explicit task constraints
2. Design principles
3. Accepted project architecture and decisions
4. Relevant existing implementations and tests
5. Task-specific choices

Treat existing code as precedent, not automatic authority. Classify relevant
precedents as established, persuasive, weak, or legacy.

For each material recommendation, explain:
- which higher-level principle or architectural rule supports it;
- which concrete precedents support or challenge it;
- whether the plan follows, distinguishes, or deliberately replaces them.

Infer new principles only from repeated coherent decisions, not from one local
implementation.

---

# Output

Produce a concise planning recommendation in terms of an `.md` file in an untracked directory.

The exact format is task-dependent.

Always include

- definition of terms/concepts
- recommended direction
- rationale
- important tradeoffs
- evidence required before implementation
- unresolved decisions requiring the user

The recommendation should support the user's decision rather than make the
decision for them.

---

# Stop

Stop after presenting the recommendation.

Do not

- edit files
- create/remove worktrees
- create/remove branches
- create pull requests
- implement 
- continue autonomously without another user instruction