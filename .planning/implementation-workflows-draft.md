# Draft: Standalone Implementation Skills

`impl-test` and `impl-feat` are standalone skills shaped like RPlan. They use
named agent roles, independent initial analysis, targeted deliberation,
orchestrator synthesis, and explicit stop conditions.

- [impl-test skill draft](impl-test-draft.md): define executable behavior
  without changing production code.
- [impl-feat skill draft](impl-feat-draft.md): implement an accepted contract
  and independently verify it.

They may consume an accepted RPlan output and reuse its surviving agents, but
remain separate workflows with editing authority and their own stop conditions.
