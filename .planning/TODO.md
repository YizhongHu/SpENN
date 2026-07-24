# SpENN — open threads (last updated 2026-07-24)

Tracked planning ledger. Keep unfinished tasks here; prune completed/history
once GitHub issues or PRs carry the durable record.

`experiments/toolkit-roadmap.md` is the canonical v4 sequence and acceptance
ledger. This file orders work across that roadmap and other repository threads.
Current repository state still has no `pair_stability_v4` surface or frozen
V4-0 reference/evidence.

## Current priority ladder

### P0 — close the V4-0 kickoff decisions

This is the next work. Do not start checkpoint-candidate feature code.

- [x] **Planning authority resolved 2026-07-24.** Planning records live under
      tracked `.planning/`; this file remains the priority ledger and
      `experiments/toolkit-roadmap.md` remains canonical v4 authority.
- [x] **Dedicated implementation branch created from `dev`.**
      `codex/experiment-v4-restructure` starts at `4eadb55`.
- [x] **V4 restructuring implementation resumed.** V4-0 is the first
      executable vertical slice; V4-1 is the first contract slice afterward.
- [x] **Static ten-stage bootstrap audit and RPlan complete.** Current v3 CLIs
      accept explicit roots, configs, profiles, and attempt lineage. The
      recommendation, ownership gates, fallback, tests, and implementation
      sequence are in
      [`v4-0-bootstrap-decision-memo.md`](v4-0-bootstrap-decision-memo.md).
- [x] **V4-0 bootstrap decisions approved 2026-07-24:** a v4-owned versioned
      dispatcher over pinned v3 subprocess CLIs, identity physical stage layout
      for V4-0, and tracked comparison inventory with deterministic compression
      for large tables. P0 architecture is closed; follow
      [`v4-0-implementation-instructions.md`](v4-0-implementation-instructions.md).
  - During implementation, exercise representative low- and fan-out stages in
    a disposable isolated root. If any ownership gate fails, use a sealed,
    source-only snapshot of the minimum v3 dependency closure.
  - Do not weaken the parity gate, edit v3, rewrite output, or accumulate
    stage-specific compatibility patches to preserve the preferred route.
- [x] **Reference shape and profile selected.** Use non-pilot v3 `smoke.yaml`
      on `gpu_test`.
- [x] **Existing matching v3 lineages audited.** Completed July smoke lineages
      prove workload feasibility but lack clean current-source provenance and
      use an older artifact/resource surface; none qualifies as canonical.
- [ ] Run exactly one fresh bounded v3 `smoke.yaml`/`gpu_test` reference from a
      clean recorded source revision in isolated staging. Require blind seed
      `811`, 64 scan jobs, one final replicate, complete ten-stage attempts,
      commit/dirty provenance, and checksums.

### P1 — implement and prove V4-0

- [ ] Follow the approved work packages, tests, fallback triggers, evidence
      order, and acceptance checklist in
      [`v4-0-implementation-instructions.md`](v4-0-implementation-instructions.md).
- [ ] Create the isolated `pair_stability_v4` bootstrap and guarded v4 root.
- [ ] Freeze the selected v3 reference through `reference.json`; add the
      reviewed one-to-one `layout_map.json`.
- [ ] Refactor the comparator and add negative header, row-order, metric,
      scientific-seed, map, and root-escape tests.
- [ ] Run the real ten-stage non-pilot v4 smoke on the matching profile and
      compare the complete mapped inventory. V4-0 bootstraps its own E2E gate;
      it does not depend on V4-1 through V4-5.

### P2 — land only the exercised foundational V4-1 contracts

- [ ] Add authoritative trial, seed, run, task/producer, metric, execution
      profile, and stage identity needed by the live v4 route. Keep records
      immutable, versioned, serializable, and independent of `spenn`, Hydra
      runtime state, and study-local modules.
- [ ] Decide producer-attempt semantics before checkpoint binding: one logical
      identity resumed in place, an ordered attempt lineage with explicit
      resolution, or a restarted stochastic trajectory represented as a new
      producer/trial.
- [ ] Defer unused protocols until an executable consumer exists. Every landed
      contract must round-trip, reject invalid input, import in a fresh process,
      and be read/written by a fresh full v4 smoke that preserves frozen-v3
      parity.

### P3 — choose the first post-foundation feature lane

- [ ] After V4-1, choose explicitly between:
  - the checkpoint-candidate contract lane, if a concrete first study is ready;
    or
  - generic V4-2 compact/stage composition, if no checkpoint study is ready.

Recent design work makes checkpoint contracts implementable next, but does not
replace this priority choice. If chosen, first record cadence and terminal-step
policy, seed grouping, objectives/aggregation, coverage and failure policy,
retry/equivalent-profile rules, capacity bounds, and v4 root/layout IDs. Then
follow the four small feature slices in
`.planning/checkpoint-selection-implementation-instructions.md`.

Physical custody remains later. The accepted artifact-system direction is
immutable experiment records, one finalizer, plain verified-copy reference
behavior, then a restic proof on durable storage. `$HOME` and `holystore01` are
durable tiers; `netscratch` is source/staging. Encryption, DataLad/git-annex,
and a database/dashboard are not current requirements. Before the custody
slice, produce the requested follow-up memo for measurable archive/restore,
growth, deduplication, failure-recovery, replica-count, and capacity gates.

### P4 — finish the v4 restructuring

- [ ] V4-2 generic compact interfaces and stage composition.
- [ ] V4-3 fixed-grid planning/materialization.
- [ ] V4-4 low-fan-out stage routing.
- [ ] V4-5 planning/fan-out routing and executor factory. Land required
      train-resume behavior before its requeue/resume acceptance gate.
- [ ] V4-6 full parity and cutover decision.
- [ ] V4-7 remove only duplication proven unreachable after parity.
- [ ] V4-8 adaptive search only for a concrete need and with explicit
      dependency approval.

### Other repository priorities

1. [x] Reimplement the Hooke basis as a genuine multidimensional harmonic
   oscillator basis with explicit truncation and ordering. Issue archaeology
   remains pending; do not invent an issue number.
2. [ ] Implement analytic envelope/cusp local-energy terms, retaining autodiff
   as the numerical reference. The current mathematical source is `main.typ`
   (Envelopes and Electron-electron cusp); issue archaeology is pending.
3. [ ] Profile per-step costs and optimize only after numerical-equivalence
   evidence exists. Issue archaeology is pending.
4. [ ] Undertake the major TPEN architectural migration and package/project
   rename for v0.3.0. [#125 Separate trainer, model, and sampler clocks in
   events and metrics](https://github.com/YizhongHu/SpENN/issues/125) remains
   reserved for this incompatible boundary.

Do not interleave these with an active V4 PR unless an explicit dependency
requires it. Sampler experiments, learning-rate/model-capacity tuning, and new
physical systems remain on hold until human reprioritization.

For every active implementation PR: define specification and oracle first; use
one implementation lane; run an independent critic and clean verifier; merge
only by human decision.

### Prohibited historical operation

- [ ] Ownership of Slurm job `33716357` and
      `/n/netscratch/kozinsky_lab/Everyone/rhu/SpENN-cluster-preflight-c13e070`
      remains unresolved. Never inspect, cancel, modify, delete, reuse, or cite
      either as evidence without explicit bounded authorization. This
      incident-specific prohibition does not block local work or new,
      task-scoped Slurm activity allowed by current repository instructions.

The human retains authority over architecture decisions, merges into `dev`,
and promotion to `main`.

## Event Clock Refactor (Breaking; Deferred)

Tracked as [#125 Separate trainer, model, and sampler clocks in events and
metrics](https://github.com/YizhongHu/SpENN/issues/125). Issue is source of
truth for design, interfaces, migration scope, and acceptance tests. Human
decision on 2026-07-20: reserve this migration for the TPEN breaking change into
v0.3.0 rather than the behavior-preserving v4 experiment lane.
- [ ] Implement only at an explicit breaking-version boundary.
- [ ] Replace ambiguous `Event.step`/metric `step` with explicit
      `trainer_step`, `model_step`, and `sampler_step`.
- [ ] Keep timing on `trainer_step`; use `model_step` for model-related
      metrics; attach both model and sampler clocks to sampler metrics.
- [ ] Persist/restore owned counters without deriving one from another.
- [ ] Keep RNG as serialized state. Do not add `rng_step`.
- [ ] Until migration, retain current zero-based event contract and PR #124's
      checkpoint-specific completed-update translation.

## Experiment Refactor Tracking

- [x] `experiments/toolkit-roadmap.md` is the canonical running guidance for
      v4 current surfaces, parity gates, decisions, and phase status.
  - [#113 Experiment Stack Design](https://github.com/YizhongHu/SpENN/issues/113)
    retains the refactoring manifesto and review discussion.
  - [#114 Experiment Refactor Plans](https://github.com/YizhongHu/SpENN/issues/114)
    remains supporting historical design/phase context.
- [x] Reviewed `experiments/toolkit-roadmap.md` and issue #113 on 2026-07-11:
      approve the governance/parity plan; keep v4 implementation paused until
      later explicit direction.
- [x] Every experiment-stack code/config change requires a fresh v4 E2E
      comparison against a frozen non-pilot v3 `configs/smoke.yaml` reference;
      do not rerun v3 after freezing it. Prefer `gpu_test`/`test`; matching
      local references are an allowed fallback. Layout/artifact renames require
      the reviewed one-to-one parity map.
- [x] Decided the core/experiment boundary for optimizer-initiated batch
      checkpoint selection on 2026-07-24. Core checkpoint interfaces stay
      unchanged; a strict experiment adapter observes exact completed
      checkpoints and owns batch/candidate/evaluation/selection contracts.
      `.planning/checkpoint-selection-boundary-decision.md` is the decision
      record; `.planning/checkpoint-selection-implementation-instructions.md`
      is the future handoff. Dynamic scheduling remains deferred.
- [x] Decided the artifact-system architecture on 2026-07-24. Immutable
      experiment records remain scientific authority; one finalizer owns
      custody; plain verified copy defines reference behavior; restic is the
      first external proof-of-concept candidate; metadata databases and
      dashboards remain optional projections. See
      `.planning/checkpoint-artifact-systems-decision.md`.
- [ ] Long-term target: SpENN-dev matures `pair_stability_v4`/
      `experiments/toolkit`; SpENN later adopts finished result and retires
      older homegrown experiment work. Real merge conflicts accepted at that
      future switchover.
- [ ] Preserve the useful PR #77 launcher-plumbing idea, but land it only as
      part of V4-5's reviewed executor-factory/routing slice rather than as an
      independent abstraction.
- [ ] Keep `pair_stability_v3` archival sync study-local until another completed
      study needs the same workflow or the custody slice starts. Then extract
      only a source-independent provenance/byte-budget plan; do not turn V3's
      bounded, checkpoint-excluding `10_sync` procedure into a shared executor.

## Repo Path Cleanup

- [ ] Old checkout at `/n/netscratch/kozinsky_lab/Everyone/rhu/SpENN` is clean,
      tracks `origin/codex/final-report-plot-polish`, and has two registered
      prunable `/tmp/rhu` worktrees. Decision on 2026-07-10: retain it for now;
      re-evaluate retirement only after its branch/worktree provenance is no
      longer needed.
- [x] Audited backup symlinks under
      `experiments/hooke/pair_stability_v2/results/_migration/backups/**`
      on 2026-07-10. The sole extant link was relative, not old-absolute;
      archived textual paths remain provenance and were left unchanged.
- [x] Repaired the cosmetic broken backup symlink
      `experiments/hooke/pair_stability_v2/results/_migration/backups/20260702T154006-0400_27305702/03_collect/latest`
      to `../../../../03_collect/20260702T143714-0400`; live result pointers
      were left untouched.

## Irrep-Space Activation

- [ ] Decide whether "irrep activation is uninformative" means all irrep-space
      nonlinearity or only redundancy with existing real-space gates
      (`RealNormGate`, `RealRMSGate`, `RealGaussianNormGate`).
- [ ] Decide whether v2 results need interpretation caveat: v2 mechanism axis
      has no "off" branch for irrep activation, and several arms are
      irrep-activation-gate ablations.
- [ ] Decide whether dropping irrep activation affects project naming. Current
      evidence: "SpENN" ties to Specht modules broadly, not activation alone.

## Log-Amplitude Reference Diagnostics

- [ ] Add exact/reference `logabs` values to cusp, tail, and symmetry
      diagnostic records so reports can compute meaningful relative logabs
      errors. Define normalization/offset convention explicitly before adding
      relative-error plots.

## QoL Backlog

- [ ] Add `spenn` console entry point:

      ```toml
      [project.scripts]
      spenn = "spenn.run:main"
      ```

      Then `uv run spenn ...` replaces `uv run python run.py ...`. Verify with
      `uv run spenn --help` and one smoke config. Keep `run.py` during
      transition; remove only after scripts/docs migrate.

## Train Resume / Job Restart

Tracked as [#50 Implement train resume](https://github.com/YizhongHu/SpENN/issues/50).
Issue is source of truth.

Remaining PR order:

- [ ] `auto` restore mode in `spenn/checkpoint/restore.py`: resume from
      `latest.json` if present, else fresh. Add CPU bitwise resume-equivalence
      integration test (`train(2N)` == `train(N)` + `resume(N)`).
- [ ] Requeue-able sbatch template plus stable checkpoint-root convention in
      `experiments/`; Hydra run dirs are per invocation, checkpoint root must
      be pinned. Document in `experiments/README.md`.
- [ ] Extend `_resume_overrides` from `final_train.py` to scan-stage
      `train.py`; also fixes latent `FileExistsError` when reclaimed scan rows
      restart into dirs with leftover `step_N` checkpoints.
- [ ] Persist per-execution records in attempt dirs, including
      `RestoreReport`/`checkpoint_restored`, instead of overwriting without
      trace.
- [ ] Deferred: W&B resume-id continuity.
