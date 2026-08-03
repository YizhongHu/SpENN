# SpENN — open threads (last updated 2026-08-03)

Untracked scratch note. Do not commit. Keep unfinished tasks here; prune
completed/history once GitHub issues or PRs carry the record.


## Implementation priorities after synchronization

Cluster synchronization is complete. This section is the common point of
reference for focused implementation order. Do not mark any priority complete
until its gates are satisfied, and do not change this ordering without a
documented dependency argument.

1. [ ] Reimplement the Hooke basis as a genuine multidimensional harmonic
       oscillator basis with explicit truncation and ordering. Issue archaeology
       is pending; do not invent an issue number.
2. [ ] Implement analytic envelope/cusp local-energy terms, retaining autodiff
       as the numerical reference. The current mathematical source is
       `main.typ` (Envelopes and Electron-electron cusp); issue archaeology is
       pending.
3. [ ] Profile per-step costs and optimize only after numerical-equivalence
       evidence exists. Issue archaeology is pending.
4. [ ] Undertake the major TPEN architectural migration and package/project
       rename. Issue archaeology is pending.
5. [ ] Only after priority 4 and fresh human direction, migrate to runner →
       events → callbacks → loggers while preserving current experiment
       behavior. [#113 Experiment Stack Design](https://github.com/YizhongHu/SpENN/issues/113)
       is the verified experiment-refactor source of truth.

Every priority has the same required gates, in order:

- [ ] Specification critic.
- [ ] Oracle/test contract.
- [ ] Single implementation lane.
- [ ] Independent PR critic.
- [ ] Clean verifier.
- [ ] Human merge.

Explicitly on hold until reordered by the human with a documented dependency
argument:

- [ ] Sampler experiments.
- [ ] Learning-rate and model-capacity tuning.
- [ ] New physical systems.

The human retains authority over architecture decisions, merges into `dev`,
and promotion to `main`.

## Event Clock Refactor (Breaking; Deferred)

Tracked as [#125 Separate trainer, model, and sampler clocks in events and
metrics](https://github.com/YizhongHu/SpENN/issues/125). Issue is source of
truth for design, interfaces, migration scope, and acceptance tests.

- [ ] Implement only at an explicit breaking-version boundary.
- [ ] Replace ambiguous `Event.step`/metric `step` with explicit
      `trainer_step`, `model_step`, and `sampler_step`.
- [ ] Keep timing on `trainer_step`; use `model_step` for model-related
      metrics; attach both model and sampler clocks to sampler metrics.
- [ ] Persist/restore owned counters without deriving one from another.
- [ ] Keep RNG as serialized state. Do not add `rng_step`.
- [ ] Until migration, retain current zero-based event contract and PR #124's
      checkpoint-specific completed-update translation.

## Experiment Refactor Tracking (Deferred until after TPEN)

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
- [ ] Decide the core/experiment boundary for optimizer-initiated batch
      checkpoint selection. `experiments/checkpoint-selection-options.md` records
      the requirements. Dynamic scheduling is deferred: first scope only
      serializable checkpoint/batch/candidate/evaluation/selection interfaces
      and pure policy boundaries, not a poller, queue, worker loop, controller,
      database, or runtime evaluation materializer.
- [ ] Long-term target: SpENN-dev matures `pair_stability_v4`/
      `experiments/toolkit`; SpENN later adopts finished result and retires
      older homegrown experiment work. Real merge conflicts accepted at that
      future switchover.
- [ ] Re-implement useful PR #77 idea in SpENN-dev when toolkit work resumes:
      extract launcher plumbing into shared `toolkit/launching.py`.
- [ ] Keep `pair_stability_v3` archival sync study-local until another completed
      study needs the same workflow. The next structural slice should extract
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
