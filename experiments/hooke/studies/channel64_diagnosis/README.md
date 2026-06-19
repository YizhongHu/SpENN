# Channel-64 diagnosis

> **⚠️ LEGACY — not maintained against the current evaluation stack.**
>
> This study consumes the pair-validation collect/select outputs
> (`pair_validation/reports/02_collect/runs.csv`, `03_select/selection.csv`),
> which were produced by the pre-PR8.5 pipeline. The corresponding
> `pair_validation` study source was removed from the current branch because it
> depended on the retired diagnostics/probe stack, so this study no longer works
> end to end against the current PR8.5–PR8.7 evaluation stack.
>
> - **Last commit on the hooke branch where this study worked:**
>   `25360a6638d537fc10b526e70abb940c4d13e01d`
>   (`[codex] Hooke 8.4 evaluation diagnostics for physical correctness (#53)`).
> - Kept for reference only; migration to the new evaluation stack is deferred.

This study diagnoses why the 64-channel candidates in
`pair_validation/reports/02_collect` did not win final selection.

## Current conclusion

64 channels were not bad because they lacked a viable learning rate. The best
64-channel candidate,
`config_lr0-0003_channels64_layers1_gate_activationsigmoid`, had the lowest
median validation energy in `03_select`:

```text
median validation/energy = 2.00020567063
```

It lost because that energy lead over the selected 32-channel candidate was
inside the configured selection margin:

```text
selected 32-channel median validation/energy = 2.00105810686
64-channel energy lead                         = 0.00085243623
selection margin                               = 0.00302308735
```

The selected model also had lower median validation local-energy variance:

```text
selected 32-channel variance = 0.00199878980
best 64-channel variance     = 0.00268040744
```

The `02_collect` train-vs-validation gaps do not show a clear overfit
signature. Across 64-channel runs, the mean validation-minus-train energy gap is
`0.00016418185`; by LR, the largest mean gap is only `0.00199023002` at
`lr=3e-3`. Train and validation variances are also close at the end of
training. The stronger diagnosis is sensitivity: 64 channels work in a narrow
sigmoid/lr pocket, while other gate/LR combinations and some seeds are
substantially higher variance.

## Regenerate

```bash
uv run python experiments/hooke/studies/channel64_diagnosis/analyze.py
```

The script reads archived files from the removed legacy pair-validation study:

```text
experiments/hooke/studies/pair_validation/reports/02_collect/runs.csv
experiments/hooke/studies/pair_validation/reports/03_select/selection.csv
```

It writes:

```text
experiments/hooke/studies/channel64_diagnosis/reports/diagnosis.md
experiments/hooke/studies/channel64_diagnosis/reports/tables/channel_summary.csv
experiments/hooke/studies/channel64_diagnosis/reports/tables/channel_lr_summary.csv
experiments/hooke/studies/channel64_diagnosis/reports/tables/channel_gate_summary.csv
experiments/hooke/studies/channel64_diagnosis/reports/tables/channel64_candidates.csv
experiments/hooke/studies/channel64_diagnosis/reports/tables/channel64_seed_rows.csv
```

## Next experiment

Run a targeted width-64 follow-up rather than a broad sweep:

- Control: selected 32-channel sigmoid/lr=3e-3 config.
- Width-64 arm: sigmoid only, with a denser LR grid around 3e-4 to 1e-3.
- Use more seeds than the original three-seed validation scan.
- Add validation at multiple checkpoints if the goal is to test true overfit.
- Keep a variance or tail-probe guard in selection, because median sampled
  energy alone can hide fragile local-energy behavior.
