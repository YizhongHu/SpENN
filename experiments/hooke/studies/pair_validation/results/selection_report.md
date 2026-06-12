# Selection report: hooke_pair_validation_v1

- selection metric: `validation/energy` (median over seeds)
- failed/ineligible seeds count as `inf`; any failed seed fails the whole config (`require_all_seeds`)
- selection margin: `max(2.0 * sqrt(stderr_A^2 + stderr_B^2), 0.25 * max(iqr_A, iqr_B), 0.0001)`
- tie-breakers (in order): `validation/energy_variance`, `validation_energy_iqr`, `validation/energy_stderr`, `geometry_warning_count`, `model_params.channels`, `runtime/wall_time_sec`
- inputs: local run outputs only (no W&B, no exact reference energy)

## Ranking

| rank | config_id | optimizer_params.lr | model_params.channels | model_params.layers | model_params.gate_activation | score | stderr | iqr | variance | geom warns | failed seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lr=0.001_channels=32_layers=1_gate_activation=sigmoid | 0.001 | 32 | 1 | sigmoid | 2.00127 | 0.00114267 | 0.00141763 | 0.00267406 | 0 | 0/3 |
| 2 | lr=0.003_channels=8_layers=1_gate_activation=sigmoid | 0.003 | 8 | 1 | sigmoid | 2.00134 | 0.00120099 | 0.00865635 | 0.002954 | 0 | 0/3 |
| 3 | lr=0.001_channels=8_layers=1_gate_activation=silu | 0.001 | 8 | 1 | silu | 2.00063 | 0.00148622 | 0.00038406 | 0.00452371 | 0 | 0/3 |
| 4 | lr=0.003_channels=8_layers=1_gate_activation=silu | 0.003 | 8 | 1 | silu | 2.00058 | 0.00155215 | 0.000775581 | 0.00493397 | 0 | 0/3 |
| 5 | lr=0.003_channels=32_layers=1_gate_activation=sigmoid | 0.003 | 32 | 1 | sigmoid | 2.00314 | 0.00195026 | 0.00117934 | 0.00778962 | 0 | 0/3 |
| 6 | lr=0.0003_channels=8_layers=1_gate_activation=silu | 0.0003 | 8 | 1 | silu | 2.00168 | 0.00203491 | 0.000930104 | 0.00848044 | 0 | 0/3 |
| 7 | lr=0.0003_channels=8_layers=1_gate_activation=sigmoid | 0.0003 | 8 | 1 | sigmoid | 2.00334 | 0.00249184 | 0.000535692 | 0.0127166 | 0 | 0/3 |
| 8 | lr=0.001_channels=8_layers=1_gate_activation=sigmoid | 0.001 | 8 | 1 | sigmoid | 2.00446 | 0.00258375 | 0.00410951 | 0.013672 | 0 | 0/3 |
| 9 | lr=0.001_channels=32_layers=1_gate_activation=silu | 0.001 | 32 | 1 | silu | 2.00638 | 0.0029628 | 0.00487689 | 0.0179778 | 0 | 0/3 |
| 10 | lr=0.0003_channels=32_layers=1_gate_activation=silu | 0.0003 | 32 | 1 | silu | 2.01265 | 0.00345363 | 0.0431782 | 0.0244276 | 0 | 0/3 |
| 11 | lr=0.0003_channels=32_layers=1_gate_activation=sigmoid | 0.0003 | 32 | 1 | sigmoid | 2.02392 | 0.00422665 | 0.0123151 | 0.0365867 | 0 | 0/3 |
| 12 | lr=0.003_channels=32_layers=1_gate_activation=silu | 0.003 | 32 | 1 | silu | 2.02584 | 0.00434542 | 0.657007 | 0.0386717 | 0 | 0/3 |
| 13 | lr=0.001_channels=128_layers=1_gate_activation=sigmoid | 0.001 | 128 | 1 | sigmoid | inf | 0.00123811 | 0.000274871 | 0.00363964 | 0 | 1/3 |
| 14 | lr=0.0003_channels=128_layers=1_gate_activation=sigmoid | 0.0003 | 128 | 1 | sigmoid | inf | 0.00270679 | 0 | 0.0150051 | 0 | 2/3 |
| 15 | lr=0.003_channels=128_layers=1_gate_activation=silu | 0.003 | 128 | 1 | silu | inf | 0.0057049 | 0 | 0.066654 | 0 | 2/3 |
| 16 | lr=0.0003_channels=128_layers=1_gate_activation=silu | 0.0003 | 128 | 1 | silu | inf | inf | 0 | inf | 0 | 3/3 |
| 17 | lr=0.001_channels=128_layers=1_gate_activation=silu | 0.001 | 128 | 1 | silu | inf | inf | 0 | inf | 0 | 3/3 |
| 18 | lr=0.003_channels=128_layers=1_gate_activation=sigmoid | 0.003 | 128 | 1 | sigmoid | inf | inf | 0 | inf | 0 | 3/3 |

## Selected

`lr=0.001_channels=32_layers=1_gate_activation=sigmoid` with median `validation/energy` = 2.00127

Seed energies: 3: 2.00127, 9: 1.99994, 11: 2.00278

### Selection margin and tie-breakers

10 candidates were within the selection margin of the best median energy; the tie was decided by `validation/energy_variance`.

Margins vs best (a candidate is tied when its score is within this margin):

- `lr=0.0003_channels=8_layers=1_gate_activation=silu`: margin = 0.0051186
- `lr=0.001_channels=8_layers=1_gate_activation=silu`: margin = 0.00429791
- `lr=0.003_channels=8_layers=1_gate_activation=silu`: margin = 0.00439014
- `lr=0.001_channels=32_layers=1_gate_activation=silu`: margin = 0.00668951
- `lr=0.003_channels=32_layers=1_gate_activation=silu`: margin = 0.164252
- `lr=0.0003_channels=8_layers=1_gate_activation=sigmoid`: margin = 0.00587144
- `lr=0.001_channels=8_layers=1_gate_activation=sigmoid`: margin = 0.00602824
- `lr=0.003_channels=8_layers=1_gate_activation=sigmoid`: margin = 0.00392507
- `lr=0.001_channels=32_layers=1_gate_activation=sigmoid`: margin = 0.00385479
- `lr=0.003_channels=32_layers=1_gate_activation=sigmoid`: margin = 0.00498505

| config_id | score | validation/energy_variance | validation_energy_iqr | validation/energy_stderr | geometry_warning_count | model_params.channels | runtime/wall_time_sec |
|---|---|---|---|---|---|---|---|
| lr=0.001_channels=32_layers=1_gate_activation=sigmoid | 2.00127 | 0.00267406 | 0.00141763 | 0.00114267 | 0 | 32 | 2352.66 |
| lr=0.003_channels=8_layers=1_gate_activation=sigmoid | 2.00134 | 0.002954 | 0.00865635 | 0.00120099 | 0 | 8 | 1442.13 |
| lr=0.001_channels=8_layers=1_gate_activation=silu | 2.00063 | 0.00452371 | 0.00038406 | 0.00148622 | 0 | 8 | 1506.77 |
| lr=0.003_channels=8_layers=1_gate_activation=silu | 2.00058 | 0.00493397 | 0.000775581 | 0.00155215 | 0 | 8 | 2324.75 |
| lr=0.003_channels=32_layers=1_gate_activation=sigmoid | 2.00314 | 0.00778962 | 0.00117934 | 0.00195026 | 0 | 32 | 1448.45 |
| lr=0.0003_channels=8_layers=1_gate_activation=silu | 2.00168 | 0.00848044 | 0.000930104 | 0.00203491 | 0 | 8 | 2359.48 |
| lr=0.0003_channels=8_layers=1_gate_activation=sigmoid | 2.00334 | 0.0127166 | 0.000535692 | 0.00249184 | 0 | 8 | 2270.33 |
| lr=0.001_channels=8_layers=1_gate_activation=sigmoid | 2.00446 | 0.013672 | 0.00410951 | 0.00258375 | 0 | 8 | 2332.5 |
| lr=0.001_channels=32_layers=1_gate_activation=silu | 2.00638 | 0.0179778 | 0.00487689 | 0.0029628 | 0 | 32 | 1452.99 |
| lr=0.003_channels=32_layers=1_gate_activation=silu | 2.02584 | 0.0386717 | 0.657007 | 0.00434542 | 0 | 32 | 2287.59 |

## Sampler geometry diagnostics

Winner geometry (informational; geometry only decides via the `geometry_warning_count` tie-breaker):

| seed | radius_mean | radius_q99 | radius_max | electron_distance_q01 | electron_distance_min | position_rms |
|---|---|---|---|---|---|---|
| 3 | 1.73780361398 | 3.58661556968 | 4.70849319168 | 0.619792217441 | 0.185803838882 | 1.08478103319 |
| 9 | 1.72922982846 | 3.55588449227 | 5.07904055961 | 0.682355907167 | 0.283546267886 | 1.08007786273 |
| 11 | 1.74017589691 | 3.59301555082 | 4.84439602777 | 0.705324266258 | 0.271094027062 | 1.08635373308 |

No suspicious walker geometry flagged.
