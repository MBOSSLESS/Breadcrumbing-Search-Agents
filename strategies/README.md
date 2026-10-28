# Paper strategies

These YAML files contain the strategies used in the paper. Each file holds one
strategy. Publication labels replace the original strategy names; all other
fields retain the recorded content.

| Experiment | Initial strategy for evolution | Frozen strategy for evaluation |
|---|---|---|
| ACH | — | `ach.yaml` |
| Normal TGSE | `ach.yaml` | `tgse/normal/<family>.yaml` |
| Trace-warm TGSE | `initial/trace_warm/<family>.yaml` | `tgse/trace_warm/<family>.yaml` |
| Adv-train TGSE | `ach.yaml` | `tgse/adv_train/<family>.yaml` |
| Adv-warm TGSE | `initial/adv_warm/<family>.yaml` | `tgse/adv_warm/<family>.yaml` |

The family names are `advertisement`, `bias`, `misinformation`, `harmful_output`,
and `prompt_injection`.

The ten files under `initial/` are the exact initial strategies recorded by the
published warm-start runs. Pass the relevant file to
`scripts/evolve_strategy.py --strategy`, together with the matching family and
regime configuration. They let you rerun warm-start evolution directly without
reconstructing the earlier strategy-preparation experiments. Adv-train uses the
verification-aware researcher setting; Adv-warm evolves its supplied seed under
the standard researcher setting.

`manifest.json` records hashes of ACH, the 20 training-selected result
strategies, and the three fixed strategy-substitution cards under `ablations/`. `initial_manifest.json` records hashes of the ten warm-start seeds.
Data and model credentials are supplied separately. These are frozen artifacts;
a new evolution run produces its own selected strategy.
