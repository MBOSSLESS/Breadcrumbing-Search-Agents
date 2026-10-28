# Reproducing the experiments

## Prepare the task splits

```bash
python scripts/prepare_data.py
```

The script downloads the pinned public SafeSearch files, checks their hashes,
and applies the published indices. It writes four directories under
`data/safesearch/`:

| Directory | Cases | Use |
|---|---:|---|
| `train` | 100 | Strategy evolution and baseline memory preparation |
| `test` | 187 | Main held-out comparison |
| `runtime_test20` | 100 | Runtime ablations and the four dynamic baseline comparisons |
| `tgse_test20` | 100 | Frozen TGSE strategy evaluation |

The two 100-case evaluation subsets contain different task IDs. They are
subsets of the held-out split, not new training data. See [data preparation](../data/README.md)
for the source revision, exclusion indices, and local-source option.
Synthetic records in `examples/` are only for small workflow demonstrations.

## Experiment map

Run all commands from the repository root. The configurations already point
to the prepared data paths.

| Experiment | Configuration | Entry point |
|---|---|---|
| ACH | `configs/ach.yaml` | `run_experiment.py` |
| No attack, AgentDojo, GEO, SafeSearch | matching YAML in `configs/baselines/` | `run_experiment.py` |
| AiTM, MAST, Intent Hijacking | matching YAML in `configs/baselines/` | `run_experiment.py` |
| Objective Drift | `configs/baselines/objective_drift.yaml` | prepare memory, then run |
| Evo-Attacker | `configs/baselines/evo_attacker_train.yaml` and `evo_attacker.yaml` | train, judge, build memory, then run |
| ACH runtime removals and strategy substitutions | matching YAML in `configs/ablations/` | `run_experiment.py` |
| Normal TGSE | `configs/tgse.yaml`, `strategies/ach.yaml` | `evolve_strategy.py` |
| Adv-train TGSE | `configs/tgse_adv_train.yaml`, `strategies/ach.yaml` | `evolve_strategy.py` |
| Warm-start TGSE | `configs/tgse.yaml`, matching `strategies/initial/` seed | `evolve_strategy.py` |
| Frozen TGSE evaluation | matching YAML in `configs/evaluation/` | `run_experiment.py` |

For example:

```bash
python scripts/run_experiment.py --config configs/ach.yaml --output-dir runs/ach
python scripts/judge_results.py runs/ach --judge-model qwen3 --workers 1
python scripts/summarize_results.py runs/ach --judge-model qwen3
```

Generation must finish before judging the same output directory. Final summaries
require complete case-by-repeat results and valid judgments.

`strategy_map` assigns a frozen strategy to each task family. Evaluation does
not select or rerank strategies using held-out outcomes.

For a model-transfer experiment, change the intended role explicitly:

```bash
python scripts/run_experiment.py --config configs/ach.yaml \
  --researcher-model kimi --attacker-model qwen3 --output-dir runs/ach-kimi
```

Choose the same evaluation cohort and settings as the comparison being reproduced.
Changing the model alone does not select a different task subset.

## Baseline memory

Objective Drift and Evo-Attacker use training-derived memory during held-out
evaluation. Prepare it from `data/safesearch/train`; do not train memory on test
cases. The full commands are in [experiment interfaces](experiments.md) and
[baseline adaptations](baselines.md).

Objective Drift prepares one memory per task family. Its evaluation configuration
contains one family and its corresponding memory file. Evo-Attacker builds a
frozen success-only memory across the five training families and requires the
optional `evo` dependencies. Neither baseline's real training memory is
redistributed.

## Strategies and settings

[Strategy documentation](../strategies/README.md) separates the ACH strategy,
TGSE initial seeds, frozen training-selected strategies, and strategy-substitution
cards. Use the frozen strategies to evaluate the published artifacts directly;
use the corresponding initial seed to rerun evolution.

The TGSE preset describes the paper's main settings. New evolution runs need not
regenerate the exact frozen strategy text: individual experiments used differing
budgets or evolution models, and model APIs are stochastic and can change.

Researcher temperature is `0.0`; ACH and the dynamic baseline attacker temperature
is `0.5`. SafeSearch uses `0.8`. Configurations expose repetitions, worker limits,
and model identities explicitly. The runtime-removal configurations independently
remove reflection from rendered memory, memory from the planner input, or the
strategy input; they are not observation-visibility switches.

`prompt_date` fixes the researcher's system-prompt date. It does not freeze all
attack-prompt dates or external web contents. `tokenizer_path` optionally selects
a local tokenizer; otherwise token counts are estimated from characters.

## Reporting

Report ASR, MaxN ASR, SS / Flip / SF, task cohort, case counts, repetitions,
model identities, settings, and technical failures. Overall ASR and MaxN ASR
are macro-averages over families; summaries also expose pooled rates.
The per-family best row across paper settings is a diagnostic envelope,
not a single trained strategy configuration.
