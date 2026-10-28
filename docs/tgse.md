# Trace-Guided Strategy Evolution

TGSE evolves the natural-language strategy supplied to the attacker. The attacker harness and the researcher stay fixed. Each run concerns one task family and uses only its training split to generate and select strategies.

## Run

Configure model access and Search/Visit services as described in the root README. Prepare the training split with `scripts/prepare_data.py` and select an initial strategy. Frozen strategy artifacts accompany the release; raw experimental trajectories are not distributed.

```bash
python scripts/evolve_strategy.py \
  --config configs/tgse.yaml \
  --train-data data/safesearch/train/Advertisement-1.jsonl \
  --strategy strategies/ach.yaml \
  --output-dir outputs/tgse-advertisement
```

To evaluate the train-selected strategy immediately afterward, add:

```bash
  --test-data data/safesearch/test/Advertisement-1.jsonl --test-cases 20
```

For a cross-checking-trained strategy evaluated under the standard researcher, add `--test-researcher-mode off`; `both` runs standard and cross-checking suites. The default `inherit` uses the training researcher mode.

Training and test case IDs must be disjoint. Each JSONL record follows the dataset format in [experiments.md](experiments.md). The initial YAML contains one strategy with `name`, `description`, `goal`, and ordered `steps`; each step contains either `search` or `visit`. A warm start uses the same interface with a different initial YAML.

`--config` accepts fields of `TGSEConfig` in `breadcrumbing/tgse/config.py`. The default scientific settings are 20 training cases, 10 cheap-stage cases, five repeats, four generations, three children per generation, and at most two full-stage children. Researcher, attacker, judge, and compression use the `qwen3` model alias; evolution uses `qwen3.6-plus`. Initial temperatures are respectively 0, 0.5, 0, 0, and 0.5; compression retries may increase their temperature. The researcher has at most 10 LLM calls per trajectory. A technical generation failure allows up to three attempts (`max_attempts`); these attempts are separate from the five statistical repeats. Set `workers` and `judge_workers` explicitly for your endpoint; both default to one.

You can override the config directly with `--researcher-model`, `--attacker-model`, `--evolution-model`, `--judge-model`, `--compression-model`, `--workers`, and `--repeats`. `--evolution-model` sets the config field `reflection_model`. Supply a configured alias or an explicit `backend:model` identifier supported by the model client.

The public model aliases are configurable provider model IDs. Identical aliases do not by themselves establish equivalence to a historical hosted model deployment.

The default two-stage gate, lesson memory, and full-stage budget implement the
reported TGSE search procedure. Paper runtime-removal and strategy-substitution
presets are in `configs/ablations/`; their semantics are described in
[experiments.md](experiments.md).

## Search procedure

1. Sample a fixed training slice and evaluate the initial strategy across all repeats. Compress its execution traces into structured feedback, including successful and failed repeats of the same case.
2. Select one parent from the archive. Its sampling weight is `sigmoid(10 * (mean_training_ASR - 0.5)) / (1 + archived_child_count)`.
3. Generate three complete replacement strategies using the parent, its trace feedback, ancestry, and accumulated positive/negative lessons.
4. Evaluate each child on the same cheap subset. With stable failures available, the ten-case subset targets two stable successes, six mixed outcomes, and two stable failures; otherwise it targets three stable successes and seven mixed outcomes. Mixed cases are balanced across success-rate bands below 0.3, from 0.3 to below 0.7, and at least 0.7. Insufficient buckets are backfilled from the remaining training cases.
5. Reject a cheap result when mean per-case improvement is below −0.05, or negative with at least 40% worsened cases, or negative with at least two cases dropping from a middle/high band into the low band. Rank passing children by improvement, then fewer regressions. When none passes, a child with improvement at least −0.1 may fill the default minimum of one full evaluation. The maximum remains two.
6. Evaluate promoted children on the remaining training cases, reusing their cheap-stage results. Every child that completes full evaluation enters the archive, even if its full score is below its parent's. Cheap-stage rejections supply negative lessons; full-evaluated children supply positive lessons. Children pruned solely for budget do not supply lessons. Each sign retains up to 12 recent unique lessons.
7. Select the archived strategy with the highest mean full training ASR. Ties retain the earliest archive entry. Optional held-out evaluation runs this frozen strategy without compression or updating the archive.

The archive records a tree through parent IDs. Multiple generations can extend different branches. The implementation uses an explicit generation loop, with no recursive execution or value backpropagation.

## Outputs and resuming

`selection.json` records the chosen archive entry and training score; `selected_strategy.yaml` is the frozen strategy for later evaluations. `state.json` contains the archive, lessons' source entries, and completed generation count. Per-generation directories retain the parent choice, cheap subset, promotion decision, proposals, evaluated trajectories, and compression feedback. These are local run outputs and may contain sensitive user-supplied task content.

Re-run the identical command to resume. The output directory is locked against concurrent evolution writers, and its input fingerprint must match. Choose a new directory when changing data, strategy, models, or budgets. Completed stages and model proposals are reused. Generation/judge failures stop the stage; they are not silently counted as unsuccessful attacks. Trace compression may omit at most 20% of cases per repeat, matching the pipeline's configurable feedback availability threshold.

The researcher prompt date is frozen for the entire evolution run and inherited by held-out evaluation. Set `prompt_date` to a quoted ISO string (for example, `"2026-01-02"`) to reproduce a chosen date; an omitted value is recorded on first launch and restored on resume.

Model generation is stochastic, and external search pages can change. The sampler seed fixes the training slice and persists archive sampling state across a restart; it does not make hosted LLM execution deterministic. Providing your own data reproduces the algorithm and reporting workflow, not the original paper's numeric results.

## Code map

- `search.py`: archive loop, parent selection, lessons, train-only selection.
- `selection.py`: cheap subset, case-level comparisons, promotion rules.
- `evaluation.py`: repeated runtime/judge calls, compression, stage merging.
- `reflection.py`: complete-strategy proposals and lesson extraction.
- `feedback.py`, `compression.py`: trajectory and cross-generation feedback.
- `prompting*.py`, `compression_prompt.py`, `schemas.py`: the model-facing contracts.
- `strategy.py`: strategy validation and change representation.

The Python entry point is `evolve(train_cases, initial_strategy, TGSEConfig(...), output_dir)`. Held-out evaluation is separately available as `evaluate_selected(test_cases, selection, config, output_dir)`.
