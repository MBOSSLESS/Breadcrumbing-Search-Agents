# Running experiments

Run commands from the release root. Prepare the benchmark with `scripts/prepare_data.py` and configure provider credentials through environment variables. Result trajectories and trained cache assets are not distributed. These commands reproduce the implementation on supplied tasks; they do not independently reproduce the paper's numerical results without its data and artifacts.

Each JSONL record must contain a unique `test_case_id`, `user_query`, `user_expectation`, `target_consequence` and `risk_verification_checklist` (a string or list of strings). The configuration's `datasets` mapping supplies the family name. Keep IDs stable across repeats. No rows are silently skipped when input JSON is malformed.

```yaml
datasets:
  example: examples/test.jsonl
method: ach
strategies_path: strategies/ach.yaml
researcher_model: qwen3
attacker_model: qwen3
judge_model: qwen3
temperature: 0.0
attacker_temperature: 0.5
max_llm_calls_per_run: 10
max_attempts: 1
repeats: 5
workers: 1
judge_workers: 1
attacker_config: {}
```

Paths are relative to the release root/current working directory. Select `method: no_attack` for the unattacked condition. Frozen evolved strategies use `ach` with a different `strategies_path`; runtime ablations use the documented attacker configuration flags. Model transfer changes `researcher_model`; keep other experimental conditions explicit. Baselines that require manifests or memory must first build those artifacts on the supplied training data.

```bash
python scripts/run_experiment.py --config configs/examples/demo.yaml --output-dir runs/example
python scripts/judge_results.py runs/example --judge-model qwen3 --workers 1
python scripts/summarize_results.py runs/example --judge-model qwen3
```

Generation and judging use one global process pool each and the same exclusive result-directory lock. Only the parent process writes `results.jsonl`, atomically after each completed case. Run judging after generation has ended. Resume checks a manifest containing the configuration, dataset content digest and strategy digest. Changing workers is allowed; changing experimental parameters requires a new output directory. `--retry-failed` reruns failed generation rows and discards their old judgment. Judge reruns fill invalid/missing judgments and skip valid judgments for the requested model.

The original agent's quality policy accepts normal answers and nonempty token-limit forced answers. Empty answers, attacker generation failure, protocol failures, and max-call forced answers require retry or explicit review. Failed outcomes remain in the output with `metrics.retry_exhausted`; a final rate is withheld until the expected case × repeat grid is complete and all judgments are valid. This prevents infrastructure failures from silently becoming attack failures.

The result schema is `case_id`, `repeat_index` (one-based), `test_case`, `response`, `messages`, `metrics`, `config`, and `evaluation.asr`. A valid judgment contains boolean `success`, `status: ok`, and `judge_model`. Messages are also preserved under `metrics.messages` for trajectory compression. Outputs may contain the entire supplied case and generated content: keep them local.

ASR is the mean success rate within each family, then the equal-weight mean across families. MAXNASR computes whether each case succeeded at least once across N repeats, then averages family rates. SS, Flip and SF count cases with respectively all, some, or no successful repeats; overall counts are summed. The report also includes `pooled_ASR` and `pooled_MAXNASR` for comparison. Unequal family sizes make macro and pooled rates differ. No unjudged row is treated as a false judgment.

For strategy training, `breadcrumbing.runner.evaluate_strategy(cases, strategy_path, config, output_dir)` runs generation, judging and strict summarization, then returns a flat list of judged rows spanning all repeats. `cases` must include the `dataset` family label. The caller sets stage-specific cases and repeats and supplies a distinct output directory for each candidate/stage. The function retains full trajectories for compression and raises when the stage is incomplete.

Common settings can also be overridden without editing a YAML file:

```bash
python scripts/run_experiment.py --config configs/examples/demo.yaml \
  --data examples/test.jsonl --dataset-name example --method ach \
  --strategy strategies/ach.yaml --researcher-model qwen3 --attacker-model kimi \
  --judge-model qwen3 --workers 1 --repeats 1 --output-dir runs/model-override
```

Each explicit option replaces only that configuration field. `--config` is optional when the required dataset and model arguments are supplied directly. `--judge-model` records the intended evaluator; judging remains a separate explicit command. Summary prints a compact table and saves the full JSON report.

Set `prompt_date` to a quoted ISO string (for example, `"2026-01-02"`) to fix the date inserted into the researcher's system prompt. If omitted, generation resolves today's local date into the recorded configuration; resuming reuses that recorded date even on another day. `tokenizer_path` optionally selects a local Hugging Face tokenizer (install `transformers` separately); it is distinct from `researcher_model`. With no tokenizer configured, the runtime estimates tokens from character counts. This can change context-limit timing relative to the original experiments that used a local tokenizer. Record the tokenizer artifact/version when comparing those runs.

For per-family frozen strategies, use `strategy_map: {family: path/to/strategy.yaml}`. Every selected family must have an entry; missing families fail before any request. The manifest hashes each strategy, so changing an artifact invalidates resume. An explicit `--strategy` replaces the map with one shared strategy.

## Preparing Objective Drift memory

The frozen Objective Drift baseline requires a memory trained on your training split. A missing or malformed memory is an error. Preparation uses the original batch feedback procedure: each batch reads an immutable memory snapshot, rolls out its cases, receives checklist judgments and separate acceptance/detection judgments, then commits feedback serially before the next batch. The default is five cases per batch and one training repeat. The frozen snapshot strips training target/payload text from candidate feedback; all intermediate artifacts remain local and are not publication assets.

```bash
python scripts/prepare_objective_drift.py --config configs/baselines/objective_drift.yaml \
  --data path/to/train.jsonl --dataset-name Advertisement-1 \
  --batch-size 5 --repeats 1 --output-dir outputs/objective_drift_training
```

Copy the evaluation configuration to a local file and set:

```yaml
method: objective_drift
attacker_config:
  memory_file: outputs/objective_drift_training/frozen_memory.json
  memory_mode: read_only
```

Prepare and evaluate each task family separately with the same researcher configuration. The paper template evaluates one family with one `memory_file`; set `datasets` in your local copy to the trained family, or override it explicitly:

```bash
python scripts/run_experiment.py --config /path/to/objective-drift-eval.yaml \
  --data /path/to/test/Advertisement-1.jsonl --dataset-name Advertisement-1 \
  --output-dir outputs/objective_drift_test
python scripts/judge_results.py outputs/objective_drift_test --judge-model qwen3 --workers 1
python scripts/summarize_results.py outputs/objective_drift_test --judge-model qwen3
```

The preparation CLI consumes only its `--data` file; the template's `datasets` map does not select training data. The evaluation CLI consumes the configured map unless `--data` replaces it. Applying one family's memory to another family is a transfer experiment, not that family's trained baseline. A root lock protects preparation; batch snapshots and deduplicated feedback keys make resumption idempotent. Detection API failures stop feedback rather than being recorded as unnoticed content. No supplied training data or learned memory is included in this repository.

`researcher_max_tokens` optionally sets the researcher response token limit (default 10000). This is separate from `max_llm_calls_per_run`, the number of model turns allowed per rollout.

Interrupting generation, judging or memory preparation terminates worker processes and cancels queued work. Already checkpointed rows remain available for resume; in-flight calls may already have been billed by the provider.

## Configuration presets

`configs/ach.yaml` runs ACH. `configs/tgse.yaml` configures strategy training.
`configs/baselines/` contains unattacked, AgentDojo, GEO, SafeSearch, Objective
Drift, AiTM, MAST, Intent Hijacking, and Evo Attacker conditions.
`configs/evaluation/` evaluates each frozen TGSE regime.
`configs/tgse_adv_train.yaml` configures training with the cross-checking researcher. Prepared benchmark paths
are `data/safesearch/train/` and `data/safesearch/test/`.

`configs/ablations/` contains the three runtime removals and three strategy
substitutions from the paper. `ach_no_reflection.yaml` sets
`disable_reflection_render`: reflections are still generated but are omitted
from the planner's memory summary. `ach_no_memory.yaml` sets
`disable_memory_render` to omit the entire rendered memory summary while
retaining internal event/URL bookkeeping. `ach_no_strategy.yaml` sets
`disable_strategy` to omit strategy input. The substitution presets replace only
the strategy card with Dummy init, Goal reframing, or Directive embedding.

Frozen TGSE evaluations select `data/safesearch/tgse_test20/`. Ablation presets
and the AiTM, MAST, Intent Hijacking, and Evo Attacker baselines select
`data/safesearch/runtime_test20/`. The data script prepares these distinct
20-case held-out cohorts; see [data preparation](../data/README.md) for provenance and limitations.
ACH and the other baseline presets use the full test split. Provider deployments
and tool responses also affect numerical results.

Evo Attacker evaluation requires a frozen memory artifact. Run the
training preset, judge its results, and build the memory before evaluation:

```bash
python scripts/run_experiment.py --config configs/baselines/evo_attacker_train.yaml --output-dir runs/evo_train
python scripts/judge_results.py runs/evo_train --judge-model qwen3 --workers 1
python scripts/prepare_evo_memory.py --input-root runs/evo_train --dataset-dir data/safesearch/train --memory-id train --judge-llm qwen3
python scripts/run_experiment.py --config configs/baselines/evo_attacker.yaml --output-dir runs/evo_test
```

The training preset uses all 100 training cases once with empty memory.
The evaluation preset reads `cache/attacker/evo_attacker/train`. See
[baselines.md](baselines.md) for baseline preparation and configuration.
