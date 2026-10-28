# Optional runtime settings

Normal use needs only model/tool credentials and an experiment YAML. The settings below expose existing runtime limits for users who need to adjust them. They are optional; keep scientific settings fixed when comparing runs.

| Environment variable | Default | Purpose |
|---|---|---|
| `OPENAI_CLIENT_TIMEOUT_SECONDS` | 180 | Model HTTP request timeout; `LLM_CLIENT_TIMEOUT_SECONDS` is its fallback alias |
| `LLM_RETRY_MAX_ATTEMPTS` | 3 | Client transport attempts; task/planner retries are separate |
| `LLM_CALLS_WARN_THRESHOLD` | 5 | Remaining researcher calls at which to encourage a final answer |
| `ENABLE_VISIT_SUMMARY` | 0 | Use a model to summarize visited pages; otherwise return bounded page content |
| `VISIT_SERVER_TIMEOUT` | 200 | Reader request timeout in seconds |
| `VISIT_SERVER_MAX_RETRIES` | 1 | Page-summary attempts |
| `WEBCONTENT_MAXLENGTH` | 150000 | Maximum returned page characters when summary is disabled |
| `JINA_TOKEN_BUDGET_ENABLED` | 1 | Send a reader token budget |
| `JINA_TOKEN_BUDGET` | 45000 | Reader token budget |
| `JINA_RESPOND_WITH` | markdown | Reader output format |
| `JINA_RETAIN_IMAGES` | none | Reader image retention |
| `JINA_NO_CACHE` | 0 | Request reader cache bypass |
| `TGSE_COMPRESSION_MAX_TOKENS` | 20000 | Maximum model output for trace compression |
| `TGSE_COMPRESSION_WORKERS` | 4 | Compression concurrency cap, also bounded by requested workers |
| `BREADCRUMBING_CACHE_DIR` | `~/.cache/breadcrumbing` | Local raw tool-response cache |
| `BREADCRUMBING_STRATEGY_OUTPUT` | `outputs/strategies` | Output for the strategy-library save helper; TGSE uses its explicit output directory |
| `CACHE_SQLITE_TIMEOUT_SECONDS` | 30 | SQLite connection timeout |
| `CACHE_SQLITE_BUSY_TIMEOUT_MS` | 30000 | SQLite busy wait |
| `CACHE_SQLITE_MAX_RETRIES` | 5 | SQLite write retries |
| `CACHE_SQLITE_RETRY_DELAY_SECONDS` | 0.2 | Delay between SQLite retries |

For endpoint, key, alias and reasoning variables, see [models.md](models.md). Shell environment variables are local to the process; they are not automatically captured in experiment YAML. Record any overrides alongside results. A worker limit bounds simultaneous tasks, not requests per second or total monetary cost.

## Experiment YAML layout

| Path | Purpose |
|---|---|
| `configs/ach.yaml` | ACH on the prepared held-out split |
| `configs/tgse.yaml` | TGSE training settings |
| `configs/tgse_adv_train.yaml` | TGSE training with the cross-checking researcher |
| `configs/baselines/` | Unattacked and baseline attack conditions |
| `configs/evaluation/` | Frozen TGSE evaluation |
| `configs/ablations/` | Paper runtime removals and fixed strategy substitutions |
| `configs/examples/` | Small demonstration tasks and training settings |

YAML files are concrete configurations; no inheritance or implicit merging is
used. Run from the repository root so strategy and dataset paths resolve.
Prepare benchmark files with `scripts/prepare_data.py` before using benchmark
presets. The examples use `examples/tasks.jsonl` and can be selected separately.
Each attacker receives the supplied case context directly. Attacks that use
query recognition match normalized exact strings.

See [experiments.md](experiments.md) for the runtime ablation semantics and
cohort requirements, and [models.md](models.md) for model aliases and endpoints.
