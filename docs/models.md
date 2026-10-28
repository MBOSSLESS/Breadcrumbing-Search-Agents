# Model configuration

All experiment roles share [`MODEL_MAP` and `BACKEND_CONFIG`](../breadcrumbing/models/config.py). A role selects a model; the model resolves its backend and endpoint.

```text
qwen3 → openrouter:qwen/qwen3-235b-a22b-2507 → OpenRouter client
```

Use an existing alias, add an alias to `MODEL_MAP`, or pass `backend:model` directly. A local model path resolves to the vLLM backend and identifies an already served model; this does not load weights or start a server.

| Backend | Key variable | Endpoint override |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| `oai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `vllm` | unused placeholder | `VLLM_BASE_URL` |
| `ollama` | unused placeholder | `OLLAMA_BASE_URL` |

The default endpoints are public service URLs or localhost. No private project gateway is included. Keep custom credentials and endpoints local. `.env.example` is a reference, not an automatically loaded configuration file.

Aliases may also be overridden without editing code:

```bash
export BREADCRUMBING_MODEL_QWEN3='oai:your-served-model-id'
export OPENAI_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='<your key>'
```

The Qwen3 example requests reasoning effort `none`, matching the existing runtime setting. `BREADCRUMBING_REASONING_EFFORT` overrides the default; `OPENROUTER_DISABLE_REASONING_POLICY=1` omits it. Explicit per-call reasoning settings take precedence. These affect model behavior and should be recorded when comparing experiments.

The `oai` backend uses structured response parsing; OpenRouter/local examples use text JSON parsing. Configure backend capabilities for the service you use. An OpenAI-compatible label alone does not guarantee support for every generation parameter or structured-output format.

## Tools and caching

Search requires `SERPER_KEY_ID`; Visit uses `JINA_API_KEYS`. Page summarization, when enabled, also makes model calls. The tool implementation documents its optional reader limits. Default caches live under `~/.cache/breadcrumbing`; override with `BREADCRUMBING_CACHE_DIR`. Caches are local artifacts and are not needed to import the package.

If your environment uses a SOCKS proxy, install its optional transport with `pip install -e ".[proxy]"`. A tokenizer can be installed with `pip install -e ".[tokenizer]"`; set `tokenizer_path` explicitly in the experiment configuration.

Additional reader, retry, cache and compression limits are listed in [configuration.md](configuration.md).

The researcher and ACH planner use their prompted text ReAct/JSON formats. When a caller has not declared native `tools` or `functions`, the client converts native function calls into the existing `<tool_call>` text format. Function names and argument values are preserved; arguments must decode to JSON objects. Existing text tool-call tags are left unchanged to avoid duplicate calls. Calls that declare native tools retain their native responses. The ACH planner also accepts a complete bare or fenced JSON object for its three strategy tools. The downstream agent or planner still applies its existing tool validation and execution limits; endpoint compatibility should be checked with the intended model.
