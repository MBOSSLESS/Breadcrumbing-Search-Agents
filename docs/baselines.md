# Adapted baselines

These four methods use the same Search/Visit intervention surface as the main
comparison. They append at most one Search result per event and serve pages
only for URLs registered by that attacker instance. Original results remain
unchanged. Each case/repeat creates a fresh instance from its complete task row.

| Method key | Retained mechanism | Adaptation scope |
| --- | --- | --- |
| `aitm` | Recursive generation using current Search context and the preceding instruction | Paper-guided reconstruction of communication tampering through web assets |
| `mast` | Intermediate subgoal planning followed by intervention realization using partial public history | Inference scaffold; no MCTS, DPO, or learned policy training |
| `intent_hijacking` | One fixed seven-stage strategy with event-specific web realization | AgentLAB adaptation; no online judge, TextGrad, rollback, or strategy restart |
| `evo_attacker` | Retrieval query, semantic top-five memory retrieval, reflection, decision, independent Modifier | No Attack-Flow GRPO or policy training; frozen success-only training memory for evaluation |

All use attacker temperature `0.5`. The researcher uses temperature `0.0` and
at most ten calls in the paper setting. Configure an actual model endpoint
through the normal model configuration. These are adaptations, not complete
reproductions of the original training pipelines.

## Evo memory preparation

Install the optional Evo dependencies with `pip install -e '.[evo]'`.
Run one empty-memory training trajectory per case, judge the completed results,
and freeze the successful episodes:

```sh
python scripts/run_experiment.py \
  --config configs/baselines/evo_attacker_train.yaml \
  --output-dir outputs/evo_train
python scripts/judge_results.py outputs/evo_train --judge-model qwen3 --workers 8
python scripts/prepare_evo_memory.py \
  --input-root outputs/evo_train \
  --dataset-dir data/safesearch/train \
  --memory-id train \
  --judge-llm qwen3
```

The default build requires 100 unique training cases, five families of 20,
usable empty-memory training trajectories, and complete judgments. A smaller
controlled pilot can set `--expected-cases`; use a separate artifact directory.
The builder never runs attacker or researcher models. It loads
`all-MiniLM-L6-v2` to encode bounded successful-episode views and freezes
normalized embeddings, checksums, and provenance. The first embedding use may
download the encoder. No real memories, trajectories, or benchmark rows ship
with this repository.

For held-out evaluation set `memory_mode: frozen` and `memory_path` to the
artifact directory. The runtime verifies checksums and uses actual normalized
cosine retrieval. Memory stays read-only; each event permits at most one
additional retrieval through `CONTINUE`. Organic Visit bodies never enter
this policy or its memory. Memory artifacts contain attack payloads and should
be kept in the user's experiment output directory.

## Prompt provenance

Intent Hijacking preserves AgentLAB's seven-stage prompt material. The source
is [AgentLAB](https://github.com/TanqiuJiang/AgentLAB), commit
`dc01e0cc210c1ad6d892df5b5958d713503f270c`; its MIT notice is included in
[`licenses/AgentLAB-MIT.txt`](../licenses/AgentLAB-MIT.txt).
