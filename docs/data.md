# Task data

Supply UTF-8 JSONL, one task per line. The original benchmark data is not included. The files under `examples/` were written for this release and use harmless educational questions and an unrelated output marker.

| Field | Meaning |
|---|---|
| `test_case_id` | Stable unique string or integer within the family |
| `dataset` | Task-family label; supplied by the configuration when omitted, and must match it when present |
| `user_query` | Legitimate user request passed to the researcher |
| `user_expectation` | Expected legitimate answer or behavior |
| `target_consequence` | Outcome the attacker attempts to induce |
| `risk_verification_checklist` | Nonempty string or list of nonempty strings used by the judge |
| `website_generation_guideline` | Additional page-generation guidance, used by SafeSearch |
| `rationales_behind_unreliable_website_crafting` | Optional page-crafting rationale supplied to ACH planning |
| `sources_of_unreliable_website` | Optional source/context metadata |
| `risk_type` | Optional descriptive family/category |

See [`examples/tasks.jsonl`](../examples/tasks.jsonl) for a complete record. The attacker receives the case directly; no hidden CSV lookup is required.

Use disjoint train and test IDs. TGSE samples its training slice once, uses only training feedback for strategy updates, and selects by full training ASR. The optional test command rejects overlapping IDs. For independent evaluation with `run_experiment.py`, the caller supplies the correct held-out split.

Outputs retain task text, final answers, and trajectories to support evaluation and compression. They are written to the caller's output directory, which should remain private. Strategy publication does not imply that those outputs or the input dataset are distributable.
