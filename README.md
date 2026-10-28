<p align="center">
  <img src="assets/banner-user-80aef8a2.png" width="900" alt="Breadcrumbing Search Agents: a hooded attacker scatters golden breadcrumbs along a path explored by a search agent holding a magnifying glass." />
</p>

<h1 align="center">Breadcrumbing Search Agents: Per-Turn Scheming Over Long-Horizon Trajectories</h1>
<p align="center"><strong>NeurIPS 2026</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.04565"><img src="assets/badges/paper.svg" alt="Paper: arXiv 2608.04565" /></a>
  <a href="#quickstart"><img src="assets/badges/python.svg" alt="Python 3.10 or newer" /></a>
  <a href="LICENSE"><img src="assets/badges/license.svg" alt="License: Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#results">Results</a> ·
  <a href="#reproduction">Reproduction</a> ·
  <a href="#data-and-strategies">Data &amp; Strategies</a> ·
  <a href="#citation">Citation</a>
</p>

---

<a id="overview"></a>
## Overview

Search agents gather evidence through successive queries, page visits, and cross-checks. We study how an attacker can coordinate small interventions across this trajectory to steer the agent's final answer.

We study a constrained tool-intermediary attack: the attacker can inject **at most one attacker-controlled result per query** and serve controlled content for **attacker-introduced URLs**, while leaving organic results and visits unchanged. Our evaluation simulates these interventions at the tool-return interface.

<p align="center">
  <a href="assets/runtime.pdf"><img src="assets/runtime.png" width="900" alt="Search agent, search and visit services, and a stateful attacker with memory, strategy and planner. Controlled content is restricted to the appended result and attacker-introduced URLs." /></a>
</p>

*A strategy-guided planner generates each payload based on the agent's trajectory.*

**Authority-Chain Hijack (ACH)** is an expert-refined strategy that coordinates successive search results and landing pages into a chain of seemingly corroborating evidence. The attacker uses trajectory memory and a strategy-guided planner to adapt each intervention.

**Trace-Guided Strategy Evolution (TGSE)** improves attacker strategies from successful and failed training trajectories. It explores an archive of candidates, evaluates promising strategies in stages, and freezes a training-selected strategy for held-out evaluation.

<p align="center">
  <a href="assets/attack-trajectory.pdf"><img src="assets/attack-trajectory.png" width="900" alt="Illustrative water-filter trajectories: independent verification supports a balanced shortlist without attack and overturns static promotion. Under adaptive attack, ACH supplies a linked report claiming newer tests to sustain the target recommendation." /></a>
</p>

*Independent verification overturns the static promotion, while ACH adapts the linked report to sustain the target recommendation.*

This repository provides the DeepResearch-style agent, ACH, TGSE, paper comparison baselines, frozen strategies, SafeSearch split indices, and checklist evaluation.

<a id="quickstart"></a>
## Quickstart

Run a small synthetic example, then judge and summarize its output. The example uses one case, one repeat, and one worker; it requires no SafeSearch download. Model and Search/Visit service calls incur usage charges.

### 1. Install

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The checked environment uses Python 3.10 on Linux. Inference uses model APIs, so the default example needs no local GPU. Dependency versions are recorded in [`pyproject.toml`](pyproject.toml).

### 2. Configure models and tools

```bash
export OPENROUTER_API_KEY='<your model key>'
export SERPER_KEY_ID='<your search key>'
export JINA_API_KEYS='<your reader key>'
```

The example uses the `qwen3` model alias, Serper for Search, and Jina for Visit. Set credentials in your shell; the runtime does not load `.env` automatically. See [model configuration](docs/models.md) for aliases, other OpenAI-compatible endpoints, local model servers, and reasoning settings.

### 3. Run, judge, and summarize

```bash
python scripts/run_experiment.py \
  --config configs/examples/demo.yaml \
  --output-dir runs/demo

python scripts/judge_results.py runs/demo --judge-model qwen3 --workers 1
python scripts/summarize_results.py runs/demo --judge-model qwen3
```

The example tasks are newly constructed educational questions with a harmless output marker. Outputs include `manifest.json`, `results.jsonl`, and `summary.json` under `runs/demo/`. The summary reports **ASR, MaxN ASR, and SS / Flip / SF**: cases with success on every repeat, mixed outcomes, or failure on every repeat. A one-repeat demo cannot exhibit mixed outcomes.

Run generation and judging in sequence. Repeating the generation command resumes the same run; use a new output directory when changing scientific settings.

<details>
<summary><strong>Change the model or run without an attacker</strong></summary>

```bash
python scripts/run_experiment.py \
  --config configs/examples/demo.yaml \
  --method no_attack --researcher-model kimi \
  --output-dir runs/no-attack
```

Researcher, attacker, and judge models can be selected independently. Available methods are `no_attack`, `ach`, `agentdojo`, `geo`, `safesearch`, `objective_drift`, `aitm`, `mast`, `intent_hijacking`, and `evo_attacker`. Objective Drift and Evo-Attacker need training-derived frozen memory; Evo-Attacker also needs `pip install -e ".[evo]"`. See [experiment interfaces](docs/experiments.md) and [baseline adaptations](docs/baselines.md).

</details>

<a id="results"></a>
## Selected results

### ACH across search-agent models

ACH attains the highest Overall ASR among the fixed attacks compared in the paper across six victim models. With Qwen3 as the victim, it reaches **55.9% ASR / 83.3% MaxN ASR** on the full 187-case held-out split, with five repeats per case.

<p align="center">
  <img src="assets/main-results.svg" width="760" alt="Paper results across six victim models: ACH has the highest Overall ASR among the compared fixed attacks, with gains of 13.0 to 36.1 percentage points over the best non-ACH attack." />
</p>

*Overall ASR is the macro-average across the five SafeSearch task families. The right panel reports absolute ASR gains in percentage points.*

### TGSE improves the frozen strategy

On the separate TGSE evaluation subset—20 held-out cases per family, five repeats per case—Trace-warm TGSE improves ASR from **56.0% to 71.4%**, a gain of **15.4 percentage points** over the matched ACH reference.

| Configuration | ASR (%) | MaxN ASR (%) |
|---|---:|---:|
| ACH reference | 56.0 | 85.0 |
| Normal TGSE | 70.0 | 93.0 |
| **Trace-warm TGSE** | **71.4** | **95.0** |
| Adv-train TGSE | 69.6 | 90.0 |
| Adv-warm TGSE | 71.0 | 95.0 |

All rows report family macro-averages. ASR measures successful trajectories; MaxN ASR counts a case as successful if any of its five repeats succeeds. These are reported paper results. The [reproduction guide](docs/reproduction.md) maps each experiment to its cohort, configuration, and strategy artifacts.

<a id="reproduction"></a>
## Reproducing the paper

Prepare the benchmark data from the pinned upstream source:

```bash
python scripts/prepare_data.py
```

This checks source hashes and reconstructs the published splits: **100 training cases**, **187 held-out cases**, and two recorded 100-case evaluation subsets. `runtime_test20` and `tgse_test20` contain different task IDs; select the cohort used by the experiment being reproduced.

| Experiment | Configuration / artifact | Guide |
|---|---|---|
| ACH on the full held-out split | [`configs/ach.yaml`](configs/ach.yaml) | [Run and evaluate](docs/reproduction.md) |
| Paper comparison baselines | [`configs/baselines/`](configs/baselines/) | [Adaptations and memory preparation](docs/baselines.md) |
| Frozen TGSE strategies | [`configs/evaluation/`](configs/evaluation/) | [Strategy artifacts](strategies/README.md) |
| Rerun strategy evolution | [`configs/tgse.yaml`](configs/tgse.yaml) | [TGSE procedure](docs/tgse.md) |
| Runtime and strategy ablations | [`configs/ablations/`](configs/ablations/) | [Experiment settings](docs/experiments.md) |

For the full ACH comparison:

```bash
python scripts/run_experiment.py --config configs/ach.yaml --output-dir runs/ach
python scripts/judge_results.py runs/ach --judge-model qwen3 --workers 1
python scripts/summarize_results.py runs/ach --judge-model qwen3
```

Report the cohort, repeats, model identities, settings, and technical failures alongside the metrics. Hosted models and web content can change between runs; the [reproduction guide](docs/reproduction.md) records the settings needed to interpret a rerun.

<details>
<summary><strong>Try the TGSE workflow on synthetic tasks</strong></summary>

```bash
python scripts/evolve_strategy.py \
  --config configs/examples/tgse.yaml \
  --train-data examples/train.jsonl \
  --strategy strategies/ach.yaml \
  --output-dir runs/tgse-demo \
  --test-data examples/test.jsonl --test-cases 1
```

This small example runs the evolution workflow with separate training and test tasks. TGSE evaluates candidate strategies on a cheap training subset, completes selected candidates' evaluation, and updates its archive and lessons. It selects the strategy with the highest full-training ASR, freezes it, and then evaluates held-out tasks. Held-out feedback never enters strategy evolution. See [TGSE](docs/tgse.md) for the full procedure and resumption rules.

</details>

<a id="data-and-strategies"></a>
## Data and strategies

| Resource | Contents |
|---|---|
| [SafeSearch split indices](data/README.md) | Pinned source revision, checksums, training/test indices, and recorded evaluation subsets. Task records are retrieved from upstream. |
| [Strategies](strategies/README.md) | ACH, 20 frozen TGSE strategies across four settings and five families, 10 warm-start seeds, and strategy-substitution cards. |
| [Strategy manifest](strategies/manifest.json) | Hashes for the released strategy artifacts. |
| [Synthetic examples](examples/) | Small educational tasks for exercising the runtime and evolution workflow. |

<a id="documentation"></a>
## Documentation and code

| Start here | What it covers |
|---|---|
| [Experiments and metrics](docs/experiments.md) | CLI overrides, outputs, resumption, judging, and failure handling |
| [Models and services](docs/models.md) | Model aliases, endpoints, credentials, and reasoning settings |
| [TGSE](docs/tgse.md) | Candidate generation, staged evaluation, archive selection, and held-out evaluation |
| [Baseline adaptations](docs/baselines.md) | Comparison methods and their implementation scope |
| [Runtime configuration](docs/configuration.md) | Optional limits and environment variables |

<details>
<summary><strong>Repository map</strong></summary>

```text
breadcrumbing/
  agent/          Research agent and Search/Visit tools
  attacks/        ACH and paper comparison baselines
  models/         Shared model aliases and API client
  tgse/           Evolution, feedback, compression, and selection
  evaluation/     Checklist judging and repeat metrics
  runner.py       Bounded concurrent task execution
scripts/          Prepare data, run, evolve, judge, and summarize
configs/          Small examples and paper experiment configurations
strategies/       Initial, frozen, and ablation strategies
data/             Source provenance and task split indices
examples/         Synthetic task records
```

</details>

<a id="citation"></a>
## Citation

```bibtex
@inproceedings{li2026breadcrumbing,
  title={Breadcrumbing Search Agents: Per-Turn Scheming Over Long-Horizon Trajectories},
  author={Li, Xuebin and Zhao, Hanqing and Liang, Siyuan and Chen, Kejiang and
          Zhang, Weiming and Tao, Dacheng and Yu, Nenghai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Attribution and license

Licensed under [Apache-2.0](LICENSE). Our agent builds on Tongyi DeepResearch. Third-party attribution and retained license notices are listed in [NOTICE.md](NOTICE.md).
